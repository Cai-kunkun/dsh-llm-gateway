/**
 * Standalone self-test that runs without a harness host: wire translation,
 * chunk folding, and end-to-end HTTP behavior against a stub llm service.
 *
 * Covered:
 *   - text streaming and non-streaming
 *   - tool calls (incremental deltas and block-end-only paths)
 *   - reasoning deltas to reasoning_content
 *   - usage folding
 *   - error finish to HTTP status mapping
 *   - unknown provider to NO_ADAPTER
 *   - authentication (loopback admission, wrong Bearer rejection, correct key)
 *
 * Run: node test/selftest.mjs
 */

import assert from 'node:assert/strict'
import { startGatewayServer, parseRoute } from '../lib/server.js'
import { fromOpenAiMessages, fromOpenAiTools, createOpenAiFolder, foldChunksToOpenAiStream, toOpenAiUsage, failureOf } from '../lib/wire.js'

const passed = []
const failed = []

async function test(name, fn) {
  try {
    await Promise.race([
      fn(),
      new Promise((_, reject) => setTimeout(() => reject(new Error(`TEST TIMEOUT (8s) — hung here`)), 8000).unref?.()),
    ])
    passed.push(name)
    console.log(`  ok  ${name}`)
  } catch (error) {
    failed.push({ name, error })
    console.error(`FAIL  ${name}`)
    console.error(error)
  }
}

// ---------------------------------------------------------------- wire translation
await test('fromOpenAiMessages: string content and role mapping', async () => {
  const { messages } = await fromOpenAiMessages([
    { role: 'system', content: 'you are helpful' },
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'hello', reasoning_content: 'thoughts' },
    { role: 'user', content: 'bye' },
  ])
  assert.equal(messages.length, 4)
  assert.deepEqual(messages[0], { role: 'system', content: [{ type: 'text', text: 'you are helpful' }] })
  assert.deepEqual(messages[1], { role: 'user', content: [{ type: 'text', text: 'hi' }] })
  assert.equal(messages[2].content[0].type, 'reasoning')
  assert.equal(messages[2].content[1].text, 'hello')
  assert.equal(messages[2].source.kind, 'model')
})

await test('fromOpenAiMessages: tool_calls and tool results', async () => {
  const { messages } = await fromOpenAiMessages([
    { role: 'assistant', content: '', tool_calls: [{ id: 'c1', function: { name: 'get_weather', arguments: '{"city":"SF"}' } }] },
    { role: 'tool', tool_call_id: 'c1', content: 'sunny' },
  ])
  assert.equal(messages.length, 2)
  assert.equal(messages[0].content[0].type, 'tool-call')
  assert.equal(messages[0].content[0].id, 'c1')
  assert.equal(messages[0].content[0].arguments, '{"city":"SF"}')
  assert.equal(messages[1].role, 'tool')
  assert.equal(messages[1].toolCallId, 'c1')
  assert.equal(messages[1].content[0].text, 'sunny')
})

await test('fromOpenAiMessages: empty messages and empty content are skipped', async () => {
  const { messages } = await fromOpenAiMessages([
    { role: 'user', content: '' },
    { role: 'user', content: [] },
    { role: 'user', content: 'ok' },
  ])
  assert.equal(messages.length, 1)
})

await test('fromOpenAiMessages: unknown content part degrades to JSON text', async () => {
  const { messages } = await fromOpenAiMessages([
    { role: 'user', content: [{ type: 'audio', foo: 'bar' }] },
  ])
  assert.equal(messages.length, 1)
  assert.equal(messages[0].content.length, 1)
  assert.equal(messages[0].content[0].type, 'text')
})

await test('fromOpenAiMessages: image data URL degrades without attachments', async () => {
  const { messages, imageFailures } = await fromOpenAiMessages([
    { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' } }] },
  ])
  assert.equal(messages[0].content[0].type, 'text')
  assert.equal(imageFailures.length, 1)
})

await test('fromOpenAiMessages: image data URL admitted through saveImage', async () => {
  const saved = []
  const { messages } = await fromOpenAiMessages([
    { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/jpeg;base64,/9j/4AAQ' } }] },
  ], { saveImage: async (input) => { saved.push(input); return { attachmentId: 'a1', mediaType: 'image/jpeg', bytes: 3, width: 1, height: 1 } } })
  assert.equal(messages[0].content[0].type, 'image')
  assert.equal(messages[0].content[0].attachment.attachmentId, 'a1')
  assert.equal(saved.length, 1)
  assert.equal(saved[0].mediaType, 'image/jpeg')
  assert.ok(saved[0].data.length > 0)
})

await test('fromOpenAiMessages: system and developer fold to system', async () => {
  const { messages } = await fromOpenAiMessages([
    { role: 'system', content: 'a' },
    { role: 'developer', content: 'b' },
  ])
  assert.equal(messages.length, 2)
  assert.equal(messages[0].role, 'system')
  assert.equal(messages[1].role, 'system')
})

await test('fromOpenAiTools: nested and flat spellings', async () => {
  const tools = fromOpenAiTools([
    { type: 'function', function: { name: 'f1', description: 'd', parameters: { type: 'object' } } },
    { name: 'f2', parameters: { type: 'object' } },
  ])
  assert.equal(tools.length, 2)
  assert.equal(tools[0].name, 'f1')
  assert.equal(tools[1].name, 'f2')
  assert.equal(fromOpenAiTools([]), undefined)
  assert.equal(fromOpenAiTools(undefined), undefined)
})

// ---------------------------------------------------------------- chunk folding
function textStream(text, opts = {}) {
  const chunks = [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    ...(opts.usage ? [{ type: 'usage', usage: opts.usage }] : []),
    { type: 'finish', reason: { kind: 'stop' } },
  ]
  return chunks
}

await test('foldChunksToOpenAiStream: text deltas and finish', () => {
  const folded = foldChunksToOpenAiStream(textStream('hello'), { id: 'x', model: 'm', created: 1 })
  const contents = folded.events.map((event) => event.delta?.content).filter(Boolean)
  assert.equal(contents.join(''), 'hello')
  assert.equal(folded.finishReason, 'stop')
})

await test('foldChunksToOpenAiStream: reasoning deltas', () => {
  const chunks = [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'think' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'think' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
  const folded = foldChunksToOpenAiStream(chunks, { id: 'x', model: 'm', created: 1 })
  assert.equal(folded.events[0].delta.reasoning_content, 'think')
})

await test('foldChunksToOpenAiStream: tool-call incremental path', () => {
  const chunks = [
    { type: 'block-start', index: 1, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 1, id: 'call_9', name: 'get', argumentsDelta: '{"a"' },
    { type: 'tool-call-delta', index: 1, argumentsDelta: ':1}' },
    { type: 'block-end', index: 1, block: { type: 'tool-call', id: 'call_9', name: 'get', arguments: '{"a":1}' } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ]
  const folded = foldChunksToOpenAiStream(chunks, { id: 'x', model: 'm', created: 1 })
  assert.equal(folded.finishReason, 'tool_calls')
  assert.equal(folded.hasToolCalls, true)
  const calls = folded.events.flatMap((event) => event.delta?.tool_calls ?? [])
  const name = calls.find((call) => call.function?.name)?.function.name
  assert.equal(name, 'get')
  const args = calls.map((call) => call.function?.arguments ?? '').join('')
  assert.equal(args, '{"a":1}')
  assert.ok(calls.some((call) => call.id === 'call_9'))
})

await test('foldChunksToOpenAiStream: tool-call at block-end only', () => {
  const chunks = [
    { type: 'block-start', index: 2, blockType: 'tool-call' },
    { type: 'block-end', index: 2, block: { type: 'tool-call', id: 'call_5', name: 'lookup', arguments: '{"q":"x"}' } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ]
  const folded = foldChunksToOpenAiStream(chunks, { id: 'x', model: 'm', created: 1 })
  const calls = folded.events.flatMap((event) => event.delta?.tool_calls ?? [])
  const name = calls.find((call) => call.function?.name)?.function.name
  assert.equal(name, 'lookup')
  const args = calls.map((call) => call.function?.arguments ?? '').join('')
  assert.equal(args, '{"q":"x"}')
})

await test('toOpenAiUsage: counters and cache details', () => {
  const usage = toOpenAiUsage({ inputTokens: 10, outputTokens: 5, cacheReadTokens: 7, reasoningTokens: 3, totalTokens: 15 })
  assert.equal(usage.prompt_tokens, 10)
  assert.equal(usage.completion_tokens, 5)
  assert.equal(usage.total_tokens, 15)
  assert.equal(usage.prompt_tokens_details.cached_tokens, 7)
})

await test('failureOf: status mapping', () => {
  assert.equal(failureOf({ reason: { kind: 'error', failure: { message: 'nope', code: 'NO_ADAPTER' } } }).status, 404)
  assert.equal(failureOf({ reason: { kind: 'error', failure: { message: 'rate', code: 'RATE_LIMIT', status: 429 } } }).status, 429)
})

// ---------------------------------------------------------------- HTTP end to end
function fakeLlm(streamImpl) {
  return {
    listProviders: () => [{ id: 'fake', name: 'Fake Provider' }, { id: 'empty', name: 'Empty' }],
    listModels: async (provider) => provider === 'fake'
      ? [{ provider, id: 'm1', name: 'Model One' }, { provider, id: 'm2', name: 'Model Two' }]
      : [],
    stream: streamImpl,
    get: () => undefined,
  }
}

function post(port, path, body, headers = {}) {
  return fetch(`http://127.0.0.1:${port}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  })
}

async function withServer(depsConfig, fn) {
  const handle = startGatewayServer({
    getLlm: () => globalThis.__fakeLlm,
    getService: () => undefined,
    config: { port: 0, host: '127.0.0.1', apiKey: null, imageMode: 'auto', log: () => {}, ...depsConfig },
  })
  await handle.ready
  const port = handle.address().port
  try {
    return await fn(port, handle)
  } finally {
    await handle.dispose()
  }
}

// Each test assigns globalThis.__fakeLlm before calling withServer.

await test('HTTP: /healthz', async () => {
  await withServer({}, async (port) => {
    const response = await fetch(`http://127.0.0.1:${port}/healthz`)
    assert.equal(response.status, 200)
    const body = await response.json()
    assert.equal(body.ok, true)
  })
})

await test('HTTP: /v1/models aggregates providers', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () { yield { type: 'finish', reason: { kind: 'stop' } } })
  await withServer({}, async (port) => {
    const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
    assert.equal(response.status, 200)
    const body = await response.json()
    assert.equal(body.data.length, 2)
    assert.equal(body.data[0].id, 'fake/m1')
  })
})

await test('HTTP: non-streaming completion (text)', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* (options) {
    assert.equal(options.provider, 'fake')
    assert.equal(options.model, 'm1')
    assert.equal(options.messages[0].role, 'system')
    assert.equal(options.messages[1].content[0].text, 'hi')
    yield* textStream('hello world', { usage: { inputTokens: 3, outputTokens: 2 } })
  })
  await withServer({}, async (port) => {
    const response = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'system', content: 'sys' }, { role: 'user', content: 'hi' }] })
    assert.equal(response.status, 200)
    const body = await response.json()
    assert.equal(body.choices[0].message.content, 'hello world')
    assert.equal(body.choices[0].finish_reason, 'stop')
    assert.equal(body.usage.total_tokens, 5)
  })
})

await test('HTTP: streaming completion (SSE)', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () {
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: 'A' }
    yield { type: 'text-delta', index: 0, text: 'B' }
    yield { type: 'block-end', index: 0, block: { type: 'text', text: 'AB' } }
    yield { type: 'usage', usage: { inputTokens: 1, outputTokens: 2 } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  })
  await withServer({}, async (port) => {
    const response = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }], stream: true, stream_options: { include_usage: true } })
    assert.equal(response.status, 200)
    assert.match(response.headers.get('content-type') ?? '', /text\/event-stream/)
    const raw = await response.text()
    const events = raw.split('\n').filter((line) => line.startsWith('data: ')).map((line) => line.slice(6))
    assert.equal(events[events.length - 1], '[DONE]')
    const parsed = events.slice(0, -1).map((text) => JSON.parse(text))
    const withChoices = parsed.filter((row) => Array.isArray(row.choices) && row.choices.length > 0)
    const last = withChoices[withChoices.length - 1]
    const roleChunk = withChoices.find((row) => row.choices[0].delta?.role === 'assistant')
    assert.ok(roleChunk, 'first delta should carry the assistant role')
    const deltas = withChoices.filter((row) => row.choices[0].delta?.content).map((row) => row.choices[0].delta.content).join('')
    assert.equal(deltas, 'AB')
    assert.equal(last.choices[0].finish_reason, 'stop')
    assert.ok(parsed.some((row) => row.usage?.total_tokens === 3))
  })
})

await test('HTTP: first SSE frame is readable incrementally', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () {
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: 'early' }
    await new Promise((resolve) => setTimeout(resolve, 400))
    yield { type: 'text-delta', index: 0, text: 'late' }
    yield { type: 'finish', reason: { kind: 'stop' } }
  })
  await withServer({}, async (port) => {
    const started = Date.now()
    const response = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }], stream: true })
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let firstAt = null
    let received = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (firstAt === null && decoder.decode(value, { stream: true }).includes('early')) firstAt = Date.now()
      received += decoder.decode(value, { stream: true })
    }
    assert.ok(firstAt !== null, 'should receive first delta before stream end')
    // The first delta must arrive well before the stream ends (second chunk is delayed 400ms).
    assert.ok(received.includes('early') && received.includes('late'))
    assert.ok(firstAt - started < 400, `first delta arrived at ${firstAt - started}ms, expected incremental streaming`)
  })
})

await test('HTTP: non-streaming tool calls', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () {
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: 'let me check' }
    yield { type: 'block-end', index: 0, block: { type: 'text', text: 'let me check' } }
    yield { type: 'block-start', index: 1, blockType: 'tool-call' }
    yield { type: 'tool-call-delta', index: 1, id: 'c1', name: 'get_weather', argumentsDelta: '{"city":"SF"}' }
    yield { type: 'block-end', index: 1, block: { type: 'tool-call', id: 'c1', name: 'get_weather', arguments: '{"city":"SF"}' } }
    yield { type: 'finish', reason: { kind: 'tool-calls' } }
  })
  await withServer({}, async (port) => {
    const response = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'weather?' }] })
    const body = await response.json()
    assert.equal(body.choices[0].finish_reason, 'tool_calls')
    const calls = body.choices[0].message.tool_calls
    assert.equal(calls.length, 1)
    assert.equal(calls[0].function.name, 'get_weather')
    assert.equal(calls[0].function.arguments, '{"city":"SF"}')
    assert.equal(calls[0].id, 'c1')
    assert.equal(body.choices[0].message.content, 'let me check')
  })
})

await test('HTTP: error finish maps to HTTP status', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () {
    yield { type: 'finish', reason: { kind: 'error', failure: { message: 'no key', code: 'MISSING_CREDENTIAL' } } }
  })
  await withServer({}, async (port) => {
    const response = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }] })
    assert.equal(response.status, 401)
    const body = await response.json()
    assert.equal(body.error.code, 'MISSING_CREDENTIAL')
  })
})

await test('HTTP: unknown provider returns 404 NO_ADAPTER', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () {
    yield { type: 'finish', reason: { kind: 'error', failure: { message: 'unknown provider "nope"', code: 'NO_ADAPTER' } } }
  })
  await withServer({}, async (port) => {
    const response = await post(port, '/v1/chat/completions', { model: 'nope/m1', messages: [{ role: 'user', content: 'hi' }] })
    assert.equal(response.status, 404)
    const body = await response.json()
    assert.equal(body.error.code, 'NO_ADAPTER')
  })
})

await test('HTTP: apiKey authentication (missing, wrong, correct)', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () { yield* textStream('ok') })
  await withServer({ apiKey: 'sekrit' }, async (port) => {
    const noAuth = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }] })
    assert.equal(noAuth.status, 401)
    const badAuth = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }] }, { authorization: 'Bearer wrong' })
    assert.equal(badAuth.status, 401)
    const good = await post(port, '/v1/chat/completions', { model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }] }, { authorization: 'Bearer sekrit' })
    assert.equal(good.status, 200)
    const models = await fetch(`http://127.0.0.1:${port}/v1/models`)
    assert.equal(models.status, 401)
    const modelsOk = await fetch(`http://127.0.0.1:${port}/v1/models`, { headers: { authorization: 'Bearer sekrit' } })
    assert.equal(modelsOk.status, 200)
  })
})

await test('HTTP: unknown route 404, empty model 400, bad JSON 400', async () => {
  globalThis.__fakeLlm = fakeLlm(async function* () { yield* textStream('x') })
  await withServer({}, async (port) => {
    const notFound = await fetch(`http://127.0.0.1:${port}/v1/nope`)
    assert.equal(notFound.status, 404)
    const noModel = await post(port, '/v1/chat/completions', { messages: [{ role: 'user', content: 'hi' }] })
    assert.equal(noModel.status, 400)
    const badRoute = await post(port, '/v1/chat/completions', { model: 'noseparator', messages: [{ role: 'user', content: 'hi' }] })
    assert.equal(badRoute.status, 400)
    const badJson = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{oops' })
    assert.equal(badJson.status, 400)
  })
})

await test('HTTP: client disconnect aborts the upstream stream', async () => {
  let sawAbort = false
  globalThis.__fakeLlm = fakeLlm(async function* (options) {
    try {
      options.signal?.addEventListener('abort', () => { sawAbort = true })
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text: 'first' }
      // Simulate an adapter that honors abort: wait for abort or a timeout.
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 1000)
        options.signal?.addEventListener('abort', () => { clearTimeout(timer); resolve() }, { once: true })
      })
      if (options.signal?.aborted) return
      yield { type: 'text-delta', index: 0, text: 'second' }
      yield { type: 'finish', reason: { kind: 'stop' } }
    } finally {
      // the generator finally block runs after return()
    }
  })
  await withServer({}, async (port) => {
    const controller = new AbortController()
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'fake/m1', messages: [{ role: 'user', content: 'hi' }], stream: true }),
      signal: controller.signal,
    })
    const reader = response.body.getReader()
    await reader.read() // first delta received
    controller.abort()
    await new Promise((resolve) => setTimeout(resolve, 200))
    assert.equal(sawAbort, true, 'upstream should observe abort when client disconnects')
  })
})

await test('parseRoute: splitting and rejection', () => {
  assert.deepEqual(parseRoute('fake/m1'), { provider: 'fake', model: 'm1' })
  assert.deepEqual(parseRoute('fake/a/b'), { provider: 'fake', model: 'a/b' })
  assert.equal(parseRoute('noseparator'), null)
  assert.equal(parseRoute('/lead'), null)
  assert.equal(parseRoute('trail/'), null)
})

console.log(`\n${passed.length} passed, ${failed.length} failed`)
if (failed.length > 0) {
  process.exitCode = 1
}
// Never let a lingering server or timer hold the process open after reporting.
setTimeout(() => {
  console.log('FORCE EXIT (pending handles kept the process alive)')
  process.exit(process.exitCode ?? 0)
}, 1500).unref()
