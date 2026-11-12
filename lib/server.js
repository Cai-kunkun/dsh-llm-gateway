/**
 * OpenAI-compatible HTTP service.
 *
 * Two endpoints:
 *   GET  /v1/models             - every registered model across all providers
 *   POST /v1/chat/completions   - streaming (SSE) and non-streaming completion
 *
 * Authentication: when `apiKey` is configured the service requires
 * `Authorization: Bearer <key>`. Without a key only loopback clients are
 * admitted, and binding a non-loopback host is refused. This listener faces the
 * network, so access is closed by default.
 *
 * @module lib/server.js
 */

import http from 'node:http'
import { fromOpenAiMessages, fromOpenAiTools, failureOf, createOpenAiFolder, toOpenAiUsage, errorPayload } from './wire.js'

const MAX_BODY_BYTES = 32 * 1024 * 1024
const LOOPBACK_HOSTS = new Set(['127.0.0.1', '::1', 'localhost'])

/**
 * @param {object} deps
 * @param {() => { stream: (options: object) => AsyncIterable<object>, listProviders: () => { id: string, name: string }[], listModels: (provider: string) => Promise<object[]>, get: (name: string) => unknown | undefined }} deps.getLlm
 *   Returns the current llm service (resolved lazily so it survives HMR).
 * @param {(name: string) => unknown | undefined} deps.getService - resolves an optional service such as attachments.
 * @param {{ port: number, host: string, apiKey: string | null, imageMode: 'auto' | 'placeholder', log: (level: string, msg: string, ...rest: unknown[]) => void, onState: (state: object) => void }} deps.config
 * @returns {{ dispose: () => Promise<void>, address: () => { port: number, host: string } | null, state: () => object }}
 */
export function startGatewayServer(deps) {
  const log = deps.config.log ?? (() => {})
  let llmCache = null
  const llm = () => {
    if (llmCache === null) llmCache = deps.getLlm()
    return llmCache
  }
  const attachments = () => deps.getService('attachments')

  const server = http.createServer((req, res) => {
    handle(req, res).catch((error) => {
      log('error', 'gateway: unhandled request failure:', error?.message ?? error)
      if (!res.headersSent) {
        res.writeHead(500, { 'content-type': 'application/json' })
        res.end(JSON.stringify(errorPayload(500, 'INTERNAL', 'internal gateway failure')))
      } else {
        try { res.end() } catch {}
      }
    })
  })

  // Access-level request logging; bodies are never recorded.
  server.on('request', (req) => {
    if (req.url !== '/healthz') log('info', `gateway: ${req.method} ${req.url}`)
  })

  /** @type {{ port: number, host: string } | null} */
  let boundAddress = null
  /** @type {Error | null} */
  let bindError = null
  let disposed = false

  const ready = new Promise((resolve, reject) => {
    server.once('error', (error) => { bindError = error; reject(error) })
    server.listen(deps.config.port, deps.config.host, () => {
      const address = server.address()
      boundAddress = typeof address === 'object' && address !== null
        ? { port: address.port, host: deps.config.host }
        : { port: deps.config.port, host: deps.config.host }
      resolve()
    })
  })

  const noKeyLoopbackOnly = deps.config.apiKey === null || deps.config.apiKey === ''

  async function handle(req, res) {
    const url = new URL(req.url ?? '/', 'http://localhost')
    const path = url.pathname.replace(/\/+$/, '') || '/'

    if (path === '/healthz') {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ ok: true, service: 'dsh-llm-gateway' }))
      return
    }

    // -- authentication --
    if (!noKeyLoopbackOnly) {
      const auth = String(req.headers.authorization ?? '')
      const supplied = auth.startsWith('Bearer ') ? auth.slice(7).trim() : ''
      const expected = String(deps.config.apiKey)
      if (!timingSafeEqual(supplied, expected)) {
        res.writeHead(401, { 'content-type': 'application/json' })
        res.end(JSON.stringify(errorPayload(401, 'INVALID_API_KEY', 'invalid or missing API key (send "Authorization: Bearer <key>")')))
        return
      }
    } else if (!LOOPBACK_HOSTS.has(String(req.socket.remoteAddress ?? ''))) {
      res.writeHead(403, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(403, 'FORBIDDEN', 'gateway is loopback-only while no apiKey is configured')))
      return
    }

    if (req.method === 'GET' && (path === '/v1/models' || path === '/models')) {
      await handleModels(res)
      return
    }

    if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/chat/completions')) {
      await handleCompletions(req, res)
      return
    }

    res.writeHead(404, { 'content-type': 'application/json' })
    res.end(JSON.stringify(errorPayload(404, 'NOT_FOUND', `no route for ${req.method} ${path}`)))
  }

  async function handleModels(res) {
    try {
      const service = llm()
      if (service === undefined || service === null) {
        res.writeHead(503, { 'content-type': 'application/json' })
        res.end(JSON.stringify(errorPayload(503, 'LLM_UNAVAILABLE', 'llm service is not mounted in this profile')))
        return
      }
      const providers = service.listProviders()
      const settled = await Promise.allSettled(providers.map((provider) => service.listModels(provider.id)))
      const data = []
      const created = Math.floor(Date.now() / 1000)
      for (let i = 0; i < providers.length; i += 1) {
        const provider = providers[i]
        const outcome = settled[i]
        if (outcome.status !== 'fulfilled') {
          log('warn', `gateway: listModels failed for provider "${provider.id}":`, outcome.reason?.message ?? outcome.reason)
          continue
        }
        for (const model of outcome.value) {
          data.push({
            id: `${provider.id}/${model.id}`,
            object: 'model',
            created,
            owned_by: provider.name || provider.id,
            x_gateway: { provider: provider.id, model: model.id },
          })
        }
      }
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ object: 'list', data }))
    } catch (error) {
      log('error', 'gateway: /v1/models failure:', error?.message ?? error)
      res.writeHead(500, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(500, 'INTERNAL', 'failed to list models')))
    }
  }

  async function handleCompletions(req, res) {
    const service = llm()
    if (service === undefined || service === null) {
      res.writeHead(503, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(503, 'LLM_UNAVAILABLE', 'llm service is not mounted in this profile')))
      return
    }

    let body
    try {
      body = await readJsonBody(req)
    } catch (error) {
      res.writeHead(error?.code === 'TOO_LARGE' ? 413 : 400, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(error?.code === 'TOO_LARGE' ? 413 : 400, 'BAD_REQUEST', error?.message ?? 'invalid request body')))
      return
    }
    if (body === null || typeof body !== 'object') {
      res.writeHead(400, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(400, 'BAD_REQUEST', 'request body must be a JSON object')))
      return
    }

    const route = parseRoute(body.model)
    if (route === null) {
      res.writeHead(400, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(400, 'BAD_MODEL', '"model" is required; use "<provider>/<model>" (see GET /v1/models)')))
      return
    }

    const store = attachments()
    const saveImage = store !== undefined && store !== null && typeof store.saveImage === 'function'
      ? (input) => store.saveImage(input)
      : null
    const { messages, imageFailures } = await fromOpenAiMessages(body.messages, {
      saveImage,
      imageMode: deps.config.imageMode,
    })
    if (messages.length === 0) {
      res.writeHead(400, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(400, 'EMPTY_MESSAGES', '"messages" produced no convertible content')))
      return
    }
    if (imageFailures.length > 0) {
      log('warn', `gateway: ${imageFailures.length} image part(s) degraded to placeholder text:`, imageFailures[0])
    }

    const options = {
      provider: route.provider,
      model: route.model,
      messages,
      ...(Array.isArray(body.stop) && body.stop.length > 0 ? { stop: body.stop.map(String) } : {}),
    }
    if (typeof body.temperature === 'number' && Number.isFinite(body.temperature)) options.temperature = body.temperature
    if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) && body.max_tokens > 0) options.maxTokens = Math.trunc(body.max_tokens)
    // OpenAI reasoning_effort maps onto harness reasoningEffort; both official adapters accept it literally.
    if (typeof body.reasoning_effort === 'string' && body.reasoning_effort !== '') options.reasoningEffort = body.reasoning_effort
    const tools = fromOpenAiTools(body.tools)
    if (tools !== undefined) options.tools = tools

    const completionId = `chatcmpl-${randomToken(24)}`
    const created = Math.floor(Date.now() / 1000)
    const wantStream = body.stream === true
    const modelLabel = body.model ?? `${route.provider}/${route.model}`

    // -- start the upstream stream --
    let upstream
    try {
      upstream = service.stream(options)[Symbol.asyncIterator]()
    } catch (error) {
      // A synchronous stream() throw is a waterfall or listener failure, not a model failure.
      log('error', 'gateway: stream threw synchronously:', error?.message ?? error)
      res.writeHead(500, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(500, 'STREAM_FAILURE', error?.message ?? 'stream failed before it started')))
      return
    }

    // A client disconnect aborts the upstream request through the adapter signal.
    const controller = new AbortController()
    const onClientClose = () => controller.abort(new Error('client disconnected'))
    res.on('close', onClientClose)
    options.signal = controller.signal

    const folder = createOpenAiFolder()

    try {
      if (!wantStream) {
        // -- non-streaming: drain the whole stream, then fold --
        while (true) {
          const row = await upstream.next()
          if (row.done === true) break
          folder.push(row.value)
        }
      } else {
        // -- streaming: commit SSE headers, then forward each chunk as it lands --
        res.writeHead(200, {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
          'x-request-id': completionId,
        })
        const baseChunk = { id: completionId, object: 'chat.completion.chunk', created, model: modelLabel }
        const send = (payload) => {
          if (!res.writableEnded && !res.destroyed) res.write(`data: ${JSON.stringify(payload)}\n\n`)
        }
        send({ ...baseChunk, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] })

        while (true) {
          const row = await upstream.next()
          if (row.done === true) break
          for (const event of folder.push(row.value)) {
            send({ ...baseChunk, choices: [{ index: 0, delta: event.delta, finish_reason: null }] })
          }
        }

        const usage = toOpenAiUsage(folder.usage)
        send({
          ...baseChunk,
          choices: [{ index: 0, delta: {}, finish_reason: folder.finishReason ?? 'stop' }],
          ...(usage !== undefined && body.stream_options?.include_usage === true ? { usage } : {}),
        })
        if (usage !== undefined && body.stream_options?.include_usage === true) {
          send({ ...baseChunk, choices: [], usage })
        }
        res.write('data: [DONE]\n\n')
        res.end()
        return
      }
    } catch (error) {
      // Upstream iteration threw: a middleware or consumer failure. Adapter-internal failures arrive as an error finish chunk instead.
      log('error', 'gateway: upstream stream iteration failed:', error?.message ?? error)
      if (!res.headersSent) {
        res.writeHead(500, { 'content-type': 'application/json' })
        res.end(JSON.stringify(errorPayload(500, 'STREAM_FAILURE', error?.message ?? 'upstream stream failed')))
      } else {
        try { res.end() } catch {}
      }
      return
    } finally {
      res.off('close', onClientClose)
    }

    // -- non-streaming epilogue --
    const failure = folder.error
    if (failure !== null) {
      const mapped = failureOf({ reason: { kind: 'error', failure } })
      res.writeHead(mapped.status, { 'content-type': 'application/json' })
      res.end(JSON.stringify(errorPayload(mapped.status, mapped.code, mapped.message)))
      return
    }

    writeJsonCompletion(res, folder, { id: completionId, model: modelLabel, created })
  }

  function writeJsonCompletion(res, folder, meta) {
    let text = ''
    let reasoning = ''
    const toolCalls = []
    for (const event of folder.events()) {
      if (typeof event.delta?.content === 'string') text += event.delta.content
      if (typeof event.delta?.reasoning_content === 'string') reasoning += event.delta.reasoning_content
      for (const call of event.delta?.tool_calls ?? []) {
        const slot = typeof call.index === 'number' ? call.index : toolCalls.length
        while (toolCalls.length <= slot) toolCalls.push({ id: '', type: 'function', function: { name: '', arguments: '' } })
        if (typeof call.id === 'string' && call.id !== '') toolCalls[slot].id = call.id
        if (typeof call.function?.name === 'string' && call.function.name !== '') toolCalls[slot].function.name += call.function.name
        if (typeof call.function?.arguments === 'string') toolCalls[slot].function.arguments += call.function.arguments
      }
    }
    const message = {
      role: 'assistant',
      content: text === '' ? null : text,
      ...(toolCalls.length > 0 ? { tool_calls: toolCalls.map((call, index) => ({ ...call, index })) } : {}),
      ...(reasoning !== '' ? { reasoning_content: reasoning } : {}),
    }
    const usage = toOpenAiUsage(folder.usage)
    res.writeHead(200, { 'content-type': 'application/json', 'x-request-id': meta.id })
    res.end(JSON.stringify({
      id: meta.id,
      object: 'chat.completion',
      created: meta.created,
      model: meta.model,
      choices: [{
        index: 0,
        message,
        finish_reason: folder.finishReason ?? 'stop',
      }],
      ...(usage !== undefined ? { usage } : {}),
    }))
  }

  async function dispose() {
    if (disposed) return
    disposed = true
    await new Promise((resolve) => server.close(() => resolve()))
  }

  function state() {
    return {
      listening: boundAddress !== null,
      bindError: bindError === null ? null : String(bindError.message ?? bindError),
      ...boundAddress,
      authMode: noKeyLoopbackOnly ? 'loopback-only' : 'api-key',
    }
  }

  deps.config.onState?.(state())
  return { dispose, address: () => boundAddress, state, ready }
}

/** Parse "<provider>/<model>", splitting at the first slash so ids may contain more. */
export function parseRoute(model) {
  if (typeof model !== 'string') return null
  const slash = model.indexOf('/')
  if (slash <= 0 || slash === model.length - 1) return null
  return { provider: model.slice(0, slash), model: model.slice(slash + 1) }
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > MAX_BODY_BYTES) {
        const error = new Error('request body exceeds the 32 MiB limit')
        error.code = 'TOO_LARGE'
        req.destroy()
        reject(error)
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8')
      if (raw.trim() === '') { resolve(null); return }
      try {
        resolve(JSON.parse(raw))
      } catch {
        reject(new Error('request body is not valid JSON'))
      }
    })
    req.on('error', reject)
  })
}

/** Drain a chunk stream into an array, aborting when the client disconnects. */
function timingSafeEqual(supplied, expected) {
  const a = Buffer.from(String(supplied))
  const b = Buffer.from(expected)
  if (a.length !== b.length) {
    // Compare even on length mismatch so timing does not leak the difference.
    Buffer.compare(a, b)
    return false
  }
  return Buffer.compare(a, b) === 0
}

function randomToken(length) {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let out = ''
  for (let i = 0; i < length; i += 1) out += alphabet[Math.floor(Math.random() * alphabet.length)]
  return out
}
