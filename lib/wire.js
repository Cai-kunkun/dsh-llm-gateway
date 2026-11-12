/**
 * OpenAI wire to harness `GenerateOptions` message translation.
 *
 * Translation is honest: unknown content parts degrade to text, and assistant
 * `reasoning_content` becomes a `reasoning` block while `tool_calls` and
 * `tool_call_id` map onto harness tool-call and tool-result vocabulary.
 *
 * @module lib/wire.js
 */

const KNOWN_MEDIA = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif'])
const IMG_MEDIA_ALIASES = new Map([
  ['image/jpg', 'image/jpeg'],
  ['image/pjpeg', 'image/jpeg'],
  ['image/x-webp', 'image/webp'],
])

/** Resolve one data-URL or bare media type into an admitted media type, or null. */
export function resolveImageMediaType(value) {
  if (typeof value !== 'string') return null
  let mime = value
  if (mime.startsWith('data:')) {
    const semi = mime.indexOf(';')
    if (semi > 0) mime = mime.slice(5, semi)
    else mime = mime.slice(5)
  } else {
    // Bare or "type;parameter" spelling: strip the parameter segment.
    const semi = mime.indexOf(';')
    if (semi > 0) mime = mime.slice(0, semi)
  }
  mime = mime.trim().toLowerCase()
  mime = IMG_MEDIA_ALIASES.get(mime) ?? mime
  return KNOWN_MEDIA.has(mime) ? mime : null
}

/** Decode one `data:` URL into { mediaType, bytes }, or null when malformed. */
export function decodeDataUrl(value) {
  if (typeof value !== 'string' || !value.startsWith('data:')) return null
  const comma = value.indexOf(',')
  if (comma < 0) return null
  const header = value.slice(5, comma)
  const payload = value.slice(comma + 1)
  if (!/;base64$/i.test(header)) return null
  const mediaType = resolveImageMediaType(header)
  if (mediaType === null) return null
  let bytes
  try {
    bytes = Buffer.from(payload, 'base64')
  } catch {
    return null
  }
  if (bytes.length === 0) return null
  return { mediaType, bytes }
}

/**
 * OpenAI chat messages → harness RequestMessage[].
 *
 * Inbound image policy: a data URL is admitted through `attachments.saveImage()`
 * so vision-capable routes receive native input. Without an attachment store, or
 * under an explicit `imageMode: 'placeholder'`, it degrades to deterministic
 * placeholder text. Remote `http(s)` URLs are never fetched: no extra egress.
 *
 * @param {readonly unknown[]} bodyMessages - OpenAI `messages` array.
 * @param {object} [opts]
 * @param {(input: { data: Uint8Array, mediaType: string, name?: string }) => Promise<{ attachmentId: string, mediaType: string, bytes: number, width: number, height: number }>} [opts.saveImage]
 * @param {'auto'|'placeholder'} [opts.imageMode]
 * @returns {Promise<{ messages: object[], imageFailures: string[] }>} harness messages plus degradation records.
 */
export async function fromOpenAiMessages(bodyMessages, opts = {}) {
  const rows = Array.isArray(bodyMessages) ? bodyMessages : []
  const out = []
  const imageFailures = []
  const saveImage = typeof opts.saveImage === 'function' ? opts.saveImage : null
  const placeholderMode = opts.imageMode === 'placeholder' || saveImage === null

  for (const row of rows) {
    if (row === null || typeof row !== 'object') continue
    const role = typeof row.role === 'string' ? row.role : 'user'
    const content = []
    const textParts = []

    const pushText = (value) => {
      if (typeof value === 'string' && value !== '') {
        content.push({ type: 'text', text: value })
        textParts.push(value)
      }
    }

    // -- content normalization --
    if (typeof row.content === 'string') {
      pushText(row.content)
    } else if (Array.isArray(row.content)) {
      for (const part of row.content) {
        if (part === null || part === undefined) continue
        if (typeof part === 'string') { pushText(part); continue }
        if (typeof part !== 'object') continue
        const type = typeof part.type === 'string' ? part.type : ''
        if (type === 'text') {
          pushText(typeof part.text === 'string' ? part.text : '')
        } else if (type === 'image_url' || type === 'image') {
          const urlValue = typeof part.image_url === 'object' && part.image_url !== null
            ? part.image_url.url
            : (typeof part.image_url === 'string' ? part.image_url : part.url)
          const decoded = decodeDataUrl(urlValue)
          if (decoded !== null && !placeholderMode) {
            try {
              const ref = await saveImage({ data: decoded.bytes, mediaType: decoded.mediaType })
              content.push({ type: 'image', attachment: ref })
              continue
            } catch (error) {
              imageFailures.push(String(error?.message ?? error))
              // fall through to placeholder
            }
          } else if (decoded !== null) {
            imageFailures.push('image placeholder mode is enabled')
          } else {
            imageFailures.push('non-data-URL image content is not forwarded')
          }
          const label = typeof urlValue === 'string' && urlValue.length > 0
            ? `[image omitted: ${urlValue.length > 96 ? `${urlValue.slice(0, 96)}…` : urlValue}]`
            : '[image omitted]'
          content.push({ type: 'text', text: label })
          textParts.push(label)
        } else {
          // Unknown part type: extract text when possible, otherwise keep a JSON
          // summary placeholder so the whole message is not dropped as empty.
          const text = part.text ?? part.input_text ?? part.output_text ?? ''
          pushText(typeof text === 'string' && text !== '' ? text : `[unsupported part: ${JSON.stringify(part).slice(0, 200)}]`)
        }
      }
    } else if (row.content !== null && row.content !== undefined) {
      pushText(String(row.content))
    }

    // -- assistant tool_calls --
    if (role === 'assistant' && Array.isArray(row.tool_calls)) {
      for (const call of row.tool_calls) {
        if (call === null || typeof call !== 'object') continue
        const fn = typeof call.function === 'object' && call.function !== null ? call.function : {}
        content.push({
          type: 'tool-call',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${randomId()}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          arguments: typeof fn.arguments === 'string' && fn.arguments !== '' ? fn.arguments : '{}',
        })
      }
    }

    // -- tool result messages --
    if (role === 'tool') {
      const resultText = typeof row.content === 'string'
        ? row.content
        : Array.isArray(row.content)
          ? row.content.map((part) => typeof part === 'string' ? part : (typeof part?.text === 'string' ? part.text : JSON.stringify(part ?? ''))).join('\n')
          : JSON.stringify(row.content ?? '')
      out.push({
        role: 'tool',
        content: [{ type: 'text', text: resultText }],
        toolCallId: typeof row.tool_call_id === 'string' && row.tool_call_id !== '' ? row.tool_call_id : 'missing',
        isError: row.is_error === true,
        source: { kind: 'tool', callId: typeof row.tool_call_id === 'string' && row.tool_call_id !== '' ? row.tool_call_id : 'missing' },
      })
      continue
    }

    if (content.length === 0) continue

    if (role === 'assistant') {
      // Upstream reasoning_content (DeepSeek style) folds into a reasoning block,
      // preserving prefix-cache-friendly block order: reasoning precedes text.
      const merged = typeof row.reasoning_content === 'string' && row.reasoning_content !== ''
        ? [{ type: 'reasoning', text: row.reasoning_content }, ...content]
        : content
      out.push({
        role: 'assistant',
        content: merged,
        source: { kind: 'model', provider: 'gateway', model: String(row.name ?? '') },
      })
      continue
    }

    if (role === 'system' || role === 'developer') {
      // Both system and developer fold to system messages, TextBlock-only.
      out.push({ role: 'system', content })
      continue
    }

    out.push({ role: 'user', content })
  }

  return { messages: out, imageFailures }
}

/**
 * OpenAI `tools` to harness `ToolSchema[]`.
 * Accepts both the nested `{"type":"function","function":{...}}` and flat spellings.
 */
export function fromOpenAiTools(tools) {
  if (!Array.isArray(tools)) return undefined
  const out = []
  for (const tool of tools) {
    if (tool === null || typeof tool !== 'object') continue
    const fn = typeof tool.function === 'object' && tool.function !== null ? tool.function : tool
    const name = typeof fn.name === 'string' ? fn.name : ''
    if (name === '') continue
    const schema = typeof fn.parameters === 'object' && fn.parameters !== null ? fn.parameters : { type: 'object', properties: {} }
    out.push({
      name,
      description: typeof fn.description === 'string' ? fn.description : '',
      parameters: schema,
    })
  }
  return out.length > 0 ? out : undefined
}

/** Map an OpenAI wire failure surface onto { message, code, status }. */
export function failureOf(chunk) {
  const failure = chunk?.reason?.failure
  if (!failure) return { message: 'unknown error', code: 'INTERNAL', status: 500 }
  return {
    message: typeof failure.message === 'string' ? failure.message : 'unknown error',
    code: typeof failure.code === 'string' ? failure.code : 'INTERNAL',
    status: typeof failure.status === 'number' && Number.isFinite(failure.status)
      ? clampStatus(failure.status)
      : defaultStatusOf(failure.code),
  }
}

function clampStatus(status) {
  const code = Math.trunc(status)
  return code >= 400 && code <= 599 ? code : 500
}

function defaultStatusOf(code) {
  switch (code) {
    case 'NO_ADAPTER': return 404
    case 'UNKNOWN_MODEL': return 404
    case 'INVALID_CONFIG': return 400
    case 'UNSUPPORTED_OPTION': return 400
    case 'UNSUPPORTED_REASONING_EFFORT': return 400
    case 'MISSING_CREDENTIAL': return 401
    case 'INVALID_CREDENTIAL': return 401
    case 'AUTH': return 401
    case 'RATE_LIMIT': return 429
    case 'QUOTA': return 402
    case 'ACCOUNT_QUOTA': return 402
    case 'CONTEXT_WINDOW_EXCEEDED': return 413
    case 'IMAGE_OFFLOAD_REQUIRED': return 413
    case 'ABORTED': return 499
    default: return 502
  }
}

/** Short random suffix for synthesized ids (crypto-free, collision-safe enough). */
function randomId() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4)
}

/**
 * Stateful chunk folder: yields OpenAI delta events one chunk at a time.
 * Streaming and non-streaming paths share this single folding implementation.
 *
 * @returns {{ push(chunk: object): object[], readonly finishReason: string|null, readonly usage: object|null, readonly hasText: boolean, readonly hasToolCalls: boolean, readonly error: object|null }}
 */
export function createOpenAiFolder() {
  const toolCalls = new Map() // slot -> { id, name, arguments, sentName }
  let blockIndexToToolSlot = new Map() // harness block index -> tool slot
  let nextSlot = 0
  let sawText = false
  let sawTool = false
  let finishReason = null
  let usage = null
  let error = null
  let seq = 0
  const emitted = []

  return {
    push(chunk) {
      const events = []
      const idSuffix = `${++seq}`
      switch (chunk?.type) {
        case 'block-start': {
          if (chunk.blockType === 'tool-call') {
            const slot = nextSlot++
            blockIndexToToolSlot.set(chunk.index, slot)
            toolCalls.set(slot, { id: '', name: '', arguments: '', sentName: false })
          }
          break
        }
        case 'text-delta': {
          if (chunk.text === '') break
          sawText = true
          events.push({ delta: { content: chunk.text } })
          break
        }
        case 'reasoning-delta': {
          if (chunk.text === '') break
          events.push({ delta: { reasoning_content: chunk.text } })
          break
        }
        case 'tool-call-delta': {
          const slot = blockIndexToToolSlot.get(chunk.index) ?? nextSlot++
          let call = toolCalls.get(slot)
          if (call === undefined) {
            call = { id: '', name: '', arguments: '', sentName: false }
            toolCalls.set(slot, call)
            blockIndexToToolSlot.set(chunk.index, slot)
          }
          if (typeof chunk.id === 'string' && chunk.id !== '' && call.id === '') call.id = chunk.id
          const nameDelta = typeof chunk.name === 'string' && chunk.name !== '' ? chunk.name : ''
          if (nameDelta !== '' && !call.sentName) {
            call.name += nameDelta
            call.sentName = true
            call.arguments += chunk.argumentsDelta ?? ''
            events.push({ delta: { tool_calls: [{ index: slot, id: call.id || `call_${slot}_${idSuffix}`, type: 'function', function: { name: nameDelta, arguments: chunk.argumentsDelta ?? '' } }] } })
          } else {
            if (nameDelta !== '') call.name += nameDelta
            call.arguments += chunk.argumentsDelta ?? ''
            events.push({ delta: { tool_calls: [{ index: slot, function: { arguments: chunk.argumentsDelta ?? '' } }] } })
          }
          sawTool = true
          break
        }
        case 'block-end': {
          if (chunk.block?.type === 'tool-call') {
            const slot = blockIndexToToolSlot.get(chunk.index)
            const call = toolCalls.get(slot)
            if (call !== undefined) {
              if (typeof chunk.block.id === 'string' && chunk.block.id !== '') call.id = chunk.block.id
              if (!call.sentName && typeof chunk.block.name === 'string' && chunk.block.name !== '') {
                // No name was emitted incrementally: supply it once here.
                call.name = chunk.block.name
                call.sentName = true
                events.push({ delta: { tool_calls: [{ index: slot, id: call.id || `call_${slot}_${idSuffix}`, type: 'function', function: { name: chunk.block.name, arguments: chunk.block.arguments ?? '' } }] } })
              } else if (call.sentName && typeof chunk.block.arguments === 'string' && chunk.block.arguments !== call.arguments) {
                // Name already emitted: merge block-end arguments without duplication,
                const tail = chunk.block.arguments.startsWith(call.arguments)
                  ? chunk.block.arguments.slice(call.arguments.length)
                  : chunk.block.arguments
                call.arguments = chunk.block.arguments
                if (tail !== '') events.push({ delta: { tool_calls: [{ index: slot, function: { arguments: tail } }] } })
              }
            }
          }
          break
        }
        case 'usage': {
          usage = chunk.usage ?? null
          break
        }
        case 'finish': {
          if (chunk.reason?.kind === 'stop') finishReason = 'stop'
          else if (chunk.reason?.kind === 'tool-calls') finishReason = 'tool_calls'
          else if (chunk.reason?.kind === 'max-tokens') finishReason = 'length'
          else if (chunk.reason?.kind === 'aborted') finishReason = 'stop' // client or upstream cancellation
          else if (chunk.reason?.kind === 'error') { error = chunk.reason.failure ?? null; finishReason = null }
          break
        }
        default:
          break
      }
      for (const event of events) emitted.push(event)
      return events
    },
    /** Non-streaming path: replay every delta event accumulated so far. */
    events() {
      return emitted
    },
    get finishReason() { return finishReason },
    get usage() { return usage },
    get hasText() { return sawText },
    get hasToolCalls() { return sawTool },
    get error() { return error },
  }
}

/** Convenience wrapper: fold a whole chunk stream at once (tests, non-streaming). */
export function foldChunksToOpenAiStream(chunks, meta = {}) {
  const folder = createOpenAiFolder()
  const events = []
  for (const chunk of chunks) events.push(...folder.push(chunk))
  return {
    events,
    get finishReason() { return folder.finishReason },
    get usage() { return folder.usage },
    get hasText() { return folder.hasText },
    get hasToolCalls() { return folder.hasToolCalls },
    get error() { return folder.error },
  }
}

/** Harness TokenUsage to an OpenAI `usage` object. */
export function toOpenAiUsage(usage) {
  if (!usage || typeof usage !== 'object') return undefined
  const input = num(usage.inputTokens)
  const output = num(usage.outputTokens)
  if (input === null && output === null) return undefined
  const prompt = input ?? 0
  const completion = output ?? 0
  const out = {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  }
  const cacheRead = num(usage.cacheReadTokens)
  const cacheWrite = num(usage.cacheWriteTokens)
  const reasoning = num(usage.reasoningTokens)
  const details = {}
  if (cacheRead !== null) details.cached_tokens = cacheRead
  if (cacheWrite !== null) details.cache_write_tokens = cacheWrite
  if (reasoning !== null) details.reasoning_tokens = reasoning
  if (Object.keys(details).length > 0) out.prompt_tokens_details = details
  if (cacheRead !== null || cacheWrite !== null || reasoning !== null) {
    out.completion_tokens_details = {}
    if (reasoning !== null) out.completion_tokens_details.reasoning_tokens = reasoning
  }
  return out
}

function num(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

/** OpenAI error payload for direct serialization by the HTTP layer. */
export function errorPayload(status, code, message) {
  return {
    error: {
      message,
      type: status === 401 ? 'authentication_error'
        : status === 429 ? 'rate_limit_error'
        : status === 404 ? 'not_found_error'
        : status === 400 ? 'invalid_request_error'
        : 'api_error',
      code,
    },
  }
}
