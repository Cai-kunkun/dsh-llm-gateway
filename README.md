# dsh-llm-gateway

An OpenAI-compatible HTTP gateway that exposes every model configured in [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) on a local port (default `2778`).

The plugin mounts the harness `llm` service and translates inbound OpenAI Chat Completions requests into `ctx.llm.stream()` calls. Requests are dispatched through the adapter that already serves the target provider route, so credentials, quota handling, retry policy, and session semantics remain owned by the harness. The gateway stores no credentials and keeps no model catalog of its own.

## Architecture

```
OpenAI client
     |
     v
dsh-llm-gateway  (HTTP :2778)
     |  translate OpenAI wire -> GenerateOptions
     v
ctx.llm.stream({ provider, model, messages, tools, ... })
     |
     v
Harness adapter registry  (llm-pi-ai, llm-deepseek, third-party adapters)
     |
     v
Provider API
```

Model discovery is not duplicated. `GET /v1/models` calls `ctx.llm.listProviders()` followed by `ctx.llm.listModels(provider)` for each registered route, so the gateway enumerates exactly what the mounted adapters advertise. Adding or removing a model in the harness configuration changes the gateway inventory with no gateway-side change. A provider whose adapter advertises an empty catalog (for example, a missing API key) is omitted rather than listed as a dead entry.

## Requirements

- DeepSeek Harness `0.2.x`
- Node.js `>= 20`

## Installation

From a profile, as a local package:

```sh
dsh plugin --profile web add link:/path/to/dsh-llm-gateway
```

Or from npm:

```sh
dsh plugin --profile web add dsh-llm-gateway
```

Restart the profile (`dsh web`) to activate. The plugin registers through its bundle patch; no manual profile edit is required.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/healthz` | Liveness probe, unauthenticated |
| GET | `/v1/models` | Aggregated catalog across all registered providers |
| POST | `/v1/chat/completions` | Streaming and non-streaming completion |

Model identifiers use the `<provider>/<model>` form, for example `deepseek-official/deepseek-chat` or `openai/gpt-4o-mini`. The separator is the first `/`; any remaining slashes belong to the model id.

## Usage

```sh
curl http://127.0.0.1:2778/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

Python, via the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:2778/v1", api_key="unused")
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

Node:

```js
import OpenAI from 'openai'

const client = new OpenAI({ baseURL: 'http://127.0.0.1:2778/v1', apiKey: 'unused' })
const response = await client.chat.completions.create({
  model: 'openai/gpt-4o-mini',
  messages: [{ role: 'user', content: 'hello' }],
})
console.log(response.choices[0].message.content)
```

Any client that accepts a configurable OpenAI base URL works: Cline, Continue, Open WebUI, aider, Codex CLI, LangChain, and similar tools.

## Configuration

Configuration lives in the profile patch layer:

```yaml
- id: llm-gateway
  name: dsh-llm-gateway
  config:
    port: 2778
    host: 127.0.0.1
    apiKey: null
    allowLan: false
    imageMode: auto
```

| Key | Default | Description |
|---|---|---|
| `port` | `2778` | Listen port |
| `host` | `127.0.0.1` | Listen address |
| `apiKey` | `null` | When non-empty, requires `Authorization: Bearer <key>` |
| `allowLan` | `false` | Admits non-loopback clients without a credential when no `apiKey` is set. Opt-in only |
| `imageMode` | `auto` | `auto` admits data-URL images through the attachment store; `placeholder` always degrades them to text |

## Request translation

The gateway maps the OpenAI wire format onto harness types without inventing fields.

- `messages` — `system` and `developer` roles both fold to harness `system` messages. Assistant `tool_calls` become `tool-call` blocks; `tool` messages carry their `tool_call_id` as the harness `toolCallId`. Assistant `reasoning_content` is restored as a leading `reasoning` block.
- `tools` — accepted in both nested (`{"type":"function","function":{...}}`) and flat forms.
- Sampling — `temperature`, `max_tokens`, and `stop` pass through directly. `reasoning_effort` maps to the harness `reasoningEffort` identifier.
- Images — a `data:` URL is admitted through `attachments.saveImage()` and forwarded as a durable image reference, so vision-capable routes receive native image input. Without an attachment service, or under `imageMode: placeholder`, the image degrades to deterministic placeholder text, matching the projection the runtime already applies to text-only routes. Remote `http(s)` image URLs are not fetched; the gateway performs no additional egress.
- Unknown content parts degrade to a text placeholder rather than dropping the message.

## Response translation

Harness `StreamChunk` output is folded into OpenAI deltas by a stateful folder shared between the streaming and non-streaming paths.

- Text and reasoning deltas map to `delta.content` and `delta.reasoning_content`.
- Tool calls are renumbered from zero. Upstream block indices are shared with text and reasoning blocks, so a client accumulating by index would otherwise produce a sparse array. Arguments are emitted incrementally, and the assembled block-end arguments are merged without duplication.
- `finish_reason` maps `stop`, `tool-calls`, and `max-tokens` to `stop`, `tool_calls`, and `length`. An aborted stream maps to `stop`; failures use the HTTP error channel instead of an SSE frame.
- Usage is always present on non-streaming responses. On streaming responses it is emitted when `stream_options.include_usage` is set, following the OpenAI convention of a finish frame plus a terminal usage frame. Cache counters surface as `prompt_tokens_details.cached_tokens`.
- Streams terminate with `data: [DONE]`.

## Error handling

Harness failure codes map to HTTP status codes, and the body uses the OpenAI `{"error": {...}}` shape.

| Harness code | HTTP |
|---|---|
| `NO_ADAPTER` | 404 |
| `BAD_MODEL` (gateway) | 400 |
| `MISSING_CREDENTIAL`, `INVALID_CREDENTIAL`, `AUTH` | 401 |
| `UNSUPPORTED_OPTION`, `UNSUPPORTED_REASONING_EFFORT` | 400 |
| `QUOTA`, `ACCOUNT_QUOTA` | 402 |
| `RATE_LIMIT` | 429 |
| `CONTEXT_WINDOW_EXCEEDED`, `IMAGE_OFFLOAD_REQUIRED` | 413 |
| `TRANSPORT`, `SERVER`, unmapped | 502 |

Streaming responses are committed to SSE headers only after the upstream iterator is constructed successfully. Failures that arrive as a terminal `error` finish chunk before any frame is written are reported as an HTTP error status; middleware-level throws surface as `STREAM_FAILURE`.

## Security model

The gateway is a network-facing listener with no session behind it, so access is closed by default. Admission is decided by three explicit modes:

| Mode | Condition | Behavior |
|---|---|---|
| `api-key` | `apiKey` set | Every request needs a matching `Authorization: Bearer`, compared in constant time over equal-length buffers |
| `loopback` | no `apiKey`, `allowLan: false` | Non-loopback peers are refused at the request boundary, independent of the bind address |
| `open-network` | no `apiKey`, `allowLan: true` | Any peer that can reach the port is admitted without a credential |

`allowLan` is a deliberate separate switch rather than a side effect of binding a wildcard address. Pointing `host` at `0.0.0.0` while leaving `allowLan` off still refuses remote clients, and a configured `apiKey` is never bypassed by `allowLan`.

`open-network` is intended for a trusted private network only. It grants every host on that network access to the configured models and their provider quota, so enable it knowingly.

Other protections apply in all modes:

- Loopback comparison normalizes IPv4-mapped IPv6 addresses (`::ffff:127.0.0.1`), so a dual-stack listener cannot misclassify a local peer.
- Client disconnects abort the upstream request through an `AbortSignal`, so the provider-side call is cancelled rather than left running.
- Request bodies are capped at 32 MiB and are never persisted. Access logging records method and path only.

## Testing

```sh
node test/selftest.mjs
```

The suite runs without a harness host, using a stub `llm` service. It covers message and tool conversion, chunk folding for text, reasoning, and both tool-call paths, SSE incrementality, authentication, status mapping, and cancellation propagation.

## Limitations

- `POST /v1/responses` is not implemented; only Chat Completions is served.
- `tool_choice`, `top_p`, and penalty parameters are not forwarded, because the harness call vocabulary does not carry them.
- PNG, JPEG, WebP, and GIF data-URL images are supported; other media types degrade to placeholder text.
- Model discovery reflects the advertising adapters. A provider that fails to list models is logged and skipped rather than failing the whole catalog request.

## License

MIT
