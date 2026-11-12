/**
 * dsh-llm-gateway: exposes every model configured in DeepSeek Harness as a
 * local OpenAI-compatible HTTP service (default port 2778).
 *
 * Other agent tools (Cline, Open WebUI, LangChain, aider, Codex CLI, and
 * similar) point their base URL at `http://127.0.0.1:2778/v1`. The gateway
 * forwards each request to `ctx.llm.stream()`, so the harness adapter that
 * already owns the target provider route performs the real call. Credentials,
 * quota handling, retry policy, and session semantics stay with the harness;
 * the gateway itself stores no credentials.
 *
 * Model identifiers use the `<provider>/<model>` form, for example
 * `deepseek-official/deepseek-chat` or `openai/gpt-4o-mini`. `GET /v1/models`
 * returns the complete aggregated inventory.
 *
 * Security model: `apiKey: null` plus a loopback-only bind is the default. With
 * no key configured the gateway refuses every non-loopback connection;
 * configuring a key enables Bearer authentication and permits binding to a
 * non-loopback address.
 *
 * @module index.js
 */

const name = 'llm-gateway'
const inject = ['llm']

/**
 * Configuration defaults: cordis passes an un-schema'd config object through
 * unchanged, so every field is normalized defensively and falls back to its default.
 */
function normalizeConfig(raw) {
  const config = raw !== null && typeof raw === 'object' && !Array.isArray(raw) ? raw : {}
  const portCandidate = Number(config.port)
  const port = Number.isFinite(portCandidate) && portCandidate > 0 && portCandidate < 65536 && Math.trunc(portCandidate) === portCandidate
    ? Math.trunc(portCandidate)
    : 2778
  const host = typeof config.host === 'string' && config.host !== '' ? config.host : '127.0.0.1'
  const apiKey = typeof config.apiKey === 'string' && config.apiKey.length > 0 ? config.apiKey : null
  const imageMode = config.imageMode === 'placeholder' ? 'placeholder' : 'auto'
  return { port, host, apiKey, imageMode }
}

export function apply(ctx, rawConfig) {
  const config = normalizeConfig(rawConfig)
  const logger = typeof ctx.logger?.info === 'function' ? ctx.logger : { info: () => {}, warn: () => {}, error: () => {} }

  const log = (level, message, ...rest) => {
    const method = level === 'error' ? 'error' : level === 'warn' ? 'warn' : 'info'
    try {
      logger[method]?.(`[llm-gateway] ${message}`, ...rest)
    } catch {}
  }

  // -- llm service resolution: injection guarantees presence, resolution stays lazy --
  const getLlm = () => ctx.get('llm')
  const getService = (serviceName) => ctx.get(serviceName)

  let serverHandle = null
  let startFailure = null

  // Start on the next tick so the synchronous apply phase never blocks the
  // event loop and adapter registration (also an apply-phase step) settles first.
  const startTimer = setTimeout(() => {
    import('./lib/server.js').then(({ startGatewayServer }) => {
      serverHandle = startGatewayServer({
        getLlm,
        getService,
        config: {
          ...config,
          log: (level, message, ...rest) => log(level, message, ...rest),
          onState: () => {},
        },
      })
      serverHandle.ready.then(() => {
        const address = serverHandle.address()
        log('info', `gateway listening on http://${address.host}:${address.port}/v1 (${config.apiKey === null ? 'loopback-only, no apiKey' : 'bearer apiKey required'})`)
        log('info', `models listing: http://${address.host}:${address.port}/v1/models`)
      }).catch((error) => {
        startFailure = error
        log('error', `gateway failed to bind ${config.host}:${config.port}: ${error?.message ?? error}`)
      })
    }).catch((error) => {
      startFailure = error
      log('error', `gateway failed to start: ${error?.message ?? error}`)
    })
  }, 0)

  ctx.effect(() => async () => {
    clearTimeout(startTimer)
    if (serverHandle !== null) {
      try {
        await serverHandle.dispose()
        log('info', 'gateway stopped')
      } catch (error) {
        log('warn', `gateway dispose reported: ${error?.message ?? error}`)
      }
      serverHandle = null
    }
  }, 'llm-gateway: http server')

  // -- diagnostic state so host logs can answer what the gateway is doing --
  const status = () => ({
    service: name,
    config: { port: config.port, host: config.host, authMode: config.apiKey === null ? 'loopback-only' : 'api-key', imageMode: config.imageMode },
    server: serverHandle === null ? (startFailure !== null ? { error: String(startFailure.message ?? startFailure) } : null) : serverHandle.state(),
  })

  if (typeof ctx.logger?.info === 'function') {
    log('info', `llm-gateway initializing (will listen on ${config.host}:${config.port})`)
  }

  // Exposed for tests and debugging: not a contract, internal to this plugin.
  apply.__gatewayStatus = status
}

export { name, inject }
export default { name, inject, apply }
