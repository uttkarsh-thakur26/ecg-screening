/// <reference types="vitest/config" />
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { parse } from 'yaml'

// The API's address and its one allowed browser origin are set in config/default.yaml, like every other setting of
// the project. The dev server takes its origin from api.cors_origins, so the page always runs on the origin the API
// allows, and the client calls the API at api.host:api.port. Nothing here repeats those values.
const CONFIG_FILE = fileURLToPath(new URL('../config/default.yaml', import.meta.url))

interface ApiSettings {
  origin: URL         // where the dev server (and vite preview) serves the page
  apiBaseUrl: string  // where the page calls the API
}

function apiSettings(): ApiSettings {
  const api = parse(readFileSync(CONFIG_FILE, 'utf8'))?.api
  if (api == null) throw new Error(`${CONFIG_FILE} has no api section`)

  const origins: unknown = api.cors_origins
  if (!Array.isArray(origins) || origins.length !== 1 || typeof origins[0] !== 'string') {
    throw new Error(`api.cors_origins in ${CONFIG_FILE} must list exactly one origin, the Vite dev server's; it is ` +
                    JSON.stringify(origins))
  }
  const origin = new URL(origins[0])
  if (origin.protocol !== 'http:' || origin.port === '' || origin.origin !== origins[0]) {
    throw new Error(`api.cors_origins in ${CONFIG_FILE} must be an origin like http://localhost:<port>, with an ` +
                    `explicit port and no path; it is ${JSON.stringify(origins[0])}`)
  }

  const host: unknown = api.host
  const port = Number(api.port)
  if (typeof host !== 'string' || host === '') {
    throw new Error(`api.host in ${CONFIG_FILE} must be the address the API listens on; it is ${JSON.stringify(host)}`)
  }
  if (host === '0.0.0.0' || host === '::') {
    throw new Error(`api.host in ${CONFIG_FILE} is ${JSON.stringify(host)}, an address to listen on, not one the ` +
                    'browser can call; set the address the API is reached at, such as 127.0.0.1')
  }
  if (!Number.isInteger(port) || port <= 0) {
    throw new Error(`api.port in ${CONFIG_FILE} must be a port number; it is ${JSON.stringify(api.port)}`)
  }
  return { origin, apiBaseUrl: `http://${host.includes(':') ? `[${host}]` : host}:${port}` }
}

const { origin, apiBaseUrl } = apiSettings()
// strictPort: if the port is taken, Vite stops with an error instead of moving to the next free port, where the API
// would refuse every request (CORS).
const server = { host: origin.hostname, port: Number(origin.port), strictPort: true }

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: { __API_BASE_URL__: JSON.stringify(apiBaseUrl) },
  server,
  preview: server,
  test: {
    projects: [
      // Pure functions only; no network. css.include: the contrast test reads the design tokens in src/index.css
      // (imported ?raw), which Vitest would otherwise replace with an empty string.
      { extends: true, test: { name: 'unit', include: ['src/**/*.test.ts'], css: { include: [/index\.css/] } } },
      // The client against the running API (python -m src.api.main): every response must pass the client's schemas.
      { extends: true, test: { name: 'contract', include: ['src/**/*.contract.ts'], testTimeout: 30_000 } },
    ],
  },
})
