// The client's error handling, with fetch replaced. No prediction is made up here: the responses below are errors
// and malformed bodies only. The client's schemas are checked against the real API by api.contract.ts.
import { afterEach, describe, expect, it, vi } from 'vitest'
import { API_BASE_URL, ApiError, errorMessage, getHealth, getRecords, predictRecord } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

function respond(status: number, body: string, statusText = '') {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(body, { status, statusText })))
}

async function failure(promise: Promise<unknown>): Promise<ApiError> {
  const error = await promise.then(() => null, (e: unknown) => e)
  if (!(error instanceof ApiError)) throw new Error(`expected an ApiError, got ${String(error)}`)
  return error
}

describe('errorMessage', () => {
  it("is FastAPI's detail, verbatim", () => {
    expect(errorMessage(413, 'Payload Too Large', '{"detail": "The request is 2,000,000 bytes"}'))
      .toBe('The request is 2,000,000 bytes')
  })

  it('is the status and the raw body when there is no string detail', () => {
    expect(errorMessage(500, 'Internal Server Error', 'Internal Server Error'))
      .toBe('HTTP 500 Internal Server Error: Internal Server Error')
    expect(errorMessage(422, '', '{"detail": [{"loc": ["body"]}]}')).toBe('HTTP 422: {"detail": [{"loc": ["body"]}]}')
    expect(errorMessage(502, 'Bad Gateway', '')).toBe('HTTP 502 Bad Gateway')
  })
})

describe('request', () => {
  it('calls the configured API address', async () => {
    const fetch = vi.fn(async () => new Response('[]'))
    vi.stubGlobal('fetch', fetch)
    await getRecords()
    expect(fetch).toHaveBeenCalledWith(`${API_BASE_URL}/api/records`, expect.objectContaining({ method: 'GET' }))
  })

  it('sends a test record as JSON', async () => {
    const fetch = vi.fn(async () => new Response('{"detail": "x"}', { status: 404 }))
    vi.stubGlobal('fetch', fetch)
    await failure(predictRecord('9'))
    expect(fetch).toHaveBeenCalledWith(`${API_BASE_URL}/api/predict`, expect.objectContaining({
      method: 'POST', body: '{"record_id":"9"}', headers: { 'Content-Type': 'application/json' },
    }))
  })

  it('reports no response at all as unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))
    const error = await failure(getRecords())
    expect(error).toMatchObject({ kind: 'unreachable', status: null, endpoint: 'GET /api/records' })
  })

  it("reports an error status with the API's own message", async () => {
    respond(400, '{"detail": "The record has 3 signals; the model needs the 12 standard leads"}')
    const error = await failure(predictRecord('9'))
    expect(error).toMatchObject({ kind: 'http', status: 400, endpoint: 'POST /api/predict',
                                  message: 'The record has 3 signals; the model needs the 12 standard leads' })
  })

  it('refuses a 2xx body of the wrong shape instead of passing it on', async () => {
    respond(200, '{"status": "ok"}')
    expect(await failure(getHealth())).toMatchObject({ kind: 'invalid-response', endpoint: 'GET /api/health' })
    respond(200, '[{"id": 9, "duration_s": 10, "sampling_rate": 100}]') // id must be a string
    expect(await failure(getRecords())).toMatchObject({ kind: 'invalid-response' })
    respond(200, 'not json')
    expect(await failure(getRecords())).toMatchObject({ kind: 'invalid-response' })
  })

  it('rethrows a cancelled request as it is, not as an API failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new DOMException('aborted', 'AbortError')
    }))
    const error = await getRecords().then(() => null, (e: unknown) => e)
    expect(error).toBeInstanceOf(DOMException)
  })
})
