// Client for the FastAPI backend (src/api/main.py). The schemas mirror src/api/schemas.py field for field and are
// checked on every response: a response of another shape is an error the page shows, never data it displays.
// The page computes no numbers of its own; everything it shows comes from these responses.

import { z } from 'zod'

/** The API's address, from api.host and api.port in config/default.yaml (vite.config.ts). */
export const API_BASE_URL: string = __API_BASE_URL__

// ------------------------------------------------------------------------------------------------ schemas

const Probability = z.number().min(0).max(1)

const Health = z.strictObject({
  status: z.literal('ok'),
  model_loaded: z.boolean(),
  model_version: z.string(),
  model_tag: z.string(),
  device: z.string(),
})

/** A held-out test record: its id and shape only. The API sends no labels and no patient data. */
const RecordSummary = z.strictObject({
  id: z.string(),
  duration_s: z.number().int(),
  sampling_rate: z.number().int(),
})

const Signal = z
  .strictObject({
    leads: z.array(z.string()).min(1),
    sampling_rate: z.number().int(),
    units: z.literal('mV'),
    data: z.array(z.array(z.number())), // (leads, samples), before normalisation
  })
  .refine((s) => s.data.length === s.leads.length, { message: 'data must have one row per lead' })

const ClassFields = {
  class: z.string(),
  label: z.string(),       // plain-English name of the class
  logit: z.number(),       // the model's raw output
  probability: Probability, // calibrated
}

/** An abnormal class: its calibrated probability, the operating point's threshold, and probability >= threshold. */
const AbnormalClass = z.strictObject({ ...ClassFields, threshold: Probability, flagged: z.boolean() })

/** The normal class (NORM): a probability only. It has no threshold and is never flagged, so both are null. */
const NormalClass = z.strictObject({ ...ClassFields, threshold: z.null(), flagged: z.null() })

const ClassPrediction = z.union([AbnormalClass, NormalClass])

const InputFields = {
  sampling_rate_hz: z.number(), // as recorded, before any resampling
  n_samples: z.number().int(),  // as recorded
  resampled: z.boolean(),
}

const RecordInput = z.strictObject({
  ...InputFields,
  source: z.literal('test_record'),
  record_id: z.string(),
  checksums_verified: z.null(),
})

const UploadInput = z.strictObject({
  ...InputFields,
  source: z.literal('upload'),
  record_id: z.null(),
  checksums_verified: z.boolean(), // the .dat matched the header's checksums; false only if the header has none
})

const PredictionFields = {
  predictions: z.array(ClassPrediction).min(1), // every class, in the model's class order
  operating_point: z.string(),
  operating_point_rule: z.string(),
  flagged_count: z.number().int(),
  abnormal_count: z.number().int(),
  summary: z.string(),                          // written by the API: "N of 4 abnormal categories flagged"
  attention: z.array(Probability),              // Grad-CAM of attention_class, one value per sample, in [0, 1]
  attention_class: z.string(),                  // the abnormal class with the highest calibrated probability
  attention_raw_peak: z.number(),               // 0: no time position raised the class score
  model_version: z.string(),
  inference_ms: z.number(),
  attention_ms: z.number(),
}

type PredictionShape = { predictions: z.infer<typeof ClassPrediction>[]; flagged_count: number;
                         abnormal_count: number; attention_class: string }

/** The counts the API reports must agree with its own rows. */
function consistent(p: PredictionShape): boolean {
  const abnormal = p.predictions.filter((c) => c.threshold !== null)
  return p.abnormal_count === abnormal.length &&
         p.flagged_count === abnormal.filter((c) => c.flagged === true).length &&
         abnormal.some((c) => c.class === p.attention_class)
}
const CONSISTENT = { message: 'flagged_count, abnormal_count or attention_class disagree with the per-class rows' }

/** A held-out test record's prediction. Its signal is GET /api/records/{id}/signal. */
const RecordPrediction = z
  .strictObject({ ...PredictionFields, input: RecordInput, signal: z.null() })
  .refine(consistent, CONSISTENT)

/** An upload's prediction, with the signal as the model read it (resampled to the model's rate if needed). */
const UploadPrediction = z
  .strictObject({ ...PredictionFields, input: UploadInput, signal: Signal })
  .refine(consistent, CONSISTENT)

/** A test record's labels, which the API reveals only after that record was predicted. */
const Truth = z.strictObject({
  id: z.string(),
  labels: z.array(z.string()),
  classes: z.array(z.strictObject({ class: z.string(), label: z.string(), present: z.boolean() })),
})

export type Health = z.infer<typeof Health>
export type RecordSummary = z.infer<typeof RecordSummary>
export type Signal = z.infer<typeof Signal>
export type AbnormalClass = z.infer<typeof AbnormalClass>
export type NormalClass = z.infer<typeof NormalClass>
export type ClassPrediction = z.infer<typeof ClassPrediction>
export type RecordPrediction = z.infer<typeof RecordPrediction>
export type UploadPrediction = z.infer<typeof UploadPrediction>
export type Prediction = RecordPrediction | UploadPrediction
export type Truth = z.infer<typeof Truth>

/** The schemas, for checking values that did not come straight from a response (for example, history state). */
export const schemas = { RecordPrediction, UploadPrediction } as const

// ------------------------------------------------------------------------------------------------ errors

/**
 * unreachable: no HTTP response at all (the API is not running, or the browser blocked the call: CORS).
 * http: the API answered with an error status; message is the API's own `detail`, verbatim.
 * invalid-response: the API answered 2xx with a body that does not match the schema.
 */
export type ApiErrorKind = 'unreachable' | 'http' | 'invalid-response'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly endpoint: string     // e.g. "POST /api/predict"
  readonly status: number | null

  constructor(kind: ApiErrorKind, endpoint: string, message: string, status: number | null, cause?: unknown) {
    super(message, { cause })
    this.name = 'ApiError'
    this.kind = kind
    this.endpoint = endpoint
    this.status = status
  }
}

/** FastAPI's error body is {"detail": "<message>"}; its message, or the raw body if it has another shape. */
export function errorMessage(status: number, statusText: string, body: string): string {
  try {
    const parsed: unknown = JSON.parse(body)
    if (typeof parsed === 'object' && parsed !== null && 'detail' in parsed && typeof parsed.detail === 'string') {
      return parsed.detail
    }
  } catch {
    // not JSON: show the body as it came
  }
  const head = `HTTP ${status}${statusText ? ` ${statusText}` : ''}`
  return body.trim() ? `${head}: ${body.trim()}` : head
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

async function request<T>(method: 'GET' | 'POST', path: string, schema: z.ZodType<T>,
                          init: RequestInit = {}): Promise<T> {
  const endpoint = `${method} ${path}`
  let response: Response
  let body: string
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, method })
    body = await response.text()
  } catch (error) {
    if (isAbort(error)) throw error // the caller cancelled; not an API failure
    throw new ApiError('unreachable', endpoint, `No response from the API at ${API_BASE_URL} (${endpoint})`, null,
                       error)
  }
  if (!response.ok) {
    throw new ApiError('http', endpoint, errorMessage(response.status, response.statusText, body), response.status)
  }
  let json: unknown
  try {
    json = JSON.parse(body)
  } catch (error) {
    throw new ApiError('invalid-response', endpoint, `${endpoint} returned a body that is not JSON`, response.status,
                       error)
  }
  const parsed = schema.safeParse(json)
  if (!parsed.success) {
    throw new ApiError('invalid-response', endpoint, `${endpoint} returned a response this page does not ` +
                       `understand:\n${z.prettifyError(parsed.error)}`, response.status, parsed.error)
  }
  return parsed.data
}

// ------------------------------------------------------------------------------------------------ endpoints

const recordPath = (id: string) => `/api/records/${encodeURIComponent(id)}`

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request('GET', '/api/health', Health, { signal })
}

/** The held-out test records (PTB-XL fold 10) the API serves for the demo. */
export function getRecords(signal?: AbortSignal): Promise<RecordSummary[]> {
  return request('GET', '/api/records', z.array(RecordSummary), { signal })
}

/** A test record's 12 leads in mV, before normalisation. */
export function getSignal(id: string, signal?: AbortSignal): Promise<Signal> {
  return request('GET', `${recordPath(id)}/signal`, Signal, { signal })
}

/** Screen a held-out test record. */
export function predictRecord(id: string, signal?: AbortSignal): Promise<RecordPrediction> {
  return request('POST', '/api/predict', RecordPrediction, {
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ record_id: id }),
  })
}

/** Screen an uploaded WFDB record. The API refuses it with a 400 (or 413 if too large) and says why. */
export function predictUpload(hea: File, dat: File, signal?: AbortSignal): Promise<UploadPrediction> {
  const form = new FormData()
  form.append('hea', hea, hea.name)
  form.append('dat', dat, dat.name)
  return request('POST', '/api/predict', UploadPrediction, { signal, body: form })
}

/** A test record's labels. The API answers 409 until that record has been predicted. */
export function getTruth(id: string, signal?: AbortSignal): Promise<Truth> {
  return request('GET', `${recordPath(id)}/truth`, Truth, { signal })
}
