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

// ------------------------------------------------------------------------------------------------ metrics.json

// GET /api/metrics serves reports/metrics.json byte for byte: the one-time evaluation of the release model on the
// held-out test split (src/evaluation/evaluate_test.py). These schemas type the parts the Performance screen shows, as
// that file writes them; the parts it does not show (environment, checks, validation, ...) are not read. The screen
// shows these numbers as they are and computes none of its own beyond rounding and counting.

const Count = z.number().int().nonnegative()

/** A bootstrapped metric: its value on all records and the percentile interval over resamples of patients. */
const Estimate = z
  .strictObject({ value: z.number(), ci_low: z.number(), ci_high: z.number() })
  .refine((e) => e.ci_low <= e.ci_high, { message: 'ci_low is above ci_high' })

const RankingFields = { auroc: Estimate, auprc: Estimate }
const ClassRanking = z.strictObject({ ...RankingFields, n_positive: Count, n_negative: Count, prevalence: Probability })
const MacroRanking = z.strictObject({ ...RankingFields, prevalence: Probability })

const PointFields = { sensitivity: Estimate, specificity: Estimate, precision: Estimate, npv: Estimate, f1: Estimate }
/** One class at one operating point: its frozen threshold, the confusion counts and the metrics they give. */
const ClassPoint = z.strictObject({ threshold: Probability, tp: Count, fp: Count, fn: Count, tn: Count, ...PointFields })
const MacroPoint = z.strictObject(PointFields)

const PointCommon = { rule: z.string(), classes: z.array(z.string()).min(1), per_class: z.record(z.string(), ClassPoint) }
const MaxF1Point = z.strictObject({ ...PointCommon, macro: MacroPoint, macro_abnormal: MacroPoint })
const HighSensitivityPoint = z.strictObject({ ...PointCommon, macro_abnormal: MacroPoint,
                                              target_sensitivity: Probability })

/** Expected calibration error before and after the frozen calibrator; interval_note says why the intervals of ECE are
 *  not confidence intervals (the screen shows the point values only). */
const EceFields = { uncalibrated: Estimate, calibrated: Estimate, calibrated_minus_uncalibrated: Estimate }
const ClassEce = z.object({ ...EceFields, calibrator: z.object({ method: z.enum(['temperature', 'platt']) }) })

const SeedSpread = z.strictObject({ mean: z.number(), std: z.number().nonnegative() })

const BaselineRanking = z.object({ macro: z.strictObject(RankingFields), macro_abnormal: z.strictObject(RankingFields) })

/** The test-set figures, project-relative PNG paths. The API serves each at GET /api/figures/{name}. */
const Figures = z.strictObject({
  roc: z.string(),
  pr: z.string(),
  confusion_max_f1: z.string(),
  confusion_high_sensitivity: z.string(),
  reliability: z.string(),
})

const sameSet = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && [...a].sort().join('\n') === [...b].sort().join('\n')

const MetricsReport = z
  .object({
    description: z.string(),
    date_evaluated: z.string(),
    git: z.object({ commit: z.string().regex(/^[0-9a-f]{40}$/), dirty: z.boolean() }),
    fold10_access_log: z.array(z.object({ opened_at: z.string(), rerun_reason: z.string().nullable() })).min(1),
    reruns: z.array(z.unknown()),
    dataset: z.object({ name: z.string(), version: z.string() }),
    split: z.literal('test'),
    folds: z.array(z.number().int()).min(1),
    records: Count,
    patients: Count,
    classes: z.array(z.string()).min(1),
    class_descriptions: z.record(z.string(), z.string()),
    abnormal_classes: z.array(z.string()).min(1),
    model: z.object({ version: z.string(), tag: z.string(), run_id: z.string() }),
    bootstrap: z.object({ unit: z.string(), resamples: Count, ci_level: z.number().gt(0).lt(1) }),
    ranking: z.object({
      per_class: z.record(z.string(), ClassRanking),
      macro: MacroRanking,
      macro_abnormal: MacroRanking,
    }),
    operating_points: z.object({ max_f1: MaxF1Point, high_sensitivity: HighSensitivityPoint }),
    calibration_error: z.object({
      bins: Count,
      interval_note: z.string(),
      per_class: z.record(z.string(), ClassEce),
      macro: z.strictObject(EceFields),
    }),
    baselines: z.object({
      feature_gbdt: z.object({ description: z.string(), ranking: BaselineRanking }),
      cnn_minus_feature_gbdt: z.object({ description: z.string(), ranking: BaselineRanking }),
    }),
    seed_stability: z.object({
      description: z.string(),
      split: z.literal('val'), // validation: the screen labels these numbers as such
      std_definition: z.string(),
      runs: z.array(z.object({ run_id: z.string(), seed: z.number().int() })).min(2),
      macro_auroc: SeedSpread,
      macro_auprc: SeedSpread,
    }),
    figures: Figures,
  })
  .superRefine((m, ctx) => {
    // Every per-class table must cover exactly the classes the report lists, so no row can be missing or extra.
    const problems: string[] = []
    const covers = (what: string, keys: readonly string[], expected: readonly string[]) => {
      if (!sameSet(keys, expected)) problems.push(`${what} covers ${keys.join(', ')}; expected ${expected.join(', ')}`)
    }
    covers('class_descriptions', Object.keys(m.class_descriptions), m.classes)
    if (!m.abnormal_classes.every((c) => m.classes.includes(c))) problems.push('abnormal_classes are not all classes')
    covers('ranking.per_class', Object.keys(m.ranking.per_class), m.classes)
    covers('operating_points.max_f1.classes', m.operating_points.max_f1.classes, m.classes)
    covers('operating_points.max_f1.per_class', Object.keys(m.operating_points.max_f1.per_class), m.classes)
    covers('operating_points.high_sensitivity.classes', m.operating_points.high_sensitivity.classes,
           m.abnormal_classes)
    covers('operating_points.high_sensitivity.per_class', Object.keys(m.operating_points.high_sensitivity.per_class),
           m.abnormal_classes)
    covers('calibration_error.per_class', Object.keys(m.calibration_error.per_class), m.classes)
    for (const message of problems) ctx.addIssue({ code: 'custom', message })
  })

export type Estimate = z.infer<typeof Estimate>
export type ClassPoint = z.infer<typeof ClassPoint>
export type MacroPoint = z.infer<typeof MacroPoint>
export type MetricsReport = z.infer<typeof MetricsReport>
export type FigureName = keyof z.infer<typeof Figures>

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

/** The schemas, for checking values that did not come straight from a response (for example, history state, or
 *  reports/metrics.json read from disk in a test). */
export const schemas = { RecordPrediction, UploadPrediction, MetricsReport } as const

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

/** One call: a 2xx response's body, read with `read`. No response at all is an 'unreachable' error, and an error
 *  status an 'http' error carrying the API's own message. */
async function call<B>(method: 'GET' | 'POST', path: string, init: RequestInit,
                       read: (response: Response) => Promise<B>): Promise<{ endpoint: string; status: number; body: B }> {
  const endpoint = `${method} ${path}`
  let response: Response
  let body: { ok: true; value: B } | { ok: false; text: string }
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, method })
    body = response.ok ? { ok: true, value: await read(response) } : { ok: false, text: await response.text() }
  } catch (error) {
    if (isAbort(error)) throw error // the caller cancelled; not an API failure
    throw new ApiError('unreachable', endpoint, `No response from the API at ${API_BASE_URL} (${endpoint})`, null,
                       error)
  }
  if (!body.ok) {
    throw new ApiError('http', endpoint, errorMessage(response.status, response.statusText, body.text),
                       response.status)
  }
  return { endpoint, status: response.status, body: body.value }
}

async function request<T>(method: 'GET' | 'POST', path: string, schema: z.ZodType<T>,
                          init: RequestInit = {}): Promise<T> {
  const { endpoint, status, body } = await call(method, path, init, (r) => r.text())
  let json: unknown
  try {
    json = JSON.parse(body)
  } catch (error) {
    throw new ApiError('invalid-response', endpoint, `${endpoint} returned a body that is not JSON`, status, error)
  }
  const parsed = schema.safeParse(json)
  if (!parsed.success) {
    throw new ApiError('invalid-response', endpoint, `${endpoint} returned a response this page does not ` +
                       `understand:\n${z.prettifyError(parsed.error)}`, status, parsed.error)
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

/** reports/metrics.json, the one-time test-set evaluation. The API answers 404 when it has not been run. */
export function getMetrics(signal?: AbortSignal): Promise<MetricsReport> {
  return request('GET', '/api/metrics', MetricsReport, { signal })
}

/** The address of a test-set figure, for opening it at full size. */
export function figureUrl(name: FigureName): string {
  return `${API_BASE_URL}/api/figures/${encodeURIComponent(name)}`
}

/** A test-set figure listed in reports/metrics.json, as the PNG the evaluation wrote. */
export async function getFigure(name: FigureName, signal?: AbortSignal): Promise<Blob> {
  const { endpoint, status, body } = await call('GET', `/api/figures/${encodeURIComponent(name)}`, { signal },
                                                (r) => r.blob())
  if (body.type !== 'image/png') {
    throw new ApiError('invalid-response', endpoint, `${endpoint} returned ${body.type || 'a body with no type'}, ` +
                       'not a PNG image', status)
  }
  return body
}
