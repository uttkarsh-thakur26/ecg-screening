// The client against the running API: npm run test:contract, with python -m src.api.main serving. Every response
// passes through the client's schemas (strict: an added, renamed or missing field fails), and the checks below hold
// the API to what the screens rely on. Predicting a record unlocks its labels on that server, as in the demo.
import { describe, expect, it } from 'vitest'
import { ApiError, getHealth, getRecords, getSignal, getTruth, predictRecord, predictUpload } from './api'

async function failure(promise: Promise<unknown>): Promise<ApiError> {
  const error = await promise.then(() => null, (e: unknown) => e)
  if (!(error instanceof ApiError)) throw new Error(`expected an ApiError, got ${String(error)}`)
  return error
}

describe('the client against the running API', () => {
  it('reads the health of a loaded model', async () => {
    expect((await getHealth()).model_loaded).toBe(true)
  })

  it('reads the test records, a signal, a prediction and then the labels', async () => {
    const records = await getRecords()
    expect(records.length).toBeGreaterThan(0)
    expect(new Set(records.map((r) => r.id)).size).toBe(records.length)
    const record = records[0]!
    const samples = record.duration_s * record.sampling_rate

    const signal = await getSignal(record.id)
    expect(signal.sampling_rate).toBe(record.sampling_rate)
    expect(signal.data.every((lead) => lead.length === samples)).toBe(true)

    const prediction = await predictRecord(record.id)
    expect(prediction.input.record_id).toBe(record.id)
    expect(prediction.attention).toHaveLength(samples)
    // NORM has a probability only; every other class has a threshold and a flag.
    expect(prediction.predictions.filter((c) => c.threshold === null).map((c) => c.class)).toEqual(['NORM'])
    expect(prediction.summary)
      .toBe(`${prediction.flagged_count} of ${prediction.abnormal_count} abnormal categories flagged`)
    expect(prediction.abnormal_count).toBe(4)

    const truth = await getTruth(record.id)
    expect(truth.id).toBe(record.id)
    expect(truth.classes.map((c) => c.class)).toEqual(prediction.predictions.map((c) => c.class))
  })

  it("passes on the API's own message when it refuses a request", async () => {
    const unknown = await failure(predictRecord('0'))
    expect(unknown).toMatchObject({ kind: 'http', status: 404 })
    expect(unknown.message).toContain('held-out test records')

    const upload = await failure(predictUpload(new File(['not a WFDB header'], 'x.hea'),
                                              new File([new Uint8Array(8)], 'x.dat')))
    expect(upload).toMatchObject({ kind: 'http', status: 400, endpoint: 'POST /api/predict' })
    expect(upload.message.length).toBeGreaterThan(0)
  })
})
