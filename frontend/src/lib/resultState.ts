// What the Select screen hands to the Result route: the API's prediction, unchanged. It travels in the browser's
// history state, so it survives a reload of the same tab; it is never stored anywhere else. A test record's result
// route carries the record id in the URL as well, an upload's does not (the API keeps no uploads).

import { schemas, type RecordPrediction, type UploadPrediction } from './api'

export type ResultState =
  | { source: 'test_record'; prediction: RecordPrediction }
  | { source: 'upload'; prediction: UploadPrediction; files: { hea: string; dat: string } }

/** The state if it is a result of the expected kind (and, for a test record, of the record in the URL); else null. */
export function readResultState(state: unknown, source: ResultState['source'],
                                recordId?: string): ResultState | null {
  if (typeof state !== 'object' || state === null || !('source' in state) || state.source !== source) return null
  const prediction = 'prediction' in state ? state.prediction : undefined
  if (source === 'test_record') {
    const parsed = schemas.RecordPrediction.safeParse(prediction)
    return parsed.success && parsed.data.input.record_id === recordId
      ? { source, prediction: parsed.data } : null
  }
  const parsed = schemas.UploadPrediction.safeParse(prediction)
  const files = 'files' in state ? state.files : undefined
  if (!parsed.success || typeof files !== 'object' || files === null || !('hea' in files) || !('dat' in files) ||
      typeof files.hea !== 'string' || typeof files.dat !== 'string') return null
  return { source, prediction: parsed.data, files: { hea: files.hea, dat: files.dat } }
}
