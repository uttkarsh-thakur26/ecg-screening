import type { RecordSummary } from './api'

/**
 * The records whose id contains the query: the exact id first, then ids that start with the query, then the rest,
 * each group in the API's order. An empty query keeps every record.
 */
export function searchRecords(records: readonly RecordSummary[], query: string): RecordSummary[] {
  const q = query.trim()
  if (q === '') return [...records]
  const rank = (id: string) => (id === q ? 0 : id.startsWith(q) ? 1 : 2)
  return records.filter((r) => r.id.includes(q)).sort((a, b) => rank(a.id) - rank(b.id)) // sort is stable
}

/** The duration and sampling rate every record has, so a caption can say it once; null if they differ or the list is
 *  empty, and then each record shows its own. */
export function commonShape(records: readonly RecordSummary[]):
  Pick<RecordSummary, 'duration_s' | 'sampling_rate'> | null {
  const first = records[0]
  if (first === undefined) return null
  const same = records.every((r) => r.duration_s === first.duration_s && r.sampling_rate === first.sampling_rate)
  return same ? { duration_s: first.duration_s, sampling_rate: first.sampling_rate } : null
}
