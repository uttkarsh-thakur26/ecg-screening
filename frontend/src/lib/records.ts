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
