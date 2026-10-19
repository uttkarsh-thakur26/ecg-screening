import { describe, expect, it } from 'vitest'
import type { RecordSummary } from './api'
import { searchRecords } from './records'

const ids = (records: RecordSummary[]) => records.map((r) => r.id)
const records: RecordSummary[] = ['19', '9', '1900', '90', '219', '29'].map((id) => ({
  id, duration_s: 10, sampling_rate: 100,
}))

describe('searchRecords', () => {
  it('keeps every record, in the API order, for an empty or blank query', () => {
    expect(ids(searchRecords(records, ''))).toEqual(['19', '9', '1900', '90', '219', '29'])
    expect(ids(searchRecords(records, '   '))).toEqual(['19', '9', '1900', '90', '219', '29'])
  })

  it('puts the exact id first, then ids starting with the query, then ids containing it', () => {
    expect(ids(searchRecords(records, '9'))).toEqual(['9', '90', '19', '1900', '219', '29'])
    expect(ids(searchRecords(records, '19'))).toEqual(['19', '1900', '219'])
  })

  it('ignores surrounding spaces and finds nothing for an id that is not there', () => {
    expect(ids(searchRecords(records, ' 1900 '))).toEqual(['1900'])
    expect(searchRecords(records, '12345')).toEqual([])
    expect(searchRecords(records, 'abc')).toEqual([])
  })

  it('does not reorder the list it was given', () => {
    const before = ids(records)
    searchRecords(records, '9')
    expect(ids(records)).toEqual(before)
  })
})
