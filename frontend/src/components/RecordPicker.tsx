import { useId, useMemo } from 'react'
import type { RecordSummary } from '../lib/api'
import { formatCount } from '../lib/format'
import { searchRecords } from '../lib/records'
import { ApiErrorPanel } from './ApiErrorPanel'

export type RecordsState =
  | { status: 'loading' }
  | { status: 'error'; error: unknown }
  | { status: 'ready'; records: RecordSummary[] }

interface Props {
  state: RecordsState
  query: string
  selectedId: string | null
  showShape: boolean // each row shows its duration and rate (when they differ; else the caption says them once)
  disabled: boolean
  onQuery: (query: string) => void
  onSelect: (id: string) => void
  onRetry: () => void
}

/** The held-out test records from GET /api/records, searchable by id. One can be selected. */
export function RecordPicker({ state, query, selectedId, showShape, disabled, onQuery, onSelect,
                              onRetry }: Props) {
  const searchId = useId()
  const records = state.status === 'ready' ? state.records : null
  const matches = useMemo(() => (records ? searchRecords(records, query) : []), [records, query])

  if (state.status === 'error') {
    return <ApiErrorPanel title="Could not load the held-out test records" error={state.error} onRetry={onRetry} />
  }

  return (
    <div className="rounded-lg border border-line bg-surface">
      <div className="border-b border-line p-2">
        <label htmlFor={searchId} className="text-sm font-medium">
          Search by record ID
        </label>
        <input
          id={searchId}
          type="search"
          inputMode="numeric"
          autoComplete="off"
          spellCheck={false}
          value={query}
          disabled={disabled || records === null}
          onChange={(e) => onQuery(e.currentTarget.value)}
          className="mt-1 h-5 w-full rounded-md border border-line-strong bg-surface px-1 text-ink
                     focus-visible:border-accent focus-visible:outline-2 focus-visible:outline-accent
                     disabled:bg-canvas"
        />
        <p aria-live="polite" className="mt-1 text-sm text-ink-muted">
          {records === null
            ? 'Loading the held-out test records from the API…'
            : query.trim() === ''
              ? `${formatCount(records.length)} records`
              : `${formatCount(matches.length)} of ${formatCount(records.length)} records match`}
        </p>
      </div>

      {records !== null && (
        <div role="radiogroup" aria-label="Held-out test records" className="max-h-30 overflow-y-auto">
          {records.length === 0 ? (
            <p className="p-2 text-sm">The API returned no test records.</p>
          ) : matches.length === 0 ? (
            <p className="p-2 text-sm text-ink-muted">No record ID contains &ldquo;{query.trim()}&rdquo;.</p>
          ) : (
            matches.map((r) => (
              <label
                key={r.id}
                className="flex cursor-pointer items-center justify-between gap-2 border-b border-line px-2 py-1
                           last:border-b-0 hover:bg-canvas
                           has-checked:bg-accent/10 has-checked:text-accent
                           has-checked:shadow-[inset_4px_0_0_var(--color-accent)]
                           has-[:focus-visible]:outline-2 has-[:focus-visible]:-outline-offset-2
                           has-[:focus-visible]:outline-accent"
              >
                <span className="font-medium">
                  <input
                    type="radio"
                    name="test-record"
                    value={r.id}
                    checked={selectedId === r.id}
                    disabled={disabled}
                    onChange={() => onSelect(r.id)}
                    className="sr-only"
                  />
                  {r.id}
                </span>
                {showShape && (
                  <span className="text-sm text-ink-muted">
                    {r.duration_s} s · {r.sampling_rate} Hz
                  </span>
                )}
              </label>
            ))
          )}
        </div>
      )}
    </div>
  )
}
