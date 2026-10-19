import { Link, useLocation, useParams } from 'react-router'
import { readResultState, type ResultState } from '../lib/resultState'
import { paths } from '../paths'

/**
 * Screen 2. For now it confirms which input was screened and shows the API's summary line; the waveform and the
 * per-class rows come in the next build step.
 */
export function ResultPage({ source }: { source: ResultState['source'] }) {
  const { recordId } = useParams()
  const result = readResultState(useLocation().state, source, recordId)
  const heading = source === 'test_record' ? `Record ${recordId ?? ''}` : 'Uploaded record'

  return (
    <div className="mx-auto max-w-2xl px-3 py-5">
      <title>{`${heading} · CardioScreen`}</title>
      <p className="text-sm text-ink-muted">
        {source === 'test_record' ? 'Held-out test record' : 'Upload'}
      </p>
      <h1 className="text-2xl font-semibold tracking-tight">{heading}</h1>

      {result === null ? (
        <p className="mt-3">
          There is no prediction to show here. Predictions are not stored; screen the record again from the
          Select screen.
        </p>
      ) : (
        <>
          {result.source === 'upload' && (
            <p className="mt-1 text-sm text-ink-muted break-all">
              {result.files.hea} · {result.files.dat}
            </p>
          )}
          <p className="mt-3 text-lg font-medium">{result.prediction.summary}</p>
          <p className="mt-2 text-sm text-ink-muted">
            The waveform and the per-class results are added to this screen in the next build step.
          </p>
        </>
      )}

      <Link
        to={paths.select}
        className="mt-4 inline-block text-sm font-medium text-accent underline underline-offset-2
                   hover:text-accent-strong"
      >
        Screen another ECG
      </Link>
    </div>
  )
}
