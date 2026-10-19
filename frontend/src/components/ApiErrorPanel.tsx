import { API_BASE_URL, ApiError } from '../lib/api'

interface Props {
  title: string
  error: unknown
  onRetry?: () => void
}

/**
 * An API failure, said plainly. The API's own message is shown verbatim. Neutral colours: amber and red are kept for
 * results, so an error never looks like a finding.
 */
export function ApiErrorPanel({ title, error, onRetry }: Props) {
  return (
    <div role="alert" className="rounded-lg border border-l-4 border-line border-l-ink bg-surface p-2">
      <p className="font-medium">{title}</p>
      <ErrorBody error={error} />
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 rounded-md border border-line-strong bg-surface px-2 py-1 text-sm font-medium
                     hover:border-ink focus-visible:outline-2 focus-visible:outline-offset-2
                     focus-visible:outline-accent"
        >
          Try again
        </button>
      )}
    </div>
  )
}

function ErrorBody({ error }: { error: unknown }) {
  if (!(error instanceof ApiError)) {
    return <p className="mt-1 text-sm whitespace-pre-wrap">{String(error)}</p>
  }
  if (error.kind === 'unreachable') {
    return (
      <div className="mt-1 space-y-1 text-sm">
        <p>
          The API is not responding at {API_BASE_URL}. Start it from the project folder
          with <code className="rounded-sm bg-canvas px-0.5">python -m src.api.main</code>, wait until it has loaded
          the model, then try again.
        </p>
        <p className="text-ink-muted">
          If it is already running, the browser may be blocking the call: the API accepts requests only from the
          origin in api.cors_origins (config/default.yaml), and this page is on {window.location.origin}.
        </p>
      </div>
    )
  }
  return (
    <div className="mt-1 text-sm">
      <p className="whitespace-pre-wrap">{error.message}</p>
      <p className="mt-1 text-ink-muted">
        {error.status !== null ? `HTTP ${error.status} from ` : ''}{error.endpoint}
      </p>
    </div>
  )
}
