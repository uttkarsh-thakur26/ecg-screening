import { useEffect, useState } from 'react'
import { figureUrl, getFigure, type FigureName } from '../lib/api'
import { ApiErrorPanel } from './ApiErrorPanel'

type FigureState = { status: 'loading' } | { status: 'ready'; url: string } | { status: 'error'; error: unknown }

interface Props {
  name: FigureName // a figure listed in reports/metrics.json, served at GET /api/figures/{name}
  caption: string
  alt: string
}

/**
 * A test-set figure from the evaluation, as the API serves it, with its caption. Loading, a failed request and an
 * image the browser cannot decode each show as such; the figure is never replaced by anything else.
 */
export function ReportFigure({ name, caption, alt }: Props) {
  const [state, setState] = useState<FigureState>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let url: string | null = null
    getFigure(name, controller.signal).then(
      (blob) => {
        if (controller.signal.aborted) return
        url = URL.createObjectURL(blob)
        setState({ status: 'ready', url })
      },
      (error: unknown) => { if (!controller.signal.aborted) setState({ status: 'error', error }) },
    )
    return () => {
      controller.abort()
      if (url !== null) URL.revokeObjectURL(url)
    }
  }, [name, attempt])

  return (
    <figure className="rounded-lg border border-line bg-surface p-2">
      {state.status === 'ready' ? (
        <img
          src={state.url}
          alt={alt}
          className="block h-auto w-full"
          onError={() => setState({ status: 'error',
                                    error: new Error(`The browser could not display GET /api/figures/${name}`) })}
        />
      ) : state.status === 'loading' ? (
        <p role="status" className="flex h-24 items-center justify-center text-sm text-ink-muted">
          Loading the figure…
        </p>
      ) : (
        <ApiErrorPanel
          title={`Could not load the figure: ${caption}`}
          error={state.error}
          onRetry={() => {
            setState({ status: 'loading' })
            setAttempt((n) => n + 1)
          }}
        />
      )}
      <figcaption className="mt-1 text-sm">
        {caption}{' '}
        <a href={figureUrl(name)} target="_blank" rel="noreferrer"
           className="font-medium text-accent underline underline-offset-2 hover:text-accent-strong">
          Open full size
        </a>
      </figcaption>
    </figure>
  )
}
