import { useEffect, useState, type ReactNode } from 'react'
import { Link, useLocation, useParams } from 'react-router'
import { ApiErrorPanel } from '../components/ApiErrorPanel'
import { ClassResults } from '../components/ClassResults'
import { EcgStrip } from '../components/EcgStrip'
import { ApiError, getSignal, getTruth, predictRecord, type Prediction, type RecordPrediction, type Signal,
         type Truth, type UploadPrediction } from '../lib/api'
import { stripProblem } from '../lib/ecgPlot'
import { readResultState, type ResultState } from '../lib/resultState'
import { agreement, presentByClass, summaryLine, truthProblem } from '../lib/results'
import { paths } from '../paths'

type Load<T> = { status: 'loading' } | { status: 'ready'; value: T } | { status: 'error'; error: unknown }
type Reveal = { status: 'hidden' } | Load<Truth>

const LOADING = { status: 'loading' } as const

/**
 * Screen 2: the signal the model read, with its attention map, and the API's prediction for every class. A test
 * record's prediction comes from the Select screen (in the history state); opened any other way, for example from a
 * bookmarked /result/record/<id>, the page asks the API to predict the record again. An upload's result exists only
 * in the history state, as the API keeps no uploads.
 */
export function ResultPage({ source }: { source: ResultState['source'] }) {
  const { recordId } = useParams()
  const { state } = useLocation()
  if (source === 'upload') {
    const handed = readResultState(state, 'upload')
    return <UploadResult handed={handed?.source === 'upload' ? handed : null} />
  }
  if (recordId === undefined) throw new Error(`The record result route has no record id (${paths.recordResultPattern})`)
  const handed = readResultState(state, 'test_record', recordId)
  return (
    <RecordResult key={recordId} recordId={recordId}
                  handed={handed?.source === 'test_record' ? handed.prediction : null} />
  )
}

function RecordResult({ recordId, handed }: { recordId: string; handed: RecordPrediction | null }) {
  const [prediction, setPrediction] = useState<Load<RecordPrediction>>(
    () => (handed === null ? LOADING : { status: 'ready', value: handed }))
  const [signal, setSignal] = useState<Load<Signal>>(LOADING)
  const [truth, setTruth] = useState<Reveal>({ status: 'hidden' })

  // No prediction was handed over (a direct link or a new tab): run the model on the record now.
  const predicting = prediction.status === 'loading'
  useEffect(() => {
    if (!predicting) return
    return load(setPrediction, (abort) => predictRecord(recordId, abort))
  }, [recordId, predicting])

  const fetchingSignal = signal.status === 'loading'
  useEffect(() => {
    if (!fetchingSignal) return
    return load(setSignal, (abort) => getSignal(recordId, abort))
  }, [recordId, fetchingSignal])

  // The labels, only when asked for, and only once the prediction is in (the API refuses them before it, with a 409).
  const revealing = truth.status === 'loading' && prediction.status === 'ready'
  useEffect(() => {
    if (!revealing) return
    return load(setTruth, (abort) => getTruth(recordId, abort))
  }, [recordId, revealing])

  let reveal: ReactNode = null
  let present: Map<string, boolean> | null = null
  if (prediction.status === 'ready') {
    if (truth.status === 'hidden' || truth.status === 'loading') {
      reveal = (
        <button
          type="button"
          onClick={() => setTruth(LOADING)}
          disabled={truth.status === 'loading'}
          aria-busy={truth.status === 'loading'}
          className="h-5 rounded-md border border-line-strong bg-surface px-2 text-sm font-medium
                     enabled:hover:border-ink focus-visible:outline-2 focus-visible:outline-offset-2
                     focus-visible:outline-accent aria-busy:cursor-wait"
        >
          {truth.status === 'loading' ? 'Revealing the ground truth…' : 'Reveal ground truth'}
        </button>
      )
    } else if (truth.status === 'error') {
      // 409: the API process has not predicted this record since it started (it restarted, or this tab kept the
      // prediction across a reload). Trying again runs the model on the record first, then asks for the labels.
      const locked = truth.error instanceof ApiError && truth.error.status === 409
      reveal = (
        <ApiErrorPanel
          title={`Could not reveal the labels of record ${recordId}`}
          error={truth.error}
          onRetry={() => {
            if (locked) setPrediction(LOADING)
            setTruth(LOADING)
          }}
        />
      )
    } else {
      const problem = truthProblem(recordId, prediction.value.predictions, truth.value)
      if (problem !== null) {
        reveal = <ApiErrorPanel title={`Could not show the labels of record ${recordId}`} error={problem} />
      } else {
        present = presentByClass(truth.value)
        reveal = <TruthSummary recordId={recordId} prediction={prediction.value} truth={truth.value} />
      }
    }
  }

  // One error at a time (an unknown record fails both requests with the same message); trying again repeats every
  // request that failed.
  const failed = prediction.status === 'error' || signal.status === 'error'
  function retry() {
    if (prediction.status === 'error') setPrediction(LOADING)
    if (signal.status === 'error') setSignal(LOADING)
  }
  return (
    <ResultFrame eyebrow="Held-out test record" heading={`Record ${recordId}`}>
      {prediction.status === 'error' ? (
        <ApiErrorPanel title={`Could not screen record ${recordId}`} error={prediction.error} onRetry={retry} />
      ) : signal.status === 'error' && (
        <ApiErrorPanel title={`Could not load the signal of record ${recordId}`} error={signal.error} onRetry={retry} />
      )}
      {!failed && (prediction.status === 'loading' || signal.status === 'loading') && (
        <p role="status" className="text-ink-muted">
          {prediction.status === 'loading'
            ? `Screening record ${recordId}…`
            : `Loading the signal of record ${recordId}…`}
        </p>
      )}
      {prediction.status === 'ready' && signal.status === 'ready' && (
        <Results prediction={prediction.value} signal={signal.value} present={present} reveal={reveal} />
      )}
    </ResultFrame>
  )
}

function UploadResult({ handed }: { handed: Extract<ResultState, { source: 'upload' }> | null }) {
  if (handed === null) {
    return (
      <ResultFrame eyebrow="Upload" heading="Uploaded record">
        <p>
          There is no upload to show here. The API keeps no uploads, so an upload&rsquo;s result exists only in the
          browser tab that screened it; upload the files again from the Select screen.
        </p>
      </ResultFrame>
    )
  }
  const { prediction, files } = handed
  return (
    <ResultFrame eyebrow="Upload" heading="Uploaded record"
                 detail={<UploadDetail prediction={prediction} files={files} />}>
      {/* The signal the API predicted on: the upload as the model read it, resampled to its rate if needed. */}
      <Results prediction={prediction} signal={prediction.signal} present={null} reveal={null} />
    </ResultFrame>
  )
}

function UploadDetail({ prediction, files }: { prediction: UploadPrediction; files: { hea: string; dat: string } }) {
  const { input, signal } = prediction
  return (
    <p className="mt-0.5 text-sm text-ink-muted break-all">
      {files.hea} · {files.dat} · recorded at {input.sampling_rate_hz} Hz
      {input.resampled && `, shown as the model read it: resampled to ${signal.sampling_rate} Hz`}
      {!input.checksums_verified && '. The header has no checksums, so the .dat could not be checked against it'}
    </p>
  )
}

function ResultFrame({ eyebrow, heading, detail, children }:
                     { eyebrow: string; heading: string; detail?: ReactNode; children: ReactNode }) {
  return (
    <div className="mx-auto max-w-5xl px-3 py-3">
      <title>{`${heading} · CardioScreen`}</title>
      <div className="flex flex-wrap items-end justify-between gap-x-3 gap-y-1">
        <div>
          <p className="text-sm text-ink-muted">{eyebrow}</p>
          <h1 className="text-2xl font-semibold tracking-tight">{heading}</h1>
        </div>
        <Link to={paths.select} className="text-sm font-medium text-accent underline underline-offset-2
                                           hover:text-accent-strong">
          Screen another ECG
        </Link>
      </div>
      {detail}
      <div className="mt-2">{children}</div>
    </div>
  )
}

/** The waveform, the summary and the per-class rows, faded in once, when they load. */
function Results({ prediction, signal, present, reveal }:
                 { prediction: Prediction; signal: Signal; present: ReadonlyMap<string, boolean> | null;
                   reveal: ReactNode }) {
  const problem = stripProblem(signal, prediction.attention)
  return (
    <div className="animate-fade-in motion-reduce:animate-none">
      <section aria-label="ECG" className="rounded-lg border border-line bg-surface p-2">
        {problem === null ? (
          <EcgStrip signal={signal} attention={prediction.attention} attentionClass={prediction.attention_class}
                    attentionRawPeak={prediction.attention_raw_peak} />
        ) : (
          <ApiErrorPanel title="The signal and the attention map from the API cannot be drawn together"
                         error={problem} />
        )}
      </section>

      <p className="mt-2 text-lg font-medium">{summaryLine(prediction)}</p>

      <section aria-label="Result per class" className="mt-2 rounded-lg border border-line bg-surface px-2 py-1">
        <ClassResults predictions={prediction.predictions} present={present} />
      </section>

      {reveal !== null && <div className="mt-2">{reveal}</div>}

      <p className="mt-3 text-xs text-ink-muted">
        Model version {prediction.model_version} · inference {prediction.inference_ms} ms
      </p>
    </div>
  )
}

function TruthSummary({ recordId, prediction, truth }:
                      { recordId: string; prediction: RecordPrediction; truth: Truth }) {
  const labelled = truth.classes.filter((c) => c.present).map((c) => `${c.label} (${c.class})`)
  const present = presentByClass(truth)
  const disagreements = prediction.predictions.flatMap((c) => {
    const verdict = agreement(c.flagged, present.get(c.class) === true)
    return verdict === 'missed' ? [`${c.class} is in the labels but was not flagged`]
         : verdict === 'false-flag' ? [`${c.class} was flagged but is not in the labels`]
         : []
  })
  return (
    <div role="status" className="text-sm">
      <p>
        {labelled.length > 0
          ? `PTB-XL labels for record ${recordId}: ${labelled.join(', ')}.`
          : `PTB-XL gives record ${recordId} no diagnostic superclass label.`}
      </p>
      <p className="mt-0.5 font-medium">
        {disagreements.length === 0
          ? 'The flags agree with the labels for every abnormal category.'
          : `The flags disagree with the labels: ${disagreements.join('; ')}.`}
      </p>
    </div>
  )
}

/** Runs one request and puts its outcome in state; the returned cleanup cancels it, and a cancelled request's
 *  outcome is dropped. */
function load<T>(set: (value: Load<T>) => void, request: (abort: AbortSignal) => Promise<T>): () => void {
  const controller = new AbortController()
  request(controller.signal).then(
    (value) => { if (!controller.signal.aborted) set({ status: 'ready', value }) },
    (error: unknown) => { if (!controller.signal.aborted) set({ status: 'error', error }) },
  )
  return () => controller.abort()
}
