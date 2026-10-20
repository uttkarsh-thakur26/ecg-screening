import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { ApiErrorPanel } from '../components/ApiErrorPanel'
import { RecordPicker, type RecordsState } from '../components/RecordPicker'
import { UploadZone } from '../components/UploadZone'
import { ApiError, getRecords, predictRecord, predictUpload } from '../lib/api'
import { commonShape } from '../lib/records'
import type { ResultState } from '../lib/resultState'
import { EMPTY_UPLOAD, addFiles, type UploadPair } from '../lib/uploads'
import { paths } from '../paths'

type Run = { status: 'idle' } | { status: 'running' } | { status: 'error'; title: string; error: unknown }

/** Screen 1: choose one input (an uploaded record or a held-out test record) and send it to the model. */
export function SelectPage() {
  const navigate = useNavigate()
  const [records, setRecords] = useState<RecordsState>({ status: 'loading' })
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [query, setQuery] = useState('')
  // One input at a time: selecting a record removes the uploaded files, adding a file clears the record selection.
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [upload, setUpload] = useState<UploadPair>(EMPTY_UPLOAD)
  const [uploadMessage, setUploadMessage] = useState<string | null>(null)
  const [run, setRun] = useState<Run>({ status: 'idle' })
  const runController = useRef<AbortController | null>(null)
  const actions = useRef<HTMLDivElement>(null)
  const running = run.status === 'running'

  useEffect(() => {
    const controller = new AbortController()
    getRecords(controller.signal).then(
      (list) => setRecords({ status: 'ready', records: list }),
      (error: unknown) => { if (!controller.signal.aborted) setRecords({ status: 'error', error }) },
    )
    return () => controller.abort()
  }, [loadAttempt])

  // A file dropped beside the drop zone would otherwise make the browser open it in place of the app.
  useEffect(() => {
    const keepPage = (e: DragEvent) => e.preventDefault()
    window.addEventListener('dragover', keepPage)
    window.addEventListener('drop', keepPage)
    return () => {
      window.removeEventListener('dragover', keepPage)
      window.removeEventListener('drop', keepPage)
      runController.current?.abort()
    }
  }, [])

  // A failed run's message sits just above the button, which can be below the fold: bring both into view.
  useEffect(() => {
    if (run.status === 'error') actions.current?.scrollIntoView({ block: 'nearest' })
  }, [run])

  function retryRecords() {
    setRecords({ status: 'loading' })
    setLoadAttempt((n) => n + 1)
  }

  function selectRecord(id: string) {
    setSelectedId(id)
    setUpload(EMPTY_UPLOAD)
    setUploadMessage(null)
    setRun({ status: 'idle' })
  }

  function takeFiles(files: File[]) {
    const result = addFiles(upload, files)
    if (!result.ok) {
      setUploadMessage(result.message)
      return
    }
    setUpload(result.pair)
    setUploadMessage(null)
    setSelectedId(null)
    setRun({ status: 'idle' })
  }

  function clearFiles() {
    setUpload(EMPTY_UPLOAD)
    setUploadMessage(null)
    setRun({ status: 'idle' })
  }

  async function screen() {
    if (selectedId === null && !(upload.hea && upload.dat)) return // the button is disabled then
    runController.current?.abort()
    const controller = new AbortController()
    runController.current = controller
    setRun({ status: 'running' })
    try {
      if (selectedId !== null) {
        const prediction = await predictRecord(selectedId, controller.signal)
        const state: ResultState = { source: 'test_record', prediction }
        navigate(paths.recordResult(selectedId), { state })
      } else if (upload.hea && upload.dat) {
        const prediction = await predictUpload(upload.hea, upload.dat, controller.signal)
        const state: ResultState = { source: 'upload', prediction,
                                     files: { hea: upload.hea.name, dat: upload.dat.name } }
        navigate(paths.uploadResult, { state })
      }
    } catch (error) {
      if (controller.signal.aborted) return
      const rejected = error instanceof ApiError && error.kind === 'http'
      const title = selectedId !== null
        ? `Could not screen record ${selectedId}`
        : rejected ? 'The API did not accept the upload' : 'Could not screen the uploaded record'
      setRun({ status: 'error', title, error })
    }
  }

  // Every test record has the same length and rate, so the caption says them once instead of on every row.
  const shape = records.status === 'ready' ? commonShape(records.records) : null

  const canRun = selectedId !== null || (upload.hea !== null && upload.dat !== null)
  const buttonLabel = running
    ? selectedId !== null ? `Screening record ${selectedId}…` : 'Screening the uploaded record…'
    : selectedId !== null ? `Screen record ${selectedId}`
    : upload.hea && upload.dat ? 'Screen the uploaded record'
    : upload.hea ? 'Add the .dat file to screen the upload'
    : upload.dat ? 'Add the .hea file to screen the upload'
    : 'Select a record or upload a .hea and .dat'

  return (
    <div className="mx-auto max-w-2xl px-3 py-5">
      <title>Screen an ECG · CardioScreen</title>
      <h1 className="text-2xl font-semibold tracking-tight">Screen a 12-lead ECG</h1>

      <section aria-label="Upload a record" className="mt-3">
        <UploadZone pair={upload} disabled={running} message={uploadMessage} onFiles={takeFiles}
                    onClear={clearFiles} />
      </section>

      <section aria-labelledby="test-records-heading" className="mt-4">
        <div className="flex items-center gap-2">
          <span aria-hidden className="h-px flex-1 bg-line" />
          <h2 id="test-records-heading" className="text-sm font-medium text-ink-muted">
            or select a held-out test record
          </h2>
          <span aria-hidden className="h-px flex-1 bg-line" />
        </div>
        <p className="mt-2 mb-2 text-sm">
          These records are from the held-out test set
          {shape && `, each ${shape.duration_s} s at ${shape.sampling_rate} Hz`}. The model never saw them during
          training.
        </p>
        <RecordPicker state={records} query={query} selectedId={selectedId} disabled={running}
                      showShape={records.status === 'ready' && shape === null}
                      onQuery={setQuery} onSelect={selectRecord} onRetry={retryRecords} />
      </section>

      {/* scroll-mb keeps the button clear of the sticky footer when it is scrolled into view */}
      <div ref={actions} className="scroll-mb-8">
        {run.status === 'error' && (
          <div className="mt-3">
            <ApiErrorPanel title={run.title} error={run.error} />
          </div>
        )}

        <button
          type="button"
          onClick={screen}
          disabled={!canRun || running}
          aria-busy={running}
          className={`mt-3 h-6 w-full rounded-md px-3 font-medium focus-visible:outline-2
                      focus-visible:outline-offset-2 focus-visible:outline-accent ${canRun
                        ? 'bg-accent text-surface enabled:hover:bg-accent-strong aria-busy:cursor-wait'
                        : 'cursor-not-allowed bg-line text-ink-muted'}`}
        >
          {buttonLabel}
        </button>
      </div>
    </div>
  )
}
