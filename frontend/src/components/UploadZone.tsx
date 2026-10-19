import { useState, type DragEvent } from 'react'
import { formatBytes } from '../lib/format'
import type { UploadPair } from '../lib/uploads'

interface Props {
  pair: UploadPair
  disabled: boolean
  message: string | null // why the last drop was not taken
  onFiles: (files: File[]) => void
  onClear: () => void
}

/** Drag-and-drop (or browse) for a WFDB record's .hea header and .dat samples. */
export function UploadZone({ pair, disabled, message, onFiles, onClear }: Props) {
  const [dragging, setDragging] = useState(false)

  function onDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault()
    setDragging(false)
    if (!disabled) onFiles(Array.from(e.dataTransfer.files))
  }

  return (
    <div>
      <label
        onDragEnter={(e) => { e.preventDefault(); if (!disabled) setDragging(true) }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false)
        }}
        onDrop={onDrop}
        data-dragging={dragging}
        data-disabled={disabled}
        className="flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed border-line-strong
                   bg-surface px-3 py-4 text-center
                   has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2
                   has-[:focus-visible]:outline-accent
                   data-[dragging=true]:border-accent data-[dragging=true]:bg-accent/5
                   data-[disabled=true]:cursor-not-allowed data-[disabled=true]:opacity-60"
      >
        <input
          type="file"
          multiple
          accept=".hea,.dat"
          disabled={disabled}
          className="sr-only"
          onChange={(e) => {
            onFiles(Array.from(e.currentTarget.files ?? []))
            e.currentTarget.value = '' // so choosing the same file again still counts
          }}
        />
        <span className="font-medium">Drop a record&rsquo;s .hea and .dat files here</span>
        <span className="text-sm text-ink-muted">
          or <span className="text-accent underline underline-offset-2">browse your files</span>.
          A WFDB record: the header and its samples.
        </span>
      </label>

      {(pair.hea || pair.dat) && (
        <div className="mt-2 flex items-start justify-between gap-2 text-sm">
          <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5">
            <FileRow label="Header (.hea)" file={pair.hea} />
            <FileRow label="Samples (.dat)" file={pair.dat} />
          </dl>
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            className="rounded-md px-1 text-ink-muted underline underline-offset-2 hover:text-ink
                       focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed"
          >
            Remove files
          </button>
        </div>
      )}

      {message && (
        <p role="alert" className="mt-2 text-sm">
          {message}
        </p>
      )}
    </div>
  )
}

function FileRow({ label, file }: { label: string; file: File | null }) {
  return (
    <>
      <dt className="text-ink-muted">{label}</dt>
      <dd>
        {file ? (
          <>
            <span className="font-medium break-all">{file.name}</span>
            <span className="text-ink-muted"> · {formatBytes(file.size)}</span>
          </>
        ) : (
          <span className="text-ink-muted">Not added yet</span>
        )}
      </dd>
    </>
  )
}
