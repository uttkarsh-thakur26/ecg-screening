// The two files of a WFDB record upload, as the user drops or picks them. Only the file kinds are checked here, to
// know which form field each file goes in; whether the pair is a record the model can read is the API's decision,
// and its 400 message says why if not.

export interface UploadPair {
  hea: File | null
  dat: File | null
}

export const EMPTY_UPLOAD: UploadPair = { hea: null, dat: null }

export type AddFilesResult = { ok: true; pair: UploadPair } | { ok: false; message: string }

function kindOf(name: string): 'hea' | 'dat' | null {
  const lower = name.toLowerCase()
  return lower.endsWith('.hea') ? 'hea' : lower.endsWith('.dat') ? 'dat' : null
}

/**
 * The pair with the new files in their slots: a .hea replaces the header, a .dat replaces the samples. Files can come
 * one at a time or together. A drop with any other file, or with two files of the same kind, changes nothing and
 * says why.
 */
export function addFiles(pair: UploadPair, files: readonly File[]): AddFilesResult {
  if (files.length === 0) return { ok: false, message: 'No files were dropped.' }
  const others = files.filter((f) => kindOf(f.name) === null)
  if (others.length > 0) {
    return { ok: false, message: `Only a WFDB record's .hea and .dat files can be uploaded; ` +
                                 `${others.map((f) => f.name).join(', ')} ${others.length === 1 ? 'is' : 'are'} not.` }
  }
  const next = { ...pair }
  for (const kind of ['hea', 'dat'] as const) {
    const ofKind = files.filter((f) => kindOf(f.name) === kind)
    if (ofKind.length > 1) {
      return { ok: false, message: `Add one .${kind} file, not ${ofKind.length} ` +
                                   `(${ofKind.map((f) => f.name).join(', ')}).` }
    }
    if (ofKind[0] !== undefined) next[kind] = ofKind[0]
  }
  return { ok: true, pair: next }
}
