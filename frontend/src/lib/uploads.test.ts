import { describe, expect, it } from 'vitest'
import { EMPTY_UPLOAD, addFiles } from './uploads'

const file = (name: string) => new File([], name)

describe('addFiles', () => {
  it('puts a .hea and a .dat dropped together in their slots, whatever the case of the extension', () => {
    const hea = file('00009_lr.HEA')
    const dat = file('00009_lr.dat')
    expect(addFiles(EMPTY_UPLOAD, [dat, hea])).toEqual({ ok: true, pair: { hea, dat } })
  })

  it('takes the files one at a time and replaces a file of the same kind', () => {
    const first = addFiles(EMPTY_UPLOAD, [file('a.hea')])
    if (!first.ok) throw new Error(first.message)
    expect(first.pair.dat).toBeNull()
    const dat = file('a.dat')
    const second = addFiles(first.pair, [dat])
    if (!second.ok) throw new Error(second.message)
    expect(second.pair).toEqual({ hea: first.pair.hea, dat })
    const hea = file('b.hea')
    expect(addFiles(second.pair, [hea])).toEqual({ ok: true, pair: { hea, dat } })
  })

  it('refuses the whole drop if any file is not a .hea or .dat, and says which', () => {
    const result = addFiles(EMPTY_UPLOAD, [file('a.hea'), file('a.atr'), file('notes.txt')])
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.message).toContain('a.atr, notes.txt are not')
  })

  it('refuses two files of the same kind in one drop', () => {
    const result = addFiles(EMPTY_UPLOAD, [file('a.dat'), file('b.dat')])
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.message).toBe('Add one .dat file, not 2 (a.dat, b.dat).')
  })

  it('refuses an empty drop', () => {
    expect(addFiles(EMPTY_UPLOAD, []).ok).toBe(false)
  })
})
