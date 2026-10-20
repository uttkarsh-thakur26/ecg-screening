// WCAG AA contrast of the colours used for text, read from the design tokens in src/index.css, on the page
// background (canvas, #FAFAFA) and on cards (surface, white). AA asks 4.5:1 for normal-size text.
import { describe, expect, it } from 'vitest'
import css from '../index.css?raw'

const AA_NORMAL_TEXT = 4.5
const TEXT_TOKENS = ['ink', 'ink-muted', 'slate', 'accent'] as const
const BACKGROUNDS = ['canvas', 'surface'] as const

function token(name: string): string {
  const match = new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6})\\s*;`).exec(css)
  if (match?.[1] === undefined) throw new Error(`src/index.css has no 6-digit hex --color-${name}`)
  return match[1]
}

/** Relative luminance (WCAG 2.x). */
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }) as [number, number, number]
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
  return (hi + 0.05) / (lo + 0.05)
}

// Every source file, to check where the amber is used.
const sources = import.meta.glob<string>(['../**/*.tsx'], { query: '?raw', import: 'default', eager: true })

describe('text contrast (WCAG AA)', () => {
  for (const fg of TEXT_TOKENS) {
    for (const bg of BACKGROUNDS) {
      it(`${fg} text on ${bg} is at least ${AA_NORMAL_TEXT}:1`, () => {
        expect(contrast(token(fg), token(bg))).toBeGreaterThanOrEqual(AA_NORMAL_TEXT)
      })
    }
  }

  it('matches the known ratios of the grey text: #525252 is 7.5:1 on #FAFAFA', () => {
    expect(contrast('#525252', '#fafafa')).toBeCloseTo(7.49, 2)
    expect(contrast('#ffffff', '#000000')).toBeCloseTo(21, 9)
  })

  it('never colours text amber, which is below AA on white', () => {
    expect(contrast(token('flagged'), token('surface'))).toBeLessThan(AA_NORMAL_TEXT)
    expect(Object.keys(sources).length).toBeGreaterThan(0)
    for (const [file, source] of Object.entries(sources)) {
      expect(source, file).not.toMatch(/\btext-flagged\b/)
    }
  })
})
