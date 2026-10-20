import { describe, expect, it } from 'vitest'
import { formatAgainstThreshold, formatBytes, formatCount, formatProbability } from './format'

describe('format', () => {
  it('writes probabilities and thresholds with three decimals', () => {
    expect(formatProbability(0.07272719688635824)).toBe('0.073')
    expect(formatProbability(0.9492055574807482)).toBe('0.949')
    expect(formatProbability(0)).toBe('0.000')
    expect(formatProbability(1)).toBe('1.000')
  })

  it('writes a probability and its threshold with enough decimals to tell them apart', () => {
    expect(formatAgainstThreshold(0.566, 0.1812876491245789)).toEqual(['0.566', '0.181'])
    expect(formatAgainstThreshold(0.0996, 0.0994)).toEqual(['0.100', '0.099'])   // already apart at three
    expect(formatAgainstThreshold(0.09996, 0.09994)).toEqual(['0.1000', '0.0999']) // both "0.100" at three
    expect(formatAgainstThreshold(0.09996, 0.10004)).toEqual(['0.09996', '0.10004'])
    expect(formatAgainstThreshold(0.1, 0.1)).toEqual(['0.100', '0.100'])
  })

  it('writes counts with separators and sizes in kilobytes', () => {
    expect(formatCount(2158)).toBe('2,158')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(24000)).toBe('23.4 kB')
  })
})
