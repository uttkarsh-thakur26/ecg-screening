// The strip's geometry. The arrays here are a few hand-written values to place on the canvas, not ECGs, and no
// prediction is involved: the question is only where a value goes.
import { describe, expect, it } from 'vitest'
import type { Signal } from './api'
import {
  GRID_STEPS, MARGIN, SCALE, attentionSpan, leadLabel, minorSpacing, sampleTime, stripLayout, stripProblem, timeGrid,
  voltageGrid, voltageWindow, xOf, yOf,
} from './ecgPlot'

function signal(leads: string[], samples: number, samplingRate = 100): Signal {
  return { leads, sampling_rate: samplingRate, units: 'mV', data: leads.map(() => new Array<number>(samples).fill(0)) }
}

describe('stripProblem', () => {
  it('is null when every lead and the attention map have one value per sample, and lead II is there', () => {
    expect(stripProblem(signal(['I', 'II'], 1000), new Array<number>(1000).fill(0))).toBeNull()
  })

  it('refuses an attention map that is not one value per sample', () => {
    expect(stripProblem(signal(['I', 'II'], 1000), new Array<number>(500).fill(0)))
      .toBe("The attention map has 500 values for 1000 samples per lead, so it cannot be placed on the signal's " +
            'time axis')
  })

  it('refuses leads of different lengths, no samples, no lead II, or no sampling rate', () => {
    const uneven = signal(['I', 'II'], 1000)
    uneven.data[1] = [0, 0]
    expect(stripProblem(uneven, [])).toBe('Lead II has 2 samples and lead I has 1000')
    expect(stripProblem(signal(['II'], 0), [])).toBe('The signal has no samples')
    expect(stripProblem(signal(['I', 'III'], 10), new Array<number>(10).fill(0)))
      .toBe('The signal has no lead II; its leads are I, III')
    expect(stripProblem(signal(['II'], 10, 0), [])).toBe("The signal's sampling rate is 0 Hz")
  })

  it('refuses a lead that is not one of the 12 standard leads', () => {
    expect(stripProblem(signal(['II', 'AVR', 'X1'], 10), new Array<number>(10).fill(0)))
      .toBe('The signal has leads that are not standard ECG leads: X1')
  })
})

describe('leadLabel', () => {
  it("writes the API's lead names in their clinical spelling", () => {
    const api = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] // as PTB-XL names them
    expect(api.map(leadLabel)).toEqual(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    expect(leadLabel('aVF')).toBe('aVF')
  })

  it('throws for a name that is not a standard lead', () => {
    expect(() => leadLabel('V7')).toThrow('"V7" is not one of the 12 standard ECG leads')
  })
})

describe('voltageWindow', () => {
  it('reaches the major line strictly beyond each extreme, then grows to 3 mV on the side with less room', () => {
    // -0.3 .. 0.73 mV: -0.5 .. 1.0 first, then a major line at a time on the tighter side, to -1.5 .. 1.5.
    expect(voltageWindow([-0.3, 0.2, 0.73])).toEqual({ low: -1.5, high: 1.5 })
  })

  it('never puts an extreme on the frame: a sample on a major line gets the next line out', () => {
    expect(voltageWindow([-3, 1.5])).toEqual({ low: -3.5, high: 2 })
  })

  it('keeps a lead taller than 3 mV at its own range, so the scale never changes', () => {
    expect(voltageWindow([-3.2, 0, 2.9])).toEqual({ low: -3.5, high: 3 })
  })

  it('holds every sample, on major lines (multiples of 0.5 mV)', () => {
    for (const lead of [[0], [0.01, -0.01], [5.37, -1.2], [-18, 0.4], [1.999, 2.001]]) {
      const { low, high } = voltageWindow(lead)
      expect(Math.min(...lead)).toBeGreaterThan(low)
      expect(Math.max(...lead)).toBeLessThan(high)
      expect(Number.isInteger(low * 2) && Number.isInteger(high * 2)).toBe(true)
      expect(high - low).toBeGreaterThanOrEqual(3)
    }
  })

  it('needs a sample', () => {
    expect(() => voltageWindow([])).toThrow('at least one sample')
  })
})

describe('stripLayout', () => {
  const layout = stripLayout(1000, 1000, 100, { low: -2, high: 2 })

  it('fills the width with the record at 25 mm/s and 20 mm/mV: 1.25 mV as tall as 1 s is wide', () => {
    expect(SCALE).toEqual({ paperSpeedMmPerSecond: 25, gainMmPerMv: 20 })
    expect(layout.seconds).toBe(10)
    expect(layout.plot.width).toBe(1000 - MARGIN.left - MARGIN.right)
    expect(layout.pxPerSecond).toBeCloseTo(layout.plot.width / 10, 9)
    expect(layout.pxPerMv * 1.25).toBeCloseTo(layout.pxPerSecond, 9)
    expect(layout.plot.height).toBeCloseTo(4 * layout.pxPerMv, 9)
    expect(layout.height).toBe(Math.ceil(MARGIN.top + layout.plot.height + MARGIN.bottom))
  })

  it('makes a minor box twice as tall as it is wide, as 20 mm/mV paper would', () => {
    expect(GRID_STEPS.minorMv * layout.pxPerMv).toBeCloseTo(2 * GRID_STEPS.minorSeconds * layout.pxPerSecond, 9)
    expect(minorSpacing(layout)).toBeCloseTo(GRID_STEPS.minorSeconds * layout.pxPerSecond, 9)
  })

  it('maps 0 s and the end of the record to the plot edges, and the window to its top and bottom', () => {
    expect(xOf(layout, 0)).toBe(layout.plot.left)
    expect(xOf(layout, 10)).toBeCloseTo(layout.plot.left + layout.plot.width, 9)
    expect(yOf(layout, 2)).toBe(layout.plot.top)
    expect(yOf(layout, -2)).toBeCloseTo(layout.plot.top + layout.plot.height, 9)
    expect(yOf(layout, 0)).toBeCloseTo(layout.plot.top + layout.plot.height / 2, 9)
  })

  it('refuses a width with no room for the plot', () => {
    expect(() => stripLayout(MARGIN.left + MARGIN.right, 1000, 100, { low: -2, high: 2 })).toThrow('needs more than')
  })
})

describe('attention and trace share the time axis', () => {
  it('centres attention value i on sample i, half a sample either side (the first and last reach the ends)', () => {
    for (const i of [1, 250, 998]) {
      const [start, end] = attentionSpan(i, 1000, 100)
      expect((start + end) / 2).toBeCloseTo(sampleTime(i, 100), 12)
      expect(end - start).toBeCloseTo(1 / 100, 12)
    }
  })

  it('tiles the strip from 0 s to its end without gaps or overlaps', () => {
    const spans = Array.from({ length: 1000 }, (_, i) => attentionSpan(i, 1000, 100))
    expect(spans[0]?.[0]).toBe(0)
    expect(spans[999]?.[1]).toBe(10)
    for (let i = 1; i < spans.length; i++) expect(spans[i]?.[0]).toBeCloseTo(spans[i - 1]?.[1] ?? NaN, 12)
  })

  it('puts sample i at i / sampling rate', () => {
    expect(sampleTime(0, 100)).toBe(0)
    expect(sampleTime(150, 100)).toBe(1.5)
    expect(sampleTime(999, 100)).toBe(9.99)
  })
})

describe('the grid', () => {
  it('has a line every 0.04 s, a major one every 0.2 s and a label every second', () => {
    const lines = timeGrid(1000, 100)
    expect(lines).toHaveLength(251)
    expect(lines[0]).toEqual({ value: 0, major: true, label: true })
    expect(lines[1]).toEqual({ value: 0.04, major: false, label: false })
    expect(lines[5]).toEqual({ value: 0.2, major: true, label: false })
    expect(lines.at(-1)).toEqual({ value: 10, major: true, label: true })
    expect(lines.filter((l) => l.major)).toHaveLength(51)
    expect(lines.filter((l) => l.label).map((l) => l.value)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  })

  it('has a line every 0.1 mV, a major one every 0.5 mV and a label every millivolt', () => {
    const lines = voltageGrid({ low: -2, high: 1.5 })
    expect(lines).toHaveLength(36)
    expect(lines[0]).toEqual({ value: -2, major: true, label: true })
    expect(lines[1]).toEqual({ value: -1.9, major: false, label: false })
    expect(lines.filter((l) => l.major).map((l) => l.value)).toEqual([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    expect(lines.filter((l) => l.label).map((l) => l.value)).toEqual([-2, -1, 0, 1])
  })
})
