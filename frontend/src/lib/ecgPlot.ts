// Geometry of the ECG strip (src/components/EcgStrip.tsx): where each value from the API goes on the canvas. Pure
// functions, so the drawing can be checked without a browser. Nothing here changes a value; it only places it.
//
// Time: sample i of a lead is at i / sampling_rate seconds. The attention map has one value per sample (Grad-CAM,
// upsampled by the API to the signal's length), so value i is drawn at the same time as sample i.

import type { Signal } from './api'

/** The standard rhythm lead (PROJECT_SPEC.md section 7), shown first. */
export const DEFAULT_LEAD = 'II'

/** The 12 leads in their clinical spelling. PTB-XL's files, and so the API, write the augmented limb leads in
 *  capitals (AVR, AVL, AVF); the screen shows them as clinicians write them. */
const CLINICAL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] as const

/** A lead's name as the API sends it -> its clinical spelling: "AVR" -> "aVR", "II" -> "II". Throws for a name that
 *  is not one of the 12 standard leads (stripProblem checks the signal's leads first). */
export function leadLabel(name: string): string {
  const label = CLINICAL_LEADS.find((l) => l.toUpperCase() === name.toUpperCase())
  if (label === undefined) throw new Error(`${JSON.stringify(name)} is not one of the 12 standard ECG leads`)
  return label
}

/**
 * The strip's scale, in ECG-machine terms: paper speed 25 mm/s and gain 20 mm/mV ("double standard", a setting ECG
 * machines offer beside the 10 mm/mV standard). The 10-second strip fills the screen's width, which makes it about
 * paper size; at 10 mm/mV a typical lead II complex would then be under 20 px tall, too small to read on a laptop
 * or a projector. The gain is the same for every record and every lead, so amplitudes stay comparable, and the
 * legend under the strip states it. At this gain a 0.04 s x 0.1 mV grid box is twice as tall as it is wide.
 */
export const SCALE = { paperSpeedMmPerSecond: 25, gainMmPerMv: 20 } as const

/** The clinical grid: a minor line every 0.04 s and every 0.1 mV, a major line every fifth minor line (0.2 s,
 *  0.5 mV). Counted in whole minor steps so that no floating-point drift moves a line. */
const MINOR_PER_SECOND = 25
const MINOR_PER_MV = 10
const MINOR_PER_MAJOR = 5
const MAJOR_PER_MV = MINOR_PER_MV / MINOR_PER_MAJOR

/** The grid's steps, for the legend under the strip. */
export const GRID_STEPS = {
  minorSeconds: 1 / MINOR_PER_SECOND,
  minorMv: 1 / MINOR_PER_MV,
  majorSeconds: MINOR_PER_MAJOR / MINOR_PER_SECOND,
  majorMv: MINOR_PER_MAJOR / MINOR_PER_MV,
} as const

/** The voltage window is at least this tall, so a low-amplitude lead is not drawn as a thin band. A taller lead gets
 *  a taller window: the scale itself never changes. */
const MIN_WINDOW_MV = 3

/** Below this spacing the minor lines would merge into a grey wash, so only the major lines are drawn. */
export const MIN_MINOR_SPACING_PX = 3

/** Room around the plot for the axis labels and titles, in CSS pixels. */
export const MARGIN = { left: 44, right: 8, top: 8, bottom: 40 } as const

export interface VoltageWindow {
  low: number  // mV, on a major line
  high: number // mV, on a major line
}

export interface StripLayout {
  width: number         // CSS pixels: the whole canvas
  height: number
  plot: { left: number; top: number; width: number; height: number }
  samples: number
  samplingRate: number  // Hz
  seconds: number       // samples / samplingRate: the x axis runs from 0 to here
  window: VoltageWindow
  pxPerSecond: number
  pxPerMv: number
}

export interface GridLine {
  value: number   // seconds or mV
  major: boolean
  label: boolean  // whole seconds and whole millivolts are labelled
}

/** Why this signal and attention map cannot be drawn as one strip, or null when they can. */
export function stripProblem(signal: Signal, attention: readonly number[]): string | null {
  if (!(signal.sampling_rate > 0)) return `The signal's sampling rate is ${signal.sampling_rate} Hz`
  const samples = signal.data[0]?.length ?? 0
  if (samples === 0) return 'The signal has no samples'
  for (const [i, lead] of signal.data.entries()) {
    if (lead.length !== samples) {
      return `Lead ${signal.leads[i]} has ${lead.length} samples and lead ${signal.leads[0]} has ${samples}`
    }
  }
  if (attention.length !== samples) {
    return `The attention map has ${attention.length} values for ${samples} samples per lead, so it cannot be ` +
           "placed on the signal's time axis"
  }
  if (!signal.leads.includes(DEFAULT_LEAD)) {
    return `The signal has no lead ${DEFAULT_LEAD}; its leads are ${signal.leads.join(', ')}`
  }
  const unknown = signal.leads.filter((name) => !CLINICAL_LEADS.some((l) => l.toUpperCase() === name.toUpperCase()))
  if (unknown.length > 0) return `The signal has leads that are not standard ECG leads: ${unknown.join(', ')}`
  return null
}

/**
 * The voltage window for one lead: from the major line below its lowest sample to the major line above its highest
 * (strictly beyond, so the trace never touches the frame), then grown one major line at a time, on the side with less
 * room, until it is at least MIN_WINDOW_MV tall.
 */
export function voltageWindow(samples: readonly number[]): VoltageWindow {
  if (samples.length === 0) throw new Error('voltageWindow needs at least one sample')
  let min = Infinity
  let max = -Infinity
  for (const v of samples) {
    if (v < min) min = v
    if (v > max) max = v
  }
  // In major steps (0.5 mV). Multiplying by 2 is exact in floating point.
  let low = Math.ceil(min * MAJOR_PER_MV) - 1
  let high = Math.floor(max * MAJOR_PER_MV) + 1
  while (high - low < MIN_WINDOW_MV * MAJOR_PER_MV) {
    if (min - low / MAJOR_PER_MV <= high / MAJOR_PER_MV - max) low -= 1
    else high += 1
  }
  return { low: low / MAJOR_PER_MV, high: high / MAJOR_PER_MV }
}

/** The strip's size and scale for a canvas cssWidth pixels wide: the plot fills the width, the height follows. */
export function stripLayout(cssWidth: number, samples: number, samplingRate: number,
                            window: VoltageWindow): StripLayout {
  const plotWidth = cssWidth - MARGIN.left - MARGIN.right
  if (!(plotWidth > 0)) throw new Error(`The strip needs more than ${MARGIN.left + MARGIN.right} px, got ${cssWidth}`)
  const seconds = samples / samplingRate
  const pxPerSecond = plotWidth / seconds
  const pxPerMv = (pxPerSecond * SCALE.gainMmPerMv) / SCALE.paperSpeedMmPerSecond
  const plotHeight = (window.high - window.low) * pxPerMv
  return {
    width: cssWidth,
    height: Math.ceil(MARGIN.top + plotHeight + MARGIN.bottom), // whole pixels, so the bitmap is not stretched
    plot: { left: MARGIN.left, top: MARGIN.top, width: plotWidth, height: plotHeight },
    samples, samplingRate, seconds, window, pxPerSecond, pxPerMv,
  }
}

/** Horizontal position of a time, in CSS pixels. */
export function xOf(layout: StripLayout, seconds: number): number {
  return layout.plot.left + seconds * layout.pxPerSecond
}

/** Vertical position of a voltage, in CSS pixels (up is positive). */
export function yOf(layout: StripLayout, mv: number): number {
  return layout.plot.top + (layout.window.high - mv) * layout.pxPerMv
}

/** The time of sample i. The trace and the attention shade both use this. */
export function sampleTime(i: number, samplingRate: number): number {
  return i / samplingRate
}

/**
 * The time span attention value i shades: centred on sample i, half a sample to either side, within the strip. The
 * strip ends a whole sample after the last sample (samples / rate), so the last value also shades that final half
 * sample, and the spans tile the strip with no gap.
 */
export function attentionSpan(i: number, samples: number, samplingRate: number): [number, number] {
  const start = Math.max(0, (i - 0.5) / samplingRate)
  const end = i === samples - 1 ? samples / samplingRate : (i + 0.5) / samplingRate
  return [start, end]
}

/** Vertical grid lines, every 0.04 s from 0 to the end of the strip. */
export function timeGrid(samples: number, samplingRate: number): GridLine[] {
  const last = Math.floor((samples * MINOR_PER_SECOND) / samplingRate)
  return Array.from({ length: last + 1 }, (_, k) => ({
    value: k / MINOR_PER_SECOND,
    major: k % MINOR_PER_MAJOR === 0,
    label: k % MINOR_PER_SECOND === 0,
  }))
}

/** Horizontal grid lines, every 0.1 mV across the window. */
export function voltageGrid(window: VoltageWindow): GridLine[] {
  const first = Math.round(window.low * MINOR_PER_MV)
  const last = Math.round(window.high * MINOR_PER_MV)
  return Array.from({ length: last - first + 1 }, (_, i) => {
    const k = first + i
    return { value: k / MINOR_PER_MV, major: k % MINOR_PER_MAJOR === 0, label: k % MINOR_PER_MV === 0 }
  })
}

/** Pixels between minor lines on the axis where they are closest together. */
export function minorSpacing(layout: StripLayout): number {
  return Math.min(layout.pxPerSecond / MINOR_PER_SECOND, layout.pxPerMv / MINOR_PER_MV)
}
