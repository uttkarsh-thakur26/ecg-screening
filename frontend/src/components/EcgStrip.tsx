import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Signal } from '../lib/api'
import {
  DEFAULT_LEAD, GRID_STEPS, MARGIN, MIN_MINOR_SPACING_PX, SCALE, attentionSpan, leadLabel, minorSpacing, sampleTime,
  stripLayout, timeGrid, voltageGrid, voltageWindow, xOf, yOf, type StripLayout,
} from '../lib/ecgPlot'

/** Opacity of the attention shade where the map is 1, its peak; 0 where the map is 0. Soft, so the grid and the trace
 *  stay readable through it. The caption's swatch uses the same opacity (bg-accent/30). */
const ATTENTION_PEAK_ALPHA = 0.3
const TRACE_WIDTH_PX = 1.25
const LABEL_SIZE_PX = 11
const LEAD_NAME_SIZE_PX = 12

interface Props {
  signal: Signal               // checked with stripProblem (src/lib/ecgPlot.ts) before it gets here
  attention: readonly number[] // one value per sample, in [0, 1]
  attentionClass: string
  attentionRawPeak: number     // 0: Grad-CAM found no time that raised the class score, so the map is all zeros
  showAttention: boolean       // false when no abnormal category is flagged: no band, and a line saying so
}

/**
 * One lead of the signal on a standard ECG grid (time in seconds, voltage in mV), with the model's attention map
 * shaded behind the trace on the same time axis when a category is flagged. Lead II first; any of the signal's leads
 * can be chosen. Leads are named in their clinical spelling (aVR, not the API's AVR).
 */
export function EcgStrip({ signal, attention, attentionClass, attentionRawPeak, showAttention }: Props) {
  const [lead, setLead] = useState(DEFAULT_LEAD)
  const [width, setWidth] = useState(0)
  const [fontLoaded, setFontLoaded] = useState(false)
  const frame = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)

  const trace = leadSamples(signal, lead)
  const range = useMemo(() => voltageWindow(trace), [trace])
  const layout = useMemo(
    () => (width > MARGIN.left + MARGIN.right ? stripLayout(width, trace.length, signal.sampling_rate, range) : null),
    [width, trace.length, signal.sampling_rate, range],
  )

  // The canvas fills the card's width; its height follows from the width and the lead's voltage window.
  useLayoutEffect(() => {
    const el = frame.current
    if (el === null) return
    setWidth(Math.floor(el.getBoundingClientRect().width))
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width
      if (w !== undefined) setWidth(Math.floor(w))
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // The axis labels are drawn in the page's font; draw again once it has loaded.
  useEffect(() => {
    let live = true
    document.fonts.load(labelFont(LABEL_SIZE_PX, 400)).then(() => { if (live) setFontLoaded(true) })
    return () => { live = false }
  }, [])

  useEffect(() => {
    if (canvas.current !== null && layout !== null) {
      drawStrip(canvas.current, layout, trace, showAttention ? attention : null, leadLabel(lead))
    }
  }, [layout, trace, attention, showAttention, lead, fontLoaded])

  const seconds = trace.length / signal.sampling_rate
  return (
    <div>
      <div role="group" aria-label="Lead shown" className="flex flex-wrap items-center gap-0.5">
        <span aria-hidden className="mr-1 text-sm font-medium text-ink-muted">Lead</span>
        {signal.leads.map((name) => (
          <button
            key={name}
            type="button"
            aria-pressed={name === lead}
            onClick={() => setLead(name)}
            className="h-4 min-w-5 rounded-md border border-line-strong bg-surface px-1 text-sm font-medium
                       hover:border-ink focus-visible:outline-2 focus-visible:outline-offset-2
                       focus-visible:outline-accent aria-pressed:border-accent aria-pressed:bg-accent
                       aria-pressed:text-surface"
          >
            {leadLabel(name)}
          </button>
        ))}
      </div>

      <div ref={frame} className="mt-2" style={{ height: layout?.height }}>
        <canvas
          ref={canvas}
          role="img"
          aria-label={`ECG lead ${leadLabel(lead)}: ${seconds} seconds at ${signal.sampling_rate} Hz, in millivolts` +
                      (showAttention ? `, with the model's attention for ${attentionClass} shaded behind the trace`
                                     : '')}
          className="block w-full"
          style={{ height: layout?.height }}
        />
      </div>

      <div className="mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-0.5 text-sm">
        {showAttention ? (
          <p className="flex items-center gap-1">
            <span aria-hidden className="inline-block h-1.5 w-3 shrink-0 rounded-sm bg-accent/30" />
            Model attention for {attentionClass}, not a clinical rationale
          </p>
        ) : (
          <p>No abnormal category flagged; attention not shown.</p>
        )}
        <p className="text-ink-muted">
          Grid {GRID_STEPS.minorSeconds} s × {GRID_STEPS.minorMv} mV, bold every {GRID_STEPS.majorSeconds} s
          and {GRID_STEPS.majorMv} mV · {SCALE.paperSpeedMmPerSecond} mm/s, {SCALE.gainMmPerMv} mm/mV
        </p>
      </div>
      {showAttention && attentionRawPeak === 0 && (
        <p className="mt-0.5 text-sm text-ink-muted">
          Grad-CAM found no time in this record that raised the {attentionClass} score, so nothing is shaded.
        </p>
      )}
    </div>
  )
}

function leadSamples(signal: Signal, lead: string): number[] {
  const samples = signal.data[signal.leads.indexOf(lead)]
  if (samples === undefined) throw new Error(`The signal has no lead ${lead}; its leads are ${signal.leads.join(', ')}`)
  return samples
}

/** A colour token from src/index.css, so the canvas uses the same palette as the rest of the page. */
function themeColour(name: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--color-${name}`).trim()
  if (value === '') throw new Error(`The colour token --color-${name} is not defined (src/index.css)`)
  return value
}

function labelFont(size: number, weight: 400 | 500 | 600): string {
  const family = getComputedStyle(document.documentElement).getPropertyValue('--font-sans').trim()
  if (family === '') throw new Error('The font token --font-sans is not defined (src/index.css)')
  return `${weight} ${size}px ${family}`
}

/** "−1" with a true minus sign, as in the rest of the typography. */
function axisNumber(value: number): string {
  return value < 0 ? `−${Math.abs(value)}` : `${value}`
}

/** attention: null draws no band. lead: the name drawn in the corner, already in its clinical spelling. */
function drawStrip(canvas: HTMLCanvasElement, layout: StripLayout, trace: readonly number[],
                   attention: readonly number[] | null, lead: string) {
  const ctx = canvas.getContext('2d')
  if (ctx === null) throw new Error('This browser cannot draw the ECG: the canvas has no 2D context')
  const dpr = window.devicePixelRatio
  canvas.width = Math.round(layout.width * dpr)
  canvas.height = Math.round(layout.height * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const { plot, samples, samplingRate } = layout
  const right = plot.left + plot.width
  const bottom = plot.top + plot.height
  // 1-device-pixel lines on whole device pixels, so the grid is sharp at any zoom.
  const crisp = (css: number) => (Math.round(css * dpr) + 0.5) / dpr
  const snap = (css: number) => Math.round(css * dpr) / dpr

  ctx.clearRect(0, 0, layout.width, layout.height)
  ctx.fillStyle = themeColour('surface')
  ctx.fillRect(plot.left, plot.top, plot.width, plot.height)

  // 1. The attention map, behind everything else. Value i shades the time of sample i (attentionSpan).
  ctx.fillStyle = themeColour('accent')
  for (const [i, value] of (attention ?? []).entries()) {
    if (value <= 0) continue
    const [start, end] = attentionSpan(i, samples, samplingRate)
    const x0 = snap(xOf(layout, start))
    const x1 = snap(xOf(layout, end))
    if (x1 <= x0) continue
    ctx.globalAlpha = value * ATTENTION_PEAK_ALPHA
    ctx.fillRect(x0, plot.top, x1 - x0, plot.height)
  }
  ctx.globalAlpha = 1

  // 2. The grid: minor lines (if they are far enough apart to read), then major lines over them.
  const times = timeGrid(samples, samplingRate)
  const volts = voltageGrid(layout.window)
  ctx.lineWidth = 1 / dpr
  for (const major of [false, true]) {
    if (!major && minorSpacing(layout) < MIN_MINOR_SPACING_PX) continue
    ctx.strokeStyle = themeColour(major ? 'grid-major' : 'grid-minor')
    ctx.beginPath()
    for (const line of times) {
      if (line.major !== major) continue
      const x = crisp(xOf(layout, line.value))
      ctx.moveTo(x, plot.top)
      ctx.lineTo(x, bottom)
    }
    for (const line of volts) {
      if (line.major !== major) continue
      const y = crisp(yOf(layout, line.value))
      ctx.moveTo(plot.left, y)
      ctx.lineTo(right, y)
    }
    ctx.stroke()
  }

  // 3. The trace: sample i at time i / sampling rate, its value in mV. Clipped to the plot, though the voltage window
  //    already holds every sample of the lead.
  ctx.save()
  ctx.beginPath()
  ctx.rect(plot.left, plot.top, plot.width, plot.height)
  ctx.clip()
  ctx.strokeStyle = themeColour('ink')
  ctx.lineWidth = TRACE_WIDTH_PX
  ctx.lineJoin = 'round'
  ctx.beginPath()
  for (const [i, mv] of trace.entries()) {
    const x = xOf(layout, sampleTime(i, samplingRate))
    const y = yOf(layout, mv)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
  ctx.restore()

  // 4. The lead's name in the top-left corner, as on a printed ECG.
  ctx.font = labelFont(LEAD_NAME_SIZE_PX, 600)
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  const nameWidth = ctx.measureText(lead).width
  ctx.fillStyle = themeColour('surface')
  ctx.fillRect(plot.left + 4, plot.top + 4, nameWidth + 8, LEAD_NAME_SIZE_PX + 6)
  ctx.fillStyle = themeColour('ink')
  ctx.fillText(lead, plot.left + 8, plot.top + 7)

  // 5. Axes: whole seconds along the bottom, whole millivolts up the left, and their units.
  ctx.font = labelFont(LABEL_SIZE_PX, 400)
  ctx.fillStyle = themeColour('ink-muted')
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  for (const line of times) {
    if (line.label) ctx.fillText(axisNumber(line.value), xOf(layout, line.value), bottom + 6)
  }
  ctx.fillText('Time (s)', plot.left + plot.width / 2, bottom + 6 + LABEL_SIZE_PX + 8)
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  for (const line of volts) {
    if (line.label) ctx.fillText(axisNumber(line.value), plot.left - 6, yOf(layout, line.value))
  }
  ctx.save()
  ctx.translate(MARGIN.left / 4, plot.top + plot.height / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.fillText('mV', 0, 0)
  ctx.restore()

  // The plot's frame, over the shade's edges.
  ctx.strokeStyle = themeColour('grid-major')
  ctx.lineWidth = 1 / dpr
  ctx.strokeRect(crisp(plot.left), crisp(plot.top), snap(plot.width), snap(plot.height))
}
