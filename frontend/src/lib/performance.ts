// Wording and number formatting for the Performance screen. Every number comes from reports/metrics.json as
// GET /api/metrics serves it; nothing here computes a metric. The only arithmetic is rounding, counting flags
// (tp + fp) and turning a false-alarm fraction into "about N in 10".

import type { Estimate, MetricsReport } from './api'
import { formatCount } from './format'

/** Metrics (AUROC, sensitivity, ECE, ...) are shown with three decimals, as in the evaluation's own figures. */
const METRIC_DECIMALS = 3
/** The seed-to-seed spread is about a thousandth, so it needs one more decimal to show as more than 0.001. */
const SPREAD_DECIMALS = 4

function formatter(decimals: number, signDisplay: 'negative' | 'exceptZero') {
  return new Intl.NumberFormat('en-GB', { minimumFractionDigits: decimals, maximumFractionDigits: decimals,
                                          signDisplay })
}
const metric = formatter(METRIC_DECIMALS, 'negative')
const difference = formatter(METRIC_DECIMALS, 'exceptZero')
const spread = formatter(SPREAD_DECIMALS, 'negative')

/** A true minus sign, as in the rest of the typography. */
const minus = (s: string) => s.replace('-', '−')

/** 0.92586 -> "0.926" */
export function formatMetric(x: number): string {
  return minus(metric.format(x))
}

/** A difference between two metrics, with its sign: 0.08645 -> "+0.086". */
export function formatDifference(x: number): string {
  return minus(difference.format(x))
}

/** An estimate's interval, "0.918–0.933"; "−0.029 to −0.002" when a bound is negative, so the dash is not read as a
 *  minus sign. */
export function formatInterval(e: Pick<Estimate, 'ci_low' | 'ci_high'>): string {
  const [low, high] = [formatMetric(e.ci_low), formatMetric(e.ci_high)]
  return e.ci_low < 0 || e.ci_high < 0 ? `${low} to ${high}` : `${low}–${high}`
}

const percent = new Intl.NumberFormat('en-GB', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 })

/** A proportion as a percentage with one decimal: 0.44625 -> "44.6%". */
export function formatPercent(x: number): string {
  return percent.format(x)
}

/** A mean and standard deviation: "0.9279 ± 0.0009". */
export function formatSpread(s: { mean: number; std: number }): string {
  return `${minus(spread.format(s.mean))} ± ${minus(spread.format(s.std))}`
}

/** bootstrap.ci_level 0.95 -> "95%". */
export function ciPercent(m: Pick<MetricsReport, 'bootstrap'>): string {
  return `${Math.round(m.bootstrap.ci_level * 100)}%`
}

/** How the intervals were made, from the report's bootstrap settings. */
export function intervalNote(m: Pick<MetricsReport, 'bootstrap'>): string {
  const { unit, resamples } = m.bootstrap
  return `${ciPercent(m)} intervals: percentile bootstrap over ${unit}s, ${formatCount(resamples)} resamples.`
}

/** [10] -> "fold 10", [9, 10] -> "folds 9 and 10". */
export function foldsLabel(folds: readonly number[]): string {
  if (folds.length === 1) return `fold ${folds[0]}`
  return `folds ${folds.slice(0, -1).join(', ')} and ${folds[folds.length - 1]}`
}

/** "Held-out test set (fold 10), evaluated once.", when the fold-10 access log has exactly one opening and no rerun;
 *  otherwise the number of openings, so the screen never claims a single evaluation the report does not back. */
export function evaluationLine(m: Pick<MetricsReport, 'folds' | 'fold10_access_log' | 'reruns'>): string {
  const opened = m.fold10_access_log.length
  const where = `Held-out test set (${foldsLabel(m.folds)})`
  if (opened === 1 && m.reruns.length === 0) return `${where}, evaluated once.`
  return `${where}, opened ${opened} times; metrics.json records each opening and the reason for each rerun.`
}

/** The row of a per-class table; the schema has checked that every class has one, so a missing row is a bug. */
export function perClass<T>(table: Readonly<Record<string, T>>, cls: string): T {
  const row = table[cls]
  if (row === undefined) throw new Error(`metrics.json has no row for class ${cls}`)
  return row
}

/** A class's plain-English name from the report. */
export function classLabel(m: Pick<MetricsReport, 'class_descriptions'>, cls: string): string {
  return perClass(m.class_descriptions, cls)
}

/** The class with the lowest test AUPRC, and whether its AUROC is also the lowest. */
export function weakestClass(m: Pick<MetricsReport, 'classes' | 'ranking'>): { cls: string; lowestAuroc: boolean } {
  const by = (key: 'auroc' | 'auprc') => (c: string) => perClass(m.ranking.per_class, c)[key].value
  const lowest = (score: (c: string) => number) =>
    m.classes.reduce((worst, c) => (score(c) < score(worst) ? c : worst))
  const cls = lowest(by('auprc'))
  return { cls, lowestAuroc: lowest(by('auroc')) === cls }
}

/** "Hypertrophy is the weakest class (...): at the high-sensitivity point about 7 in 10 HYP flags are false alarms
 *  (545 of 778 flags on the test set)." The class, the counts and the fraction all come from the report. */
export function weakestClassLine(m: MetricsReport): string {
  const { cls, lowestAuroc } = weakestClass(m)
  const basis = `lowest test AUPRC${lowestAuroc ? ' and AUROC' : ''}`
  const head = `${classLabel(m, cls)} is the weakest class (${basis})`
  const point = m.operating_points.high_sensitivity.per_class[cls]
  if (point === undefined) return `${head}.` // a class with no high-sensitivity threshold has no flags to count
  const flags = point.tp + point.fp
  if (flags === 0) return `${head}; at the high-sensitivity point it flagged no test record.`
  const inTen = Math.round((10 * point.fp) / flags)
  return `${head}: at the high-sensitivity point about ${inTen} in 10 ${cls} flags are false alarms ` +
         `(${formatCount(point.fp)} of ${formatCount(flags)} flags on the test set).`
}

/** The Limitations block, in plain language. Its only numbers are the dataset version and the weakest class's flag
 *  counts, all read from the report. */
export function limitations(m: MetricsReport): string[] {
  return [
    `Single dataset (${m.dataset.name} ${m.dataset.version}) from one source; results may not hold for other ` +
      'hospitals or devices.',
    'No external or prospective validation.',
    weakestClassLine(m),
    `Many missed cases have uncertain labels (low likelihood in ${m.dataset.name}), most clearly for myocardial ` +
      'infarction. This comes from the error analysis on the validation split, not from the test set.',
    'Attention maps show where the model responded, not a clinical reason; they are shared across leads and coarse ' +
      'in time, so they cannot point to a single beat or wave.',
    'Screening aid, not a diagnostic device.',
  ]
}

/** "Validation macro AUROC 0.9279 ± 0.0009 (mean ± standard deviation) over 3 training seeds (42, 43, 44)." */
export function seedLine(m: Pick<MetricsReport, 'seed_stability'>): string {
  const s = m.seed_stability
  return `Validation macro AUROC ${formatSpread(s.macro_auroc)} (mean ± standard deviation) over ` +
         `${s.runs.length} training seeds (${s.runs.map((r) => r.seed).join(', ')}).`
}
