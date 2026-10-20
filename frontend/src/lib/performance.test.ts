// The Performance screen's wording, on the real reports/metrics.json (the one-time test evaluation), imported as text
// and checked with the same schema the client applies to GET /api/metrics. The expected strings were read off that
// file by hand; a test that edits the report does so only to reach a branch the real file does not take.
import { describe, expect, it } from 'vitest'
import METRICS_JSON from '../../../reports/metrics.json?raw'
import README from '../../../README.md?raw'
import { schemas, type MetricsReport } from './api'
import {
  ciPercent, evaluationLine, foldsLabel, formatDifference, formatInterval, formatMetric, formatPercent, formatSpread,
  intervalNote,
  limitations, perClass, seedLine, weakestClass, weakestClassLine,
} from './performance'

const report: MetricsReport = schemas.MetricsReport.parse(JSON.parse(METRICS_JSON))

describe('the schema of GET /api/metrics', () => {
  it('reads the real reports/metrics.json', () => {
    expect(report.split).toBe('test')
    expect(report.classes).toEqual(['NORM', 'MI', 'STTC', 'CD', 'HYP'])
    expect(report.seed_stability.split).toBe('val')
  })

  it('refuses a report whose per-class tables do not cover its classes', () => {
    const raw = JSON.parse(METRICS_JSON)
    delete raw.ranking.per_class.HYP
    const parsed = schemas.MetricsReport.safeParse(raw)
    expect(parsed.success).toBe(false)
    expect(parsed.error?.message).toContain('ranking.per_class covers NORM, MI, STTC, CD; expected')
  })
})

describe('number formatting', () => {
  it('writes metrics with three decimals and a true minus sign', () => {
    expect(formatMetric(report.ranking.macro.auroc.value)).toBe('0.926')
    expect(formatMetric(report.calibration_error.macro.calibrated_minus_uncalibrated.value)).toBe('−0.075')
  })

  it('writes intervals with a dash, or "to" when a bound is negative', () => {
    expect(formatInterval(report.ranking.macro.auroc)).toBe('0.918–0.933')
    expect(formatInterval(perClass(report.calibration_error.per_class, 'NORM').calibrated_minus_uncalibrated))
      .toBe('−0.029 to −0.002')
  })

  it('writes a difference with its sign', () => {
    expect(formatDifference(report.baselines.cnn_minus_feature_gbdt.ranking.macro.auroc.value)).toBe('+0.086')
    expect(formatDifference(-0.0123)).toBe('−0.012')
  })

  it('writes a prevalence as a percentage with one decimal', () => {
    expect(formatPercent(perClass(report.ranking.per_class, 'NORM').prevalence)).toBe('44.6%')
  })

  it('writes a mean and standard deviation with four decimals', () => {
    expect(formatSpread(report.seed_stability.macro_auroc)).toBe('0.9279 ± 0.0009')
  })

  it('states the interval level and method from the bootstrap settings', () => {
    expect(ciPercent(report)).toBe('95%')
    expect(intervalNote(report)).toBe('95% intervals: percentile bootstrap over patients, 1,000 resamples.')
  })
})

describe('evaluationLine', () => {
  it('says the test set was evaluated once when the access log has one opening and no rerun', () => {
    expect(evaluationLine(report)).toBe('Held-out test set (fold 10), evaluated once.')
  })

  it('does not claim a single evaluation when the test fold was opened more than once', () => {
    const reopened = { ...report, fold10_access_log: [...report.fold10_access_log, ...report.fold10_access_log] }
    expect(evaluationLine(reopened)).toContain('opened 2 times')
  })

  it('names one or several folds', () => {
    expect(foldsLabel([10])).toBe('fold 10')
    expect(foldsLabel([8, 9, 10])).toBe('folds 8, 9 and 10')
  })
})

describe('limitations', () => {
  it('finds the weakest class from the ranking', () => {
    expect(weakestClass(report)).toEqual({ cls: 'HYP', lowestAuroc: true })
  })

  it("counts the weakest class's false alarms at the high-sensitivity point", () => {
    // HYP at the high-sensitivity point: tp 233, fp 545, so 545 of 778 flags are false alarms (0.70).
    expect(weakestClassLine(report)).toBe(
      'Hypertrophy is the weakest class (lowest test AUPRC and AUROC): at the high-sensitivity point about 7 in 10 ' +
      'HYP flags are false alarms (545 of 778 flags on the test set).')
  })

  it('has six plain-language points, naming the dataset from the report', () => {
    const lines = limitations(report)
    expect(lines).toHaveLength(6)
    expect(lines[0]).toBe('Single dataset (PTB-XL 1.0.3) from one source; results may not hold for other hospitals ' +
                          'or devices.')
    expect(lines[5]).toBe('Screening aid, not a diagnostic device.')
  })

  it('are the same list, word for word, as the Limitations section of README.md', () => {
    const readme = README.replaceAll('\r\n', '\n')
    for (const line of limitations(report)) expect(readme).toContain(`\n- ${line}\n`)
  })
})

describe('seedLine', () => {
  it('is the validation mean and standard deviation over the seeds in the report', () => {
    expect(seedLine(report)).toBe(
      'Validation macro AUROC 0.9279 ± 0.0009 (mean ± standard deviation) over 3 training seeds ' +
      '(42, 43, 44).')
  })
})
