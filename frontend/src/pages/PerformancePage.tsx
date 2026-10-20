import { useEffect, useState, type ReactNode } from 'react'
import { ApiErrorPanel } from '../components/ApiErrorPanel'
import { ReportFigure } from '../components/ReportFigure'
import { ApiError, getMetrics, type ClassPoint, type Estimate, type MacroPoint, type MetricsReport } from '../lib/api'
import { formatCount, formatProbability } from '../lib/format'
import {
  classLabel, evaluationLine, formatDifference, formatInterval, formatMetric, formatPercent, intervalNote, limitations,
  perClass, seedLine,
} from '../lib/performance'

type State =
  | { status: 'loading' }
  | { status: 'ready'; report: MetricsReport }
  | { status: 'missing'; message: string } // the API answered 404: the one-time evaluation has not been run
  | { status: 'error'; error: unknown }

/**
 * Screen 3: the one-time evaluation of the release model on the held-out test set, as GET /api/metrics serves
 * reports/metrics.json. Every number is read from that report; the screen rounds them and computes none of its own.
 * Without the report there is nothing to show, and the screen says so instead of showing any number.
 */
export function PerformancePage() {
  const [state, setState] = useState<State>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    getMetrics(controller.signal).then(
      (report) => { if (!controller.signal.aborted) setState({ status: 'ready', report }) },
      (error: unknown) => {
        if (controller.signal.aborted) return
        if (error instanceof ApiError && error.kind === 'http' && error.status === 404) {
          setState({ status: 'missing', message: error.message })
        } else {
          setState({ status: 'error', error })
        }
      },
    )
    return () => controller.abort()
  }, [attempt])

  return (
    <div className="mx-auto max-w-5xl px-3 py-3">
      <title>Model performance · CardioScreen</title>
      <h1 className="text-2xl font-semibold tracking-tight">Model performance</h1>
      {state.status === 'loading' && (
        <p role="status" className="mt-2 text-ink-muted">Loading the test-set evaluation…</p>
      )}
      {state.status === 'error' && (
        <div className="mt-2">
          <ApiErrorPanel
            title="Could not load the test-set evaluation"
            error={state.error}
            onRetry={() => {
              setState({ status: 'loading' })
              setAttempt((n) => n + 1)
            }}
          />
        </div>
      )}
      {state.status === 'missing' && <NoReport message={state.message} />}
      {state.status === 'ready' && <Report report={state.report} />}
    </div>
  )
}

function NoReport({ message }: { message: string }) {
  return (
    <div role="status" className="mt-2 rounded-lg border border-line bg-surface p-3">
      <p className="text-lg font-medium">No test-set evaluation to show</p>
      <p className="mt-1">{message}.</p>
      <p className="mt-1 text-sm text-ink-muted">
        This screen shows reports/metrics.json only, the file the one-time evaluation writes. Until that file exists
        it shows no metrics at all.
      </p>
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ the report

function Report({ report: m }: { report: MetricsReport }) {
  return (
    <div className="animate-fade-in motion-reduce:animate-none">
      <Header m={m} />
      <RankingSection m={m} />
      <OperatingPointsSection m={m} />
      <BaselineSection m={m} />
      <CalibrationSection m={m} />
      <Section title="Test-set figures">
        <div className="grid gap-2">
          <ReportFigure name="roc" caption="ROC curves per class, with the two frozen operating points marked."
                        alt="Receiver operating characteristic curve for each class on the test set" />
          <ReportFigure name="pr" caption="Precision–recall curves per class, with the two frozen operating points marked."
                        alt="Precision-recall curve for each class on the test set" />
          <ReportFigure name="confusion_max_f1" caption="Confusion matrices at the max-F1 thresholds, every class."
                        alt="Confusion matrix of each class at its max-F1 threshold on the test set" />
          <ReportFigure name="confusion_high_sensitivity"
                        caption="Confusion matrices at the high-sensitivity thresholds, abnormal classes."
                        alt="Confusion matrix of each abnormal class at its high-sensitivity threshold on the test set" />
        </div>
      </Section>
      <SeedSection m={m} />
      <LimitationsSection m={m} />
    </div>
  )
}

function Header({ m }: { m: MetricsReport }) {
  return (
    <header className="mt-1">
      <p className="text-lg font-medium">{evaluationLine(m)}</p>
      <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-2 rounded-lg border border-line bg-surface p-2
                     sm:grid-cols-3 lg:grid-cols-6">
        <Fact term="Dataset">{m.dataset.name} {m.dataset.version}</Fact>
        <Fact term="Test records">{formatCount(m.records)}</Fact>
        <Fact term="Test patients">{formatCount(m.patients)}</Fact>
        <Fact term="Model version">
          {m.model.version} <span className="text-sm font-normal text-ink-muted">({m.model.tag})</span>
        </Fact>
        <Fact term="Git commit">
          <span title={m.git.commit}>{m.git.commit.slice(0, 7)}</span>
          {m.git.dirty && <span className="block text-sm font-normal">with uncommitted changes</span>}
        </Fact>
        <Fact term="Evaluated">{m.date_evaluated}</Fact>
      </dl>
      <p className="mt-1 text-sm text-ink-muted">{m.description}</p>
      <p className="mt-0.5 text-sm text-ink-muted">{intervalNote(m)}</p>
    </header>
  )
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div>
      <dt className="text-sm text-ink-muted">{term}</dt>
      <dd className="font-medium">{children}</dd>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-4">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <div className="mt-1">{children}</div>
    </section>
  )
}

/** A note under a section heading: what the numbers are and how they were made. */
function Note({ children }: { children: ReactNode }) {
  return <p className="mb-1 max-w-3xl text-sm text-ink-muted">{children}</p>
}

// ------------------------------------------------------------------------------------------------ tables

function Table({ caption, head, children }: { caption: string; head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface px-2">
      <table className="w-full border-collapse text-left">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-line text-sm text-ink-muted">
            {head.map((h, i) => (
              <th key={h} scope="col" className={`py-1 font-medium ${i === 0 ? 'pr-2' : 'pl-2 text-right'}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

/** A table row; a macro row is set off from the per-class rows above it. */
function Row({ head, macro = false, children }: { head: ReactNode; macro?: boolean; children: ReactNode }) {
  return (
    <tr className={`border-b border-line last:border-b-0 ${macro ? 'border-t border-t-line-strong' : ''}`}>
      <th scope="row" className="py-1 pr-2 align-top font-normal">{head}</th>
      {children}
    </tr>
  )
}

function ClassHead({ m, cls }: { m: MetricsReport; cls: string }) {
  return (
    <>
      <span className="inline-block w-7 font-semibold">{cls}</span> {classLabel(m, cls)}
    </>
  )
}

function MacroHead({ over }: { over: string }) {
  return (
    <>
      <span className="font-semibold">Macro</span> <span className="text-ink-muted">{over}</span>
    </>
  )
}

/** A value with its interval underneath. */
function EstimateCell({ e, format = formatMetric }: { e: Estimate; format?: (x: number) => string }) {
  return (
    <td className="py-1 pl-2 text-right align-top">
      <span className="block font-medium">{format(e.value)}</span>
      <span className="block text-sm text-ink-muted">
        <span className="sr-only">interval </span>{formatInterval(e)}
      </span>
    </td>
  )
}

function PlainCell({ children }: { children?: ReactNode }) {
  return <td className="py-1 pl-2 text-right align-top">{children}</td>
}

const allClasses = (m: MetricsReport) => `all ${m.classes.length} classes`
const abnormalClasses = (m: MetricsReport) => `${m.abnormal_classes.length} abnormal classes`

function RankingSection({ m }: { m: MetricsReport }) {
  const { per_class, macro, macro_abnormal } = m.ranking
  return (
    <Section title="Ranking">
      <Note>
        Threshold-free: how well the calibrated probability ranks records with the class above records without it.
        Value on all test records, interval underneath.
      </Note>
      <Table caption="AUROC and AUPRC per class and macro-averaged, with intervals"
             head={['Class', 'Positive records', 'AUROC', 'AUPRC']}>
        {m.classes.map((cls) => {
          const r = perClass(per_class, cls)
          return (
            <Row key={cls} head={<ClassHead m={m} cls={cls} />}>
              <PlainCell>
                <span className="block font-medium">{formatCount(r.n_positive)}</span>
                <span className="block text-sm text-ink-muted">{formatPercent(r.prevalence)}</span>
              </PlainCell>
              <EstimateCell e={r.auroc} />
              <EstimateCell e={r.auprc} />
            </Row>
          )
        })}
        <Row macro head={<MacroHead over={allClasses(m)} />}>
          <PlainCell />
          <EstimateCell e={macro.auroc} />
          <EstimateCell e={macro.auprc} />
        </Row>
        <Row macro head={<MacroHead over={abnormalClasses(m)} />}>
          <PlainCell />
          <EstimateCell e={macro_abnormal.auroc} />
          <EstimateCell e={macro_abnormal.auprc} />
        </Row>
      </Table>
    </Section>
  )
}

const POINT_HEAD = ['Class', 'Threshold', 'Sensitivity', 'Specificity', 'Precision', 'F1']

function PointCells({ p }: { p: MacroPoint }) {
  return (
    <>
      <EstimateCell e={p.sensitivity} />
      <EstimateCell e={p.specificity} />
      <EstimateCell e={p.precision} />
      <EstimateCell e={p.f1} />
    </>
  )
}

function ClassPointRow({ m, cls, p }: { m: MetricsReport; cls: string; p: ClassPoint }) {
  return (
    <Row head={<ClassHead m={m} cls={cls} />}>
      <PlainCell>{formatProbability(p.threshold)}</PlainCell>
      <PointCells p={p} />
    </Row>
  )
}

function OperatingPointsSection({ m }: { m: MetricsReport }) {
  const { high_sensitivity: hs, max_f1: f1 } = m.operating_points
  return (
    <Section title="Operating points">
      <Note>
        A class is flagged when its calibrated probability reaches the class&rsquo;s threshold. Both sets of thresholds
        were frozen on the validation split before the test set was opened.
      </Note>

      <h3 className="mt-2 font-semibold">High-sensitivity point, abnormal classes</h3>
      <Note>Rule: {hs.rule}.</Note>
      <Table caption="Sensitivity, specificity, precision and F1 at the high-sensitivity thresholds, with intervals"
             head={POINT_HEAD}>
        {hs.classes.map((cls) => <ClassPointRow key={cls} m={m} cls={cls} p={perClass(hs.per_class, cls)} />)}
        <Row macro head={<MacroHead over={abnormalClasses(m)} />}>
          <PlainCell />
          <PointCells p={hs.macro_abnormal} />
        </Row>
      </Table>

      <h3 className="mt-3 font-semibold">Max-F1 point, every class</h3>
      <Note>Rule: {f1.rule}.</Note>
      <Table caption="Sensitivity, specificity, precision and F1 at the max-F1 thresholds, with intervals"
             head={POINT_HEAD}>
        {f1.classes.map((cls) => <ClassPointRow key={cls} m={m} cls={cls} p={perClass(f1.per_class, cls)} />)}
        <Row macro head={<MacroHead over={allClasses(m)} />}>
          <PlainCell />
          <PointCells p={f1.macro} />
        </Row>
        <Row macro head={<MacroHead over={abnormalClasses(m)} />}>
          <PlainCell />
          <PointCells p={f1.macro_abnormal} />
        </Row>
      </Table>
    </Section>
  )
}

function BaselineSection({ m }: { m: MetricsReport }) {
  const cnn = m.ranking.macro
  const baseline = m.baselines.feature_gbdt.ranking.macro
  const lead = m.baselines.cnn_minus_feature_gbdt.ranking.macro
  return (
    <Section title="Against the feature baseline">
      <Note>
        Macro-averaged over {allClasses(m)}, on the same test records. The lead is paired: each resample scores both
        models on the same patients.
      </Note>
      <Table caption="Macro AUROC and AUPRC of the CNN and the feature baseline, and the CNN's lead, with intervals"
             head={['Macro metric', '1D CNN', 'Feature baseline', 'CNN lead']}>
        {(['auroc', 'auprc'] as const).map((key) => (
          <Row key={key} head={<span className="font-semibold">{key.toUpperCase()}</span>}>
            <EstimateCell e={cnn[key]} />
            <EstimateCell e={baseline[key]} />
            <EstimateCell e={lead[key]} format={formatDifference} />
          </Row>
        ))}
      </Table>
      <p className="mt-1 max-w-3xl text-sm text-ink-muted">Feature baseline: {m.baselines.feature_gbdt.description}.</p>
    </Section>
  )
}

const CALIBRATOR_NAME = { temperature: 'Temperature scaling', platt: 'Platt scaling' } as const

function CalibrationSection({ m }: { m: MetricsReport }) {
  const ece = m.calibration_error
  return (
    <Section title="Calibration">
      <Note>
        Expected calibration error (ECE) on the test set with {ece.bins} equal-width bins, lower is better, before and
        after the calibrator frozen on the validation split. Point values only: {ece.interval_note}
      </Note>
      <Table caption="ECE per class and macro-averaged, uncalibrated and calibrated"
             head={['Class', 'Calibrator', 'ECE uncalibrated', 'ECE calibrated']}>
        {m.classes.map((cls) => {
          const c = perClass(ece.per_class, cls)
          return (
            <Row key={cls} head={<ClassHead m={m} cls={cls} />}>
              <PlainCell>{CALIBRATOR_NAME[c.calibrator.method]}</PlainCell>
              <PlainCell><span className="font-medium">{formatMetric(c.uncalibrated.value)}</span></PlainCell>
              <PlainCell><span className="font-medium">{formatMetric(c.calibrated.value)}</span></PlainCell>
            </Row>
          )
        })}
        <Row macro head={<MacroHead over={allClasses(m)} />}>
          <PlainCell />
          <PlainCell><span className="font-medium">{formatMetric(ece.macro.uncalibrated.value)}</span></PlainCell>
          <PlainCell><span className="font-medium">{formatMetric(ece.macro.calibrated.value)}</span></PlainCell>
        </Row>
      </Table>
      <div className="mt-2">
        <ReportFigure name="reliability"
                      caption="Reliability diagrams per class on the test set, before and after calibration."
                      alt="Reliability diagram of each class on the test set, uncalibrated and calibrated" />
      </div>
    </Section>
  )
}

function SeedSection({ m }: { m: MetricsReport }) {
  return (
    <Section title="Training-seed stability (validation split)">
      <p className="font-medium">{seedLine(m)}</p>
      <p className="mt-0.5 max-w-3xl text-sm text-ink-muted">
        Validation numbers, not test-set numbers. {m.seed_stability.description}
      </p>
    </Section>
  )
}

function LimitationsSection({ m }: { m: MetricsReport }) {
  return (
    <Section title="Limitations">
      <ul className="max-w-3xl list-disc space-y-1 pl-3">
        {limitations(m).map((line) => <li key={line}>{line}</li>)}
      </ul>
    </Section>
  )
}
