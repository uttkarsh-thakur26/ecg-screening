import type { ClassPrediction } from '../lib/api'
import { formatAgainstThreshold, formatProbability } from '../lib/format'
import { agreement, type Agreement } from '../lib/results'

interface Props {
  predictions: readonly ClassPrediction[] // the API's rows, in its class order
  present: ReadonlyMap<string, boolean> | null // the record's labels by class, once revealed
}

const AGREEMENT_TEXT: Record<Agreement, string> = {
  'agrees': 'flag matches',
  'missed': 'missed: not flagged',
  'false-flag': 'flagged, not in labels',
}

/**
 * One row per class, as the API sent it: the code and plain name, the calibrated probability as a number and a bar,
 * and for an abnormal class the API's flag and the threshold it was decided against. The normal class has a
 * probability only. Once the labels are revealed, each row also shows whether the class is in them.
 */
export function ClassResults({ predictions, present }: Props) {
  return (
    <table className="w-full border-collapse text-left">
      <caption className="sr-only">
        Calibrated probability of each class; abnormal classes are flagged when the probability reaches the threshold
      </caption>
      <thead>
        <tr className="border-b border-line text-sm text-ink-muted">
          <th scope="col" className="py-1 pr-2 font-medium">Class</th>
          <th scope="col" className="py-1 pr-2 text-right font-medium">Probability</th>
          <th scope="col" className="w-[30%] py-1 pr-3 font-medium">
            <span className="sr-only">Probability bar</span>
          </th>
          <th scope="col" className="py-1 pr-2 font-medium">Flag</th>
          {present && <th scope="col" className="py-1 font-medium">Ground truth</th>}
        </tr>
      </thead>
      <tbody>
        {predictions.map((c) => (
          <tr key={c.class} className="border-b border-line last:border-b-0">
            <th scope="row" className="py-1 pr-2 font-normal">
              <span className="inline-block w-7 font-semibold">{c.class}</span>{' '}
              {c.label}
            </th>
            <td className="py-1 pr-2 text-right font-medium">
              {c.threshold === null ? formatProbability(c.probability)
                                    : formatAgainstThreshold(c.probability, c.threshold)[0]}
            </td>
            <td className="py-1 pr-3">
              <ProbabilityBar probability={c.probability} threshold={c.threshold} flagged={c.flagged} />
            </td>
            <td className="py-1 pr-2">
              {c.flagged === null ? (
                <span className="sr-only">No flag: this class has no threshold</span>
              ) : (
                <>
                  <span className="flex items-center gap-1 text-sm">
                    <span
                      aria-hidden
                      className={`size-1 shrink-0 rounded-full ${c.flagged ? 'bg-flagged' : 'border border-slate'}`}
                    />
                    <span className={c.flagged ? 'font-semibold' : 'text-slate'}>
                      {c.flagged ? 'Flagged' : 'Not flagged'}
                    </span>
                  </span>
                  <span className="block text-xs text-ink-muted">
                    threshold {formatAgainstThreshold(c.probability, c.threshold)[1]}
                  </span>
                </>
              )}
            </td>
            {present && <TruthCell present={present.get(c.class)} flagged={c.flagged} />}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ProbabilityBar({ probability, threshold, flagged }:
                        { probability: number; threshold: number | null; flagged: boolean | null }) {
  return (
    <div aria-hidden className="relative h-0.75 rounded-full bg-line">
      <div
        className={`absolute inset-y-0 left-0 rounded-full ${flagged === true ? 'bg-flagged' : 'bg-slate'}`}
        style={{ width: `${probability * 100}%` }}
      />
      {threshold !== null && (
        // The threshold, so the bar shows how far the probability is from it.
        <div className="absolute -inset-y-0.5 w-0.25 -translate-x-1/2 bg-ink" style={{ left: `${threshold * 100}%` }} />
      )}
    </div>
  )
}

function TruthCell({ present, flagged }: { present: boolean | undefined; flagged: boolean | null }) {
  // truthProblem (src/lib/results.ts) has checked that the labels cover every predicted class.
  if (present === undefined) throw new Error('The revealed labels do not cover this class')
  const verdict = agreement(flagged, present)
  return (
    <td className="py-1 text-sm">
      <span className={present ? 'font-semibold' : ''}>{present ? 'In labels' : 'Not in labels'}</span>
      {verdict !== null && (
        <span className={`block text-xs ${verdict === 'agrees' ? 'text-ink-muted' : 'font-semibold text-ink'}`}>
          {AGREEMENT_TEXT[verdict]}
        </span>
      )}
    </td>
  )
}
