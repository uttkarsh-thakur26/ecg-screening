// Wording for the Result screen, from the API's responses as they are. Nothing here computes a number.

import type { Prediction, Truth } from './api'

/** "2 of 4 abnormal categories flagged at the high-sensitivity operating point.": the API's own summary, then the
 *  operating point the API used, by its name in thresholds.json ("high_sensitivity", read with a hyphen). */
export function summaryLine(prediction: Pick<Prediction, 'summary' | 'operating_point'>): string {
  return `${prediction.summary} at the ${prediction.operating_point.replaceAll('_', '-')} operating point.`
}

/** Why these labels cannot be shown beside this prediction, or null when they can: they must be the labels of the
 *  same record, for the same classes in the same order, and the list of labels must agree with the per-class rows. */
export function truthProblem(recordId: string, predictions: readonly { class: string }[], truth: Truth): string | null {
  if (truth.id !== recordId) return `The API returned the labels of record ${truth.id}, not of record ${recordId}`
  const predicted = predictions.map((c) => c.class).join(', ')
  const labelled = truth.classes.map((c) => c.class).join(', ')
  if (predicted !== labelled) return `The labels cover the classes ${labelled}; the prediction covers ${predicted}`
  // The same set of classes, in whatever order the list comes.
  const present = truth.classes.filter((c) => c.present).map((c) => c.class)
  if ([...present].sort().join(', ') !== [...truth.labels].sort().join(', ')) {
    const listed = truth.labels.join(', ') || '(none)'
    return `The API lists the labels ${listed} but marks ${present.join(', ') || '(none)'} present`
  }
  return null
}

/** Whether each class is in the record's labels, by class code. */
export function presentByClass(truth: Truth): Map<string, boolean> {
  return new Map(truth.classes.map((c) => [c.class, c.present]))
}

/**
 * An abnormal class's flag against the label: 'agrees' when the flag and the label say the same, 'missed' when the
 * class is in the labels but not flagged, 'false-flag' when it is flagged but not in the labels. The normal class
 * has no flag (null), so there is nothing to compare.
 */
export type Agreement = 'agrees' | 'missed' | 'false-flag'

export function agreement(flagged: boolean | null, present: boolean): Agreement | null {
  if (flagged === null) return null
  if (flagged === present) return 'agrees'
  return present ? 'missed' : 'false-flag'
}
