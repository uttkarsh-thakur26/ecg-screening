// The Result screen's wording. The objects here hold only the fields under test (a class code, a flag, a label); no
// probability or other model output is made up.
import { describe, expect, it } from 'vitest'
import type { Truth } from './api'
import { agreement, presentByClass, summaryLine, truthProblem } from './results'

const classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP'].map((c) => ({ class: c }))

function truth(id: string, present: string[], labels = present): Truth {
  return { id, labels, classes: classes.map((c) => ({ class: c.class, label: `${c.class} name`, present:
                                                      present.includes(c.class) })) }
}

describe('summaryLine', () => {
  it("is the API's summary at the API's operating point", () => {
    expect(summaryLine({ summary: '2 of 4 abnormal categories flagged', operating_point: 'high_sensitivity' }))
      .toBe('2 of 4 abnormal categories flagged at the high-sensitivity operating point.')
  })
})

describe('truthProblem', () => {
  it('is null for the same record and classes, with the label list matching the rows', () => {
    expect(truthProblem('9', classes, truth('9', ['MI', 'CD']))).toBeNull()
    expect(truthProblem('9', classes, truth('9', []))).toBeNull()
  })

  it('refuses the labels of another record', () => {
    expect(truthProblem('9', classes, truth('38', ['NORM'])))
      .toBe('The API returned the labels of record 38, not of record 9')
  })

  it('refuses labels for other classes or in another order', () => {
    expect(truthProblem('9', classes.slice(0, 4), truth('9', ['NORM'])))
      .toBe('The labels cover the classes NORM, MI, STTC, CD, HYP; the prediction covers NORM, MI, STTC, CD')
    expect(truthProblem('9', [...classes].reverse(), truth('9', ['NORM']))).toContain('the prediction covers HYP, CD')
  })

  it('accepts the label list in any order', () => {
    expect(truthProblem('9', classes, truth('9', ['MI', 'STTC'], ['STTC', 'MI']))).toBeNull()
  })

  it('refuses a label list that disagrees with the per-class rows', () => {
    expect(truthProblem('9', classes, truth('9', ['MI'], ['MI', 'CD'])))
      .toBe('The API lists the labels MI, CD but marks MI present')
    expect(truthProblem('9', classes, truth('9', ['MI'], [])))
      .toBe('The API lists the labels (none) but marks MI present')
  })
})

describe('presentByClass', () => {
  it('maps every class to whether it is in the labels', () => {
    expect(Object.fromEntries(presentByClass(truth('9', ['STTC'])))).toEqual(
      { NORM: false, MI: false, STTC: true, CD: false, HYP: false })
  })
})

describe('agreement', () => {
  it('compares an abnormal class flag with its label', () => {
    expect(agreement(true, true)).toBe('agrees')
    expect(agreement(false, false)).toBe('agrees')
    expect(agreement(false, true)).toBe('missed')
    expect(agreement(true, false)).toBe('false-flag')
  })

  it('has nothing to compare for the normal class, which has no flag', () => {
    expect(agreement(null, true)).toBeNull()
    expect(agreement(null, false)).toBeNull()
  })
})
