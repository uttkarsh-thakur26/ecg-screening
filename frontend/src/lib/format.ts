const count = new Intl.NumberFormat('en-GB')
const kilobytes = new Intl.NumberFormat('en-GB', { minimumFractionDigits: 1, maximumFractionDigits: 1 })

/** 2158 -> "2,158" */
export function formatCount(n: number): string {
  return count.format(n)
}

/** A file size: "512 B" below 1 kB, else kilobytes (1 kB = 1,024 bytes) with one decimal, "23.4 kB". */
export function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${count.format(bytes)} B` : `${kilobytes.format(bytes / 1024)} kB`
}

const PROBABILITY_DECIMALS = 3
const MAX_DECIMALS = 17 // enough to write any two different doubles in [0, 1] differently

function fixed(p: number, decimals: number): string {
  return new Intl.NumberFormat('en-GB', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(p)
}

/** A probability or a threshold with three decimals, 0.0727 -> "0.073". Three, so that a threshold as low as the
 *  high-sensitivity ones (below 0.1) still reads with two significant figures. */
export function formatProbability(p: number): string {
  return fixed(p, PROBABILITY_DECIMALS)
}

/**
 * A probability and its threshold, written with the same number of decimals: three, or more when three would show
 * two different values as the same number. The flag is the API's, decided on the unrounded values; without this, a
 * probability just below its threshold could read "0.100" beside "threshold 0.100" with "Not flagged".
 */
export function formatAgainstThreshold(probability: number, threshold: number): [string, string] {
  let decimals = PROBABILITY_DECIMALS
  while (probability !== threshold && decimals < MAX_DECIMALS &&
         fixed(probability, decimals) === fixed(threshold, decimals)) decimals += 1
  return [fixed(probability, decimals), fixed(threshold, decimals)]
}
