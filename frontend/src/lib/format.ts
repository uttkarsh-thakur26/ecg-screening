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
