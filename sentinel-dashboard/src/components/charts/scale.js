/**
 * Build a count axis with clean tick values.
 *
 * Rounding the maximum alone is not enough: a bound like 50 divided into four
 * intervals yields 12.5 and 37.5, and an axis carries the values that were not
 * directly labelled, so its ticks have to be readable numbers. This picks a
 * round step first and lets the bound follow from it.
 */
export function countScale(rawMax, maxIntervals = 5) {
  const target = Math.max(rawMax, 1)
  const magnitude = 10 ** Math.floor(Math.log10(target / maxIntervals || 1))

  for (const multiple of [1, 2, 5, 10, 20, 50, 100]) {
    const step = multiple * magnitude
    const intervals = Math.ceil(target / step)
    if (intervals <= maxIntervals) {
      // Whole counts only: a fractional tick on a tally of detections is noise.
      const rounded = Math.max(Math.round(step), 1)
      const max = rounded * Math.max(intervals, 1)
      return {
        max,
        ticks: Array.from({ length: Math.max(intervals, 1) + 1 }, (_, index) => max - index * rounded),
      }
    }
  }

  return { max: target, ticks: [target, 0] }
}
