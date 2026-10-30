import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getAnalyticsSummary,
  getAnalyticsTimeline,
  getCameraAnalytics,
  getFaceSizeQuality,
  getPOI,
  getSimilarityHistogram,
  getTopPersons,
} from '../utils/api'

/**
 * Bucket width for a range.
 *
 * Hourly columns stay readable for a day; past that they crowd into slivers,
 * and the weekday/hour heatmap already carries the time-of-day story, so
 * longer ranges switch to days.
 */
export function bucketFor(range) {
  return range === '24h' ? 'hour' : 'day'
}

/**
 * Load every analytics report for one time range.
 *
 * Every report is fetched together so the page never renders a mix of
 * ranges. While a new range loads the previous data is held and flagged
 * `stale`, which lets the page dim it instead of collapsing to a skeleton and
 * jumping the layout.
 */
export function useAnalytics(range) {
  const [data, setData] = useState(null)
  const [stale, setStale] = useState(false)
  const [error, setError] = useState(null)
  // Guards against a slow response for an abandoned range overwriting a newer one.
  const activeRange = useRef(range)

  const load = useCallback(async (target) => {
    activeRange.current = target
    setStale(true)
    setError(null)
    try {
      const [summary, timeline, histogram, faceSize, cameras, persons, people] = await Promise.all([
        getAnalyticsSummary(target),
        getAnalyticsTimeline(target, bucketFor(target)),
        getSimilarityHistogram(target),
        getFaceSizeQuality(target),
        getCameraAnalytics(target),
        getTopPersons(target),
        // Enrolment records rather than an analytics report: the compliance
        // panel describes who is on file, not what was detected.
        getPOI(),
      ])
      if (activeRange.current !== target) {
        return
      }
      setData({ summary, timeline, histogram, faceSize, cameras, persons, people })
    } catch (err) {
      if (activeRange.current === target) {
        setError(err)
      }
    } finally {
      if (activeRange.current === target) {
        setStale(false)
      }
    }
  }, [])

  useEffect(() => {
    load(range)
  }, [load, range])

  return { data, stale, error, refresh: () => load(range) }
}
