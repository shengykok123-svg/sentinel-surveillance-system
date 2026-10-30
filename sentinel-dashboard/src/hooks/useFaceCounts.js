import { useEffect, useState } from 'react'
import { getCameras } from '../utils/api'

export function useFaceCounts() {
  const [counts, setCounts] = useState({})

  useEffect(() => {
    let cancelled = false

    const fetchCounts = async () => {
      try {
        const cameras = await getCameras()
        if (cancelled) return
        setCounts(
          cameras.reduce((acc, camera) => {
            acc[camera.id] = camera.face_count ?? 0
            return acc
          }, {}),
        )
      } catch {
        // Keep the most recent successful counts.
      }
    }

    fetchCounts()
    const id = setInterval(fetchCounts, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return counts
}
