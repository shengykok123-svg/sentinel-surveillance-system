import { useEffect, useState } from 'react'
import { getCameras } from '../utils/api'

export function useCameraFps() {
  const [fps, setFps] = useState({})

  useEffect(() => {
    let cancelled = false

    const fetchFps = async () => {
      try {
        const cameras = await getCameras()
        if (cancelled) return
        setFps(
          cameras.reduce((acc, camera) => {
            acc[camera.id] = camera.fps ?? 0
            return acc
          }, {}),
        )
      } catch {
        // Keep the most recent successful FPS values.
      }
    }

    fetchFps()
    const id = setInterval(fetchFps, 1000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return fps
}
