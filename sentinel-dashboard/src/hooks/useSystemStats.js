import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { clearEvents as clearEventsRequest, getActiveTracks, getCameras, getEmbeddingStats, getEvents, getPOI } from '../utils/api'
import { isLiveConnected, subscribeLive } from '../utils/liveSocket'
import { mergeLivePeople, planRefresh, recordSightings } from '../utils/livePeople'

// Cameras, active tracks and events are pushed over the shared WebSocket, so
// while it is connected the 10 s tick requests nothing except people and
// embedding stats once a minute (plus a one-off catch-up of events, people and
// stats when the socket comes up). POI live status and last sighting are
// rebuilt from the pushed tracks and events in between. With the socket down
// the tick polls everything. `refresh` (after edits) always fetches in full.
const REFRESH_MS = 10000

export function useSystemStats() {
  const inFlight = useRef(null)
  const queued = useRef(null)
  const wasConnected = useRef(false)
  const lastLoad = useRef({ people: null, embeddings: null })
  const [cameras, setCameras] = useState([])
  const [events, setEvents] = useState([])
  const [basePeople, setBasePeople] = useState([])
  const [sightings, setSightings] = useState(() => new Map())
  const [activeTracks, setActiveTracks] = useState([])
  const [embeddingStats, setEmbeddingStats] = useState({
    total_vectors: 0,
    total_persons: 0,
    embedding_dim: 512,
    index_size_mb: 0,
    last_updated: null,
  })

  // Both updates land in the same render (React 18 batching); the sightings
  // Map keeps its identity when no track is newer.
  const applyTracks = useCallback((tracks) => {
    setActiveTracks(tracks)
    setSightings((prev) => recordSightings(prev, { activeTracks: tracks }))
  }, [])

  const load = useCallback(async (force) => {
    try {
      const connected = isLiveConnected()
      const plan = planRefresh({
        connected,
        wasConnected: wasConnected.current,
        now: Date.now(),
        lastLoad: lastLoad.current,
        force,
      })
      const skip = () => Promise.reject(new Error('not requested'))
      const results = await Promise.allSettled([
        plan.cameras ? getCameras() : skip(),
        plan.events ? getEvents({ limit: 50 }) : skip(),
        plan.people ? getPOI() : skip(),
        plan.tracks ? getActiveTracks() : skip(),
        plan.embeddings ? getEmbeddingStats() : skip(),
      ])

      const [cameraData, eventData, poiData, trackData, embeddingData] = results
      if (cameraData.status === 'fulfilled') setCameras(cameraData.value)
      if (eventData.status === 'fulfilled') setEvents(eventData.value.events ?? [])
      if (poiData.status === 'fulfilled') {
        setBasePeople(poiData.value)
        lastLoad.current.people = Date.now()
      }
      if (trackData.status === 'fulfilled') applyTracks(trackData.value)
      if (embeddingData.status === 'fulfilled') {
        setEmbeddingStats(embeddingData.value)
        lastLoad.current.embeddings = Date.now()
      }
      // A failed catch-up is retried on the next tick.
      wasConnected.current = connected && (!plan.events || eventData.status === 'fulfilled')
    } catch {
      // Keep the latest successful snapshot when the backend is unavailable.
    }
  }, [applyTracks])

  // Background ticks are skipped while a fetch is running. A forced refresh
  // queues one more full fetch after it, and resolves when that one is done.
  const run = useCallback((force) => {
    if (inFlight.current) {
      if (!force) return inFlight.current
      if (!queued.current) {
        queued.current = inFlight.current.then(() => {
          queued.current = null
          return run(true)
        })
      }
      return queued.current
    }
    const current = load(force).finally(() => {
      inFlight.current = null
    })
    inFlight.current = current
    return current
  }, [load])

  const refresh = useCallback(() => run(true), [run])

  useEffect(() => {
    const tick = () => run(false)
    tick()
    const id = setInterval(tick, REFRESH_MS)
    return () => clearInterval(id)
  }, [run])

  useEffect(() => subscribeLive((message) => {
    if (message.type === 'stats') {
      if (Array.isArray(message.cameras)) setCameras(message.cameras)
      if (Array.isArray(message.active_tracks)) applyTracks(message.active_tracks)
      return
    }
    setEvents((prev) => [message, ...prev].slice(0, 50))
    setSightings((prev) => recordSightings(prev, { event: message }))
  }), [applyTracks])

  const clearEvents = useCallback(async () => {
    await clearEventsRequest()
    setEvents([])
    // Cleared events no longer back anyone's last sighting.
    setSightings(new Map())
    refresh()
  }, [refresh])

  const people = useMemo(
    () => mergeLivePeople(basePeople, activeTracks, sightings),
    [basePeople, activeTracks, sightings],
  )

  const metrics = useMemo(() => {
    const camerasOnline = cameras.filter((camera) => camera.online).length
    const facesDetected = cameras.reduce((sum, camera) => sum + (camera.face_count ?? 0), 0)
    const avgFps = cameras.length
      ? Math.round(cameras.reduce((sum, camera) => sum + (camera.fps ?? 0), 0) / cameras.length)
      : 0
    const alertingPoiIds = new Set(
      activeTracks
        .filter((track) => track.status === 'recognized' && track.person_type === 'watchlist')
        .map((track) => track.person_id)
        .filter(Boolean),
    )

    return {
      camerasOnline,
      camerasTotal: cameras.length,
      facesDetected,
      avgFps,
      alerts: alertingPoiIds.size,
      totalVectors: embeddingStats.total_vectors,
    }
  }, [activeTracks, cameras, embeddingStats.total_vectors])

  return {
    metrics,
    events,
    cameras,
    people,
    activeTracks,
    embeddingStats,
    refresh,
    clearEvents,
  }
}
