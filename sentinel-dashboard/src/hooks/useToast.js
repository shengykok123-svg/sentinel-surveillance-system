import { useEffect, useRef, useState } from 'react'

import { playAlertSound } from './usePreferences'

let toastId = 0

function formatNow() {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export function useToast(events, activeTracks, preferences) {
  const [toasts, setToasts] = useState([])
  const lastAlerts = useRef(new Map())
  const preferenceRef = useRef(preferences)
  preferenceRef.current = preferences
  useEffect(() => { if (!preferences.notifications) setToasts([]) }, [preferences.notifications])
  const allowAlert = (key) => {
    const p = preferenceRef.current
    const now = Date.now()
    if (now - (lastAlerts.current.get(key) ?? -Infinity) < p.cooldown * 1000) return false
    lastAlerts.current.set(key, now)
    for (const [oldKey, time] of lastAlerts.current) {
      if (now - time > 3600000) lastAlerts.current.delete(oldKey)
    }
    if (p.sound) playAlertSound().catch(() => {})
    return p.notifications
  }
  const seenEventIds = useRef(new Set())
  const initializedEvents = useRef(false)
  const prevAlertTrackKeys = useRef(new Set())

  useEffect(() => {
    const currentAlertTrackKeys = new Set(
      activeTracks
        .filter((track) => track.status === 'recognized' && track.person_id)
        .map((track) => `${track.camera_id}:${track.track_id}:${track.person_id}`),
    )

    const newAlertTracks = activeTracks.filter((track) => {
      if (track.status !== 'recognized' || !track.person_id) {
        return false
      }
      const key = `${track.camera_id}:${track.track_id}:${track.person_id}`
      return !prevAlertTrackKeys.current.has(key)
    })

    newAlertTracks.forEach((track) => {
      if (!allowAlert(`${track.camera_id}:${track.person_id}`)) return
      const id = ++toastId
      setToasts((prev) => [
        {
          id,
          severity: 'alert',
          type: 'POI recognized',
          camera: track.camera_name,
          time: formatNow(),
          track_id: track.track_id,
          person_id: track.person_id,
          person_name: track.person_name,
          person_type: track.person_type,
        },
        ...prev,
      ].slice(0, 5))
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((toast) => toast.id !== id))
      }, 5500)
    })

    prevAlertTrackKeys.current = currentAlertTrackKeys
  }, [activeTracks])

  useEffect(() => {
    if (!events.length) {
      return
    }

    if (!initializedEvents.current) {
      events.forEach((event) => {
        seenEventIds.current.add(event.id)
      })
      initializedEvents.current = true
      return
    }

    const unseenEvents = []
    for (let index = events.length - 1; index >= 0; index -= 1) {
      const event = events[index]
      if (!seenEventIds.current.has(event.id)) {
        unseenEvents.push(event)
        seenEventIds.current.add(event.id)
      }
    }

    if (seenEventIds.current.size > 200) {
      seenEventIds.current = new Set(events.slice(0, 100).map((event) => event.id))
    }

    unseenEvents
      // Unknown detections are recorded for analytics and shown in the event
      // log, but they are too frequent to interrupt the operator with. Toasts
      // stay reserved for identified subjects.
      .filter((event) => event.severity === 'warn' && event.status !== 'unknown')
      .forEach((event) => {
        if (!allowAlert(`${event.camera_id}:${event.person_id || event.type}`)) return
        const id = ++toastId
        setToasts((prev) => [{ ...event, id }, ...prev].slice(0, 5))
        window.setTimeout(() => {
          setToasts((prev) => prev.filter((toast) => toast.id !== id))
        }, 5500)
      })
  }, [events])

  const dismiss = (id) => setToasts((prev) => prev.filter((toast) => toast.id !== id))

  return { toasts, dismiss }
}
