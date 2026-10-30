import { useEffect, useState } from 'react'
import { getResources } from '../utils/api'
import { isLiveConnected, subscribeLive } from '../utils/liveSocket'

const MAX_HISTORY = 60

const EMPTY = {
  cpu_pct: 0,
  ram_pct: 0,
  ram_used_gb: 0,
  ram_total_gb: 0,
  disk_pct: 0,
  gpu_pct: 0,
  vram_pct: 0,
  vram_used_mb: 0,
  vram_total_mb: 0,
  net_sent_mbps: 0,
  net_recv_mbps: 0,
}

function timestamp() {
  return new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function useResources() {
  const [current, setCurrent] = useState(EMPTY)
  const [history, setHistory] = useState([])

  useEffect(() => {
    let cancelled = false
    let pending = false

    const record = (data) => {
      if (cancelled) return
      setCurrent(data)
      setHistory((prev) => {
        const next = [...prev, { ...data, ts: timestamp() }]
        return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next
      })
    }

    // Samples arrive once a second on the shared WebSocket. The REST poll is
    // only a fallback for the first paint and for when the socket is down.
    const unsubscribe = subscribeLive((message) => {
      if (message.type === 'stats' && message.resources) record(message.resources)
    })

    const fetchResources = async () => {
      if (pending || isLiveConnected()) return
      pending = true
      try {
        record(await getResources())
      } catch {
        // Keep the last known sample when the backend is unavailable.
      } finally {
        pending = false
      }
    }

    fetchResources()
    const id = setInterval(fetchResources, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
      unsubscribe()
    }
  }, [])

  return { current, history }
}
