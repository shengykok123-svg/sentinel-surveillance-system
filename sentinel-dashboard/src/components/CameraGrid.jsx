import { useEffect, useMemo, useRef, useState } from 'react'
import { getStreamTicket, getStreamWebSocketUrl } from '../utils/api'
import './CameraGrid.css'

// Frames arrive one per WebSocket message and are decoded into a small queue
// before being drawn on a fixed beat. The queue absorbs network and encoder
// jitter (an MJPEG <img> repaints the instant a frame lands, so every server
// hiccup was visible) at the cost of roughly three frames of latency.
const BUFFER_FRAMES = 3
const MAX_QUEUE = 8
const DEFAULT_INTERVAL_MS = 40

function useFrameStream(cameraId, enabled, canvasRef, onStatus) {
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!enabled) return undefined

    let disposed = false
    let socket = null
    let retry = null
    let raf = 0
    let decoding = 0
    let playing = false
    let starved = false
    let live = false
    let interval = DEFAULT_INTERVAL_MS
    let lastArrival = 0
    let nextDue = 0
    let lastSeq = 0
    let drawnSeq = 0
    const queue = []
    const controller = new AbortController()

    const scheduleRetry = (delay) => {
      if (!disposed) retry = setTimeout(() => setAttempt((value) => value + 1), delay)
    }

    const draw = (bitmap) => {
      const canvas = canvasRef.current
      if (!canvas) return
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width
        canvas.height = bitmap.height
      }
      canvas.getContext('2d').drawImage(bitmap, 0, 0)
      if (!live) {
        live = true
        onStatus('live')
      }
    }

    const render = (now) => {
      raf = requestAnimationFrame(render)
      if (!playing) {
        if (queue.length < BUFFER_FRAMES) return
        playing = true
        nextDue = now
      }
      if (now < nextDue) return
      if (queue.length === 0) {
        // Ran dry: hold the last picture and keep playing. The server only
        // sends its newest frame, so nothing would arrive to refill a buffer
        // any faster; waiting for one would only lengthen the freeze.
        starved = true
        return
      }
      if (starved) {
        // Show the late frame now and restart the beat from here, so frames
        // queued behind it keep their spacing instead of flushing out at once.
        starved = false
        nextDue = now
      }
      const { seq, bitmap } = queue.shift()
      drawnSeq = seq
      draw(bitmap)
      bitmap.close()
      // Nudge the beat faster while the queue grows and slower while it
      // shrinks so latency stays bounded without ever running dry.
      const drift = Math.max(-2, Math.min(2, queue.length - BUFFER_FRAMES))
      const pace = interval * (1 - 0.05 * drift)
      nextDue = Math.max(nextDue + pace, now - pace)
    }

    const onFrame = async (blob) => {
      const now = performance.now()
      if (lastArrival) {
        const gap = now - lastArrival
        if (gap > 5 && gap < 250) interval = interval * 0.9 + gap * 0.1
      }
      lastArrival = now
      lastSeq += 1
      const seq = lastSeq
      // Decoder is behind: skip this frame instead of queueing more work.
      if (decoding > 2) return
      decoding += 1
      try {
        const bitmap = await createImageBitmap(blob)
        // Drop it if torn down or if a newer frame is already on screen.
        if (disposed || seq <= drawnSeq) {
          bitmap.close()
          return
        }
        // Decodes run concurrently and can finish out of order, so keep the
        // queue in arrival order rather than completion order.
        let index = queue.length
        while (index > 0 && queue[index - 1].seq > seq) index -= 1
        queue.splice(index, 0, { seq, bitmap })
        while (queue.length > MAX_QUEUE) queue.shift().bitmap.close()
      } catch {
        // Corrupt or truncated frame; the next one replaces it.
      } finally {
        decoding -= 1
      }
    }

    const connect = async () => {
      onStatus('connecting')
      let ticket
      try {
        ticket = await getStreamTicket(cameraId, controller.signal)
      } catch (error) {
        if (disposed) return
        onStatus('offline')
        if (error.status !== 401) scheduleRetry(error.status === 429 ? 30000 : 3000)
        return
      }
      if (disposed) return
      const current = new WebSocket(getStreamWebSocketUrl(cameraId, ticket))
      current.binaryType = 'blob'
      socket = current
      current.onmessage = (event) => {
        if (current === socket && event.data instanceof Blob) onFrame(event.data)
      }
      current.onclose = (event) => {
        if (current !== socket || disposed) return
        socket = null
        onStatus('offline')
        if (event.code !== 4401) scheduleRetry(3000)
      }
      raf = requestAnimationFrame(render)
    }

    connect()
    return () => {
      disposed = true
      controller.abort()
      clearTimeout(retry)
      cancelAnimationFrame(raf)
      const current = socket
      socket = null
      current?.close()
      while (queue.length) queue.shift().bitmap.close()
    }
  }, [cameraId, enabled, attempt, canvasRef, onStatus])

  return () => setAttempt((value) => value + 1)
}

function CameraFeed({ camera }) {
  const [status, setStatus] = useState(camera.online ? 'connecting' : 'offline')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const cellRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!camera.online) setStatus('offline')
  }, [camera.online])

  const reconnect = useFrameStream(camera.id, camera.online, canvasRef, setStatus)

  useEffect(() => {
    const handleFullscreenChange = () => {
      const node = cellRef.current
      setIsFullscreen(Boolean(node && document.fullscreenElement === node))
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const toggleFullscreen = async (event) => {
    event.preventDefault()
    event.stopPropagation()

    const node = cellRef.current
    if (!node) return

    try {
      if (document.fullscreenElement === node) {
        await document.exitFullscreen()
        return
      }
      if (document.fullscreenElement && document.fullscreenElement !== node) {
        await document.exitFullscreen()
      }
      await node.requestFullscreen()
    } catch (error) {
      console.error('Could not toggle fullscreen stream', error)
    }
  }

  return (
    <div
      ref={cellRef}
      className={`cam-cell ${isFullscreen ? 'is-fullscreen' : ''}`}
      onDoubleClick={toggleFullscreen}
    >
      <div className="cam-viewport">
        <button
          type="button"
          className="cam-expand-btn"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? `Shrink ${camera.name} stream` : `Enlarge ${camera.name} stream`}
          title={isFullscreen ? 'Exit fullscreen' : 'Enlarge stream'}
        >
          {isFullscreen ? (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 15H5v4" />
              <path d="M15 9h4V5" />
              <path d="M5 19l5-5" />
              <path d="M19 5l-5 5" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 3h6v6" />
              <path d="M9 21H3v-6" />
              <path d="M21 3l-7 7" />
              <path d="M3 21l7-7" />
            </svg>
          )}
        </button>

        {isFullscreen && (
          <div className="cam-fullscreen-meta">
            <div className="cam-fullscreen-name">{camera.name}</div>
            {status === 'live' && <div className="cam-fullscreen-badge">{camera.face_count} FACE{camera.face_count !== 1 ? 'S' : ''}</div>}
            {status === 'live' && <div className="cam-fullscreen-badge">{camera.fps} FPS</div>}
          </div>
        )}

        {status === 'connecting' && (
          <div className="cam-overlay">
            <div className="cam-spinner" />
            <div className="cam-overlay-text">Connecting...</div>
            <div className="cam-overlay-sub">{camera.name}</div>
          </div>
        )}

        {status === 'offline' && (
          <div className="cam-overlay cam-overlay-offline">
            <div className="cam-offline-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.9L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
                <path d="M3 3l18 18" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="cam-overlay-text">Feed Offline</div>
            <div className="cam-overlay-sub">{camera.name}</div>
            <button className="cam-retry-btn" onClick={reconnect}>
              Retry
            </button>
          </div>
        )}

        <canvas
          ref={canvasRef}
          className="cam-video"
          aria-label={`${camera.name} feed`}
          style={{ display: status === 'live' ? 'block' : 'none' }}
        />
      </div>

      <div className="cam-bar">
        <div className={`cam-rec ${status !== 'live' ? 'cam-rec-offline' : ''}`} />
        <div className="cam-name">{camera.name}</div>
        {status === 'live' && <div className="cam-meta-badge">{camera.face_count} FACE{camera.face_count !== 1 ? 'S' : ''}</div>}
        {status === 'live' && <div className="cam-meta-badge">{camera.fps} FPS</div>}
        <div className={`cam-status-badge ${
          status === 'live' ? 'cam-status-live' :
          status === 'connecting' ? 'cam-status-wait' : 'cam-status-off'
        }`}>
          {status === 'live' ? 'LIVE' :
            status === 'connecting' ? 'WAIT' : 'OFF'}
        </div>
      </div>
    </div>
  )
}

export default function CameraGrid({ cameras }) {
  const totalFaces = useMemo(
    () => cameras.reduce((sum, camera) => sum + (camera.face_count ?? 0), 0),
    [cameras],
  )

  return (
    <div className="panel cam-panel">
      <div className="panel-head">
        <div className="panel-title">Camera Grid</div>
        <div className="cam-total-badge">
          {totalFaces} track{totalFaces !== 1 ? 's' : ''} active
        </div>
      </div>
      <div className="cam-grid">
        {cameras.map((camera) => (
          <CameraFeed key={camera.id} camera={camera} />
        ))}
      </div>
    </div>
  )
}
