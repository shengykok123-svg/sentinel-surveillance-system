// One shared WebSocket to /ws/events for the whole dashboard.
//
// Every message the backend pushes (recognition events and the once-a-second
// `stats` snapshot of cameras, active tracks and host resources) fans out to
// every subscriber, so the hooks that used to poll four REST endpoints share
// a single connection and make no HTTP requests at all while it is open.
import { getAccessToken, getEventsWebSocketUrl, reportExpiredSession } from './api'

const listeners = new Set()
let socket = null
let retry = null
let ping = null
let connected = false

function connect() {
  if (socket || listeners.size === 0) return
  const token = getAccessToken()
  if (!token) return
  const current = new WebSocket(getEventsWebSocketUrl())
  socket = current
  current.onopen = () => {
    current.send(JSON.stringify({ type: 'authenticate', token }))
  }
  current.onmessage = (event) => {
    if (current !== socket) return
    let message
    try {
      message = JSON.parse(event.data)
    } catch {
      return
    }
    if (message.type === 'authenticated') {
      connected = true
      clearInterval(ping)
      ping = setInterval(() => {
        if (current.readyState === WebSocket.OPEN) current.send('ping')
      }, 15000)
      return
    }
    listeners.forEach((listener) => {
      try {
        listener(message)
      } catch (error) {
        console.error('Live socket listener failed', error)
      }
    })
  }
  current.onclose = (event) => {
    if (current !== socket) return
    socket = null
    connected = false
    clearInterval(ping)
    if (event.code === 4401) {
      reportExpiredSession(token)
      return
    }
    if (listeners.size > 0) retry = setTimeout(connect, 3000)
  }
}

function disconnect() {
  clearTimeout(retry)
  clearInterval(ping)
  connected = false
  const current = socket
  socket = null
  current?.close()
}

/** Subscribe to every pushed message. Returns an unsubscribe function. */
export function subscribeLive(listener) {
  listeners.add(listener)
  connect()
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) disconnect()
  }
}

/** True while the shared socket is open and authenticated. */
export function isLiveConnected() {
  return connected
}
