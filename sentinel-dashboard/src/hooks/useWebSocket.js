import { useEffect, useRef } from 'react'
import { subscribeLive } from '../utils/liveSocket'

// Kept for callers that only want raw pushed messages. Every hook now shares
// one socket (see utils/liveSocket.js); `url` is accepted for backwards
// compatibility but the shared connection always targets the events endpoint.
export function useWebSocket(_url, onMessage) {
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => subscribeLive((message) => onMessageRef.current?.(message)), [])
}
