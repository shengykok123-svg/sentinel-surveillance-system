import { useCallback, useState } from 'react'

// Remembers a collapsed/expanded flag per section across reloads.
export function useCollapsed(key, initial = false) {
  const storageKey = `sentinel.collapsed.${key}`
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const stored = localStorage.getItem(storageKey)
      return stored === null ? initial : stored === '1'
    } catch {
      return initial
    }
  })

  const toggle = useCallback(() => {
    setCollapsed((current) => {
      const next = !current
      try { localStorage.setItem(storageKey, next ? '1' : '0') } catch { /* storage unavailable */ }
      return next
    })
  }, [storageKey])

  return [collapsed, toggle]
}
