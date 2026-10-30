import { useState } from 'react'
export const defaults = { notifications: true, sound: false, cooldown: 30, theme: 'dark' }
export function usePreferences(userId) {
  const key = `sentinel.preferences.${userId}`
  const [preferences, setPreferences] = useState(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem(key) || '{}') } }
    catch { return defaults }
  })
  function savePreferences(next) {
    localStorage.setItem(key, JSON.stringify(next))
    setPreferences(next)
  }
  return [preferences, savePreferences]
}
let audioContext
export async function playAlertSound() {
  const Audio = window.AudioContext || window.webkitAudioContext
  if (!Audio) return
  audioContext ||= new Audio()
  await audioContext.resume()
  const oscillator = audioContext.createOscillator()
  const gain = audioContext.createGain()
  oscillator.connect(gain); gain.connect(audioContext.destination)
  oscillator.frequency.value = 740
  gain.gain.setValueAtTime(0.08, audioContext.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.25)
  oscillator.start(); oscillator.stop(audioContext.currentTime + 0.25)
}
