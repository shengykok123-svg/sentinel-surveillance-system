import { useEffect, useState } from 'react'
import './Topbar.css'

export default function Topbar({ alertCount, theme, onToggleTheme, cameraCount, currentUser, onLogout }) {
  const [time, setTime] = useState('')

  useEffect(() => {
    const tick = () => setTime(new Date().toTimeString().slice(0, 8))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="topbar">
      <div className="tb-title">Live Surveillance | {cameraCount} Camera Feeds</div>

      <div className="tb-clock">{time}</div>

      <div className="tb-pill tb-live">
        <div className="tb-dot" />
        LIVE
      </div>

      {alertCount > 0 && (
        <div className="tb-pill tb-alert">
          <div className="tb-dot" />
          {alertCount} POI ALERT{alertCount !== 1 ? 'S' : ''}
        </div>
      )}

      {currentUser && (
        <div className="tb-pill tb-user">
          {currentUser.username}
          {currentUser.roles?.length ? ` (${currentUser.roles.join(', ')})` : ''}
        </div>
      )}

      <button className="icon-btn" onClick={onToggleTheme} title="Toggle theme">
        {theme === 'dark' ? (
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.1 3.1l1.4 1.4M11.5 11.5l1.4 1.4M3.1 12.9l1.4-1.4M11.5 4.5l1.4-1.4"/>
            <circle cx="8" cy="8" r="2.5"/>
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M13 8.5A5.5 5.5 0 117.5 3c-.3 1.8.5 4.5 3 5.5 1.3.5 2.5.1 2.5 0z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </button>

      {currentUser && (
        <button className="icon-btn logout-btn" onClick={onLogout} title="Logout">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M6 2H3.5A1.5 1.5 0 002 3.5v9A1.5 1.5 0 003.5 14H6" strokeLinecap="round" />
            <path d="M9.5 4.5L13 8l-3.5 3.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M5 8h8" strokeLinecap="round" />
          </svg>
        </button>
      )}
    </header>
  )
}
