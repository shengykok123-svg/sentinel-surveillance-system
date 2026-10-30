import { useState } from 'react'
import './Sidebar.css'

const NAV = [
  { group: 'Monitor', items: [
    { id: 'live',      label: 'Live Dashboard',      ping: true,  badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="1" y="1" width="14" height="14" rx="2"/><path d="M4 9l3 3 5-6" strokeLinecap="round" strokeLinejoin="round"/></svg> },
  ]},
  { group: 'People', items: [
    { id: 'search',    label: 'POI Profiles',         ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="7" cy="6" r="3"/><path d="M2 14c0-3 2.2-5 5-5"/><circle cx="12" cy="9" r="2"/><path d="M10 14v-.5a2 2 0 014 0V14"/></svg> },
    { id: 'enroll',    label: 'Enrollment',           ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="8" cy="6" r="2.5"/><path d="M3 14c0-2.8 2.2-5 5-5s5 2.2 5 5"/><path d="M11 3v3M9.5 4.5h3"/></svg> },
    { id: 'records',   label: 'Recognition Records',  ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="2" y="1.5" width="12" height="13" rx="1.5"/><path d="M5 5h6M5 8h6M5 11h3" strokeLinecap="round"/></svg> },
  ]},
  { group: 'Insights', items: [
    { id: 'analytics', label: 'Analytics',            ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M2 12l3.5-4.5L8 10l3-5 3 2" strokeLinecap="round" strokeLinejoin="round"/></svg> },
    { id: 'resources', label: 'Resource Analytics',   ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="1" y="8" width="3" height="6" rx="1"/><rect x="6" y="5" width="3" height="9" rx="1"/><rect x="11" y="2" width="3" height="12" rx="1"/></svg> },
    { id: 'events',    label: 'Audit Log',            ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="8" cy="8" r="6.5"/><path d="M8 5v3.5l2 1.5"/></svg> },
  ]},
  { group: 'System', items: [
    { id: 'settings',  label: 'Settings',             ping: false, badge: null,
      icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="8" cy="8" r="2"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.1 3.1l1.4 1.4M11.5 11.5l1.4 1.4M3.1 12.9l1.4-1.4M11.5 4.5l1.4-1.4"/></svg> },
  ]},
]

export default function Sidebar({ activePage, onNavigate }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <aside className={`sidebar ${expanded ? 'expanded' : ''}`}>
      <div className="sb-logo">
        <div className="sb-logo-icon">
          <svg viewBox="0 0 16 16" fill="none" stroke="white" strokeWidth="2">
            <circle cx="8" cy="8" r="3.5"/>
            <circle cx="8" cy="8" r="6.5" strokeDasharray="2.5 2"/>
          </svg>
        </div>
        <span className="sb-logo-text">SENTINEL</span>
      </div>

      <nav className="sb-nav">
        {NAV.map(({ group, items }) => (
          <div key={group}>
            <div className="sb-group-label">{group}</div>
            {items.map(item => (
              <div
                key={item.id}
                className={`sb-item ${activePage === item.id ? 'active' : ''}`}
                onClick={() => onNavigate(item.id)}
                title={item.label}
              >
                <span className="sb-icon">{item.icon}</span>
                <span className="sb-label">{item.label}</span>
                {item.badge && <span className="sb-badge">{item.badge}</span>}
                {item.ping  && <span className="sb-ping" />}
              </div>
            ))}
          </div>
        ))}
      </nav>

      <div className="sb-toggle" onClick={() => setExpanded(e => !e)}>
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"
          style={{ transform: expanded ? 'scaleX(-1)' : 'none', transition: 'transform 0.25s' }}>
          <path d="M6 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
    </aside>
  )
}
