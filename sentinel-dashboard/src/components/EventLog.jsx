import { useMemo, useState } from 'react'
import './EventLog.css'

const ROLE_TONE = {
  watchlist: {
    accent: '#dc2626',
    bg: 'rgba(220,38,38,0.1)',
    border: 'rgba(220,38,38,0.2)',
  },
  vip: {
    accent: '#d97706',
    bg: 'rgba(217,119,6,0.1)',
    border: 'rgba(217,119,6,0.2)',
  },
  staff: {
    accent: '#16a34a',
    bg: 'rgba(22,163,74,0.1)',
    border: 'rgba(22,163,74,0.2)',
  },
  visitor: {
    accent: '#2563eb',
    bg: 'rgba(37,99,235,0.1)',
    border: 'rgba(37,99,235,0.2)',
  },
}

const SEVERITY_TONE = {
  alert: { accent: 'var(--red)', bg: 'rgba(220,38,38,0.1)', border: 'rgba(220,38,38,0.18)' },
  warn: { accent: 'var(--amber)', bg: 'rgba(217,119,6,0.08)', border: 'rgba(217,119,6,0.16)' },
  ok: { accent: 'var(--green)', bg: 'rgba(22,163,74,0.08)', border: 'rgba(22,163,74,0.16)' },
}

function getEventTone(event) {
  if (event.status === 'recognized' && event.person_type) {
    return ROLE_TONE[String(event.person_type).toLowerCase()] ?? SEVERITY_TONE[event.severity]
  }
  return SEVERITY_TONE[event.severity] ?? SEVERITY_TONE.warn
}

function formatPersonType(personType) {
  const value = String(personType ?? '').trim()
  if (!value) {
    return ''
  }
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function eventSearchText(event) {
  return [
    event.type,
    event.camera,
    event.time,
    event.track_id,
    event.person_name,
    event.person_type,
    event.status,
    event.severity,
    event.conf,
  ]
    .filter((value) => value !== null && value !== undefined)
    .join(' ')
    .toLowerCase()
}

function EventItem({ event, onOpenRecord }) {
  const tone = getEventTone(event)
  const personType = formatPersonType(event.person_type)
  const metaParts = [event.camera, event.time]
  if (event.track_id) {
    metaParts.push(`Track ${event.track_id}`)
  }
  const meta = metaParts.filter(Boolean).join(' | ')
  const canOpen = event.status === 'recognized' && Boolean(onOpenRecord)

  return (
    <div className={`ev-item ev-item-${event.severity}`} style={{ background: tone.bg, borderColor: tone.border }}>
      <div className="ev-dot" style={{ background: tone.accent }} />
      <div className="ev-body">
        <div className="ev-head">
          <div className="ev-type">{event.type}</div>
          {personType && (
            <span className="ev-role" style={{ color: tone.accent, borderColor: tone.border }}>
              {personType}
            </span>
          )}
          <div className="ev-conf" style={{ color: tone.accent }}>
            {event.conf}
          </div>
        </div>
        <div className="ev-bottom">
          <div className="ev-meta" title={event.person_name ? `${meta} | ${event.person_name}` : meta}>
            {meta}
            {event.person_name && (
              <>
                <span className="ev-sep"> | </span>
                <span className="ev-person">{event.person_name}</span>
              </>
            )}
          </div>
          {canOpen && (
            <button className="ev-record-btn" type="button" onClick={() => onOpenRecord(event.id)}>
              Record →
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default function EventLog({ events, fullPage, onClear, onOpenRecord }) {
  const [query, setQuery] = useState('')
  const [cameraFilter, setCameraFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [severityFilter, setSeverityFilter] = useState('all')

  const cameras = useMemo(
    () => [...new Set(events.map((event) => event.camera).filter(Boolean))].sort(),
    [events],
  )
  const filteredEvents = useMemo(() => {
    const search = query.trim().toLowerCase()
    return events.filter((event) => (
      (!search || eventSearchText(event).includes(search))
      && (cameraFilter === 'all' || event.camera === cameraFilter)
      && (statusFilter === 'all' || event.status === statusFilter)
      && (severityFilter === 'all' || event.severity === severityFilter)
    ))
  }, [events, query, cameraFilter, statusFilter, severityFilter])
  const visibleEvents = fullPage ? filteredEvents : events.slice(0, 8)
  const filtersActive = query || cameraFilter !== 'all' || statusFilter !== 'all' || severityFilter !== 'all'

  const resetFilters = () => {
    setQuery('')
    setCameraFilter('all')
    setStatusFilter('all')
    setSeverityFilter('all')
  }

  const handleClear = async () => {
    if (!onClear) {
      return
    }
    const confirmed = window.confirm('Clear all live event logs? This removes the current event history list.')
    if (!confirmed) {
      return
    }
    await onClear()
  }

  if (fullPage) {
    return (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 12, gap: 10, overflow: 'hidden' }}>
        <div className="panel" style={{ flex: 1 }}>
          <div className="panel-head">
            <div className="panel-title">Audit Log</div>
            <button className="ev-clear-btn" type="button" onClick={handleClear}>
              Clear
            </button>
          </div>
          <div className="ev-toolbar">
            <label className="ev-search-field">
              <span className="ev-field-label">Search</span>
              <input
                className="ev-input"
                type="search"
                placeholder="Person, camera, track, event..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            <label className="ev-filter-field">
              <span className="ev-field-label">Camera</span>
              <select className="ev-select" value={cameraFilter} onChange={(event) => setCameraFilter(event.target.value)}>
                <option value="all">All cameras</option>
                {cameras.map((camera) => <option key={camera} value={camera}>{camera}</option>)}
              </select>
            </label>
            <label className="ev-filter-field">
              <span className="ev-field-label">Status</span>
              <select className="ev-select" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                <option value="all">All status</option>
                <option value="recognized">Recognized</option>
                <option value="unknown">Unknown</option>
              </select>
            </label>
            <label className="ev-filter-field">
              <span className="ev-field-label">Severity</span>
              <select className="ev-select" value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value)}>
                <option value="all">All severity</option>
                <option value="alert">Alert</option>
                <option value="warn">Warning</option>
                <option value="ok">Normal</option>
              </select>
            </label>
            <button className="ev-reset-btn" type="button" onClick={resetFilters} disabled={!filtersActive}>
              Reset
            </button>
          </div>
          <div className="ev-results-summary" aria-live="polite">
            {filteredEvents.length} of {events.length} events shown
          </div>
          <div className="ev-list" style={{ flex: 1, overflowY: 'auto', padding: '6px 8px' }}>
            {visibleEvents.length
              ? visibleEvents.map((event) => <EventItem key={event.id} event={event} onOpenRecord={onOpenRecord} />)
              : <div className="ev-empty">No audit events match the current search and filters.</div>}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div className="panel-title">Live Events</div>
        <button className="ev-clear-btn" type="button" onClick={handleClear}>
          Clear
        </button>
      </div>
      <div className="ev-list">
        {visibleEvents.map((event) => <EventItem key={event.id} event={event} onOpenRecord={onOpenRecord} />)}
      </div>
    </div>
  )
}
