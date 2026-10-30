import { useMemo, useState } from 'react'
import './POIPanel.css'

const STATUS = {
  active: { label: 'Active', cls: 's-active' },
  inactive: { label: 'Inactive', cls: 's-inactive' },
}

function getStatusKey(person) {
  return person.status === 'active' ? 'active' : 'inactive'
}

function formatLastSeen(value) {
  if (!value) {
    return 'No recent match'
  }
  return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function shortId(id) {
  const text = String(id ?? '')
  // UUIDs are shown by their first block only; short numeric ids stay as-is.
  return text.length > 12 ? text.slice(0, 8) : text
}

function getCardAccent(person) {
  if (getStatusKey(person) === 'active') {
    return (person.person_type ?? '').toLowerCase() === 'watchlist'
      ? 'accent-alert'
      : 'accent-active'
  }
  return 'accent-inactive'
}

export default function POIPanel({ people, onOpenProfile }) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    return people
      .filter((person) =>
        !normalizedQuery ||
        person.name.toLowerCase().includes(normalizedQuery) ||
        String(person.id).toLowerCase().includes(normalizedQuery) ||
        String(person.person_type ?? '').toLowerCase().includes(normalizedQuery),
      )
      .sort((left, right) => {
        const leftActive = getStatusKey(left) === 'active' ? 1 : 0
        const rightActive = getStatusKey(right) === 'active' ? 1 : 0
        if (leftActive !== rightActive) {
          return rightActive - leftActive
        }
        const leftSeen = left.last_seen_time ? new Date(left.last_seen_time).getTime() : 0
        const rightSeen = right.last_seen_time ? new Date(right.last_seen_time).getTime() : 0
        return rightSeen - leftSeen || left.name.localeCompare(right.name)
      })
  }, [people, query])

  const openProfile = (person) => onOpenProfile?.(person.id)

  return (
    <div className="panel poi-panel">
      <div className="panel-head">
        <div>
          <div className="panel-title">Persons of Interest</div>
          <div className="pl-head-sub">Registry and latest sightings</div>
        </div>
        <span className="pl-count">{people.length}</span>
      </div>

      <div className="pl-search">
        <input
          className="pl-input"
          placeholder="Search name, ID, or type..."
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>

      <div className="pl-list">
        {filtered.map((person) => {
          const status = STATUS[getStatusKey(person)] ?? STATUS.inactive
          const location = person.last_seen_camera
            ? `${person.last_seen_camera} | ${formatLastSeen(person.last_seen_time)}`
            : 'No recognition event yet'
          const personType = person.person_type ?? 'watchlist'

          return (
            <div
              key={person.id}
              className={`pl-card ${getCardAccent(person)}`}
              role="button"
              tabIndex={0}
              title={`Open profile for ${person.name}`}
              onClick={() => openProfile(person)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  openProfile(person)
                }
              }}
            >
              <div className="pl-card-glow" />

              <div className="pl-main">
                <div className="pl-topline">
                  <div className="pl-name">{person.name}</div>
                  <div className={`pl-status ${status.cls}`}>{status.label}</div>
                </div>

                <div className="pl-meta">
                  <span className="pl-id" title={`ID ${person.id}`}>#{shortId(person.id)}</span>
                  <span className="pl-sep">·</span>
                  <span className={`pl-type type-${String(personType).toLowerCase()}`}>{personType}</span>
                  <span className="pl-sep">·</span>
                  <span>{person.total_embeddings ?? 0} emb</span>
                  {!person.is_active && (
                    <>
                      <span className="pl-sep">·</span>
                      <span className="pl-disabled">Disabled</span>
                    </>
                  )}
                </div>

                <div className="pl-bottomline">
                  <div className="pl-loc" title={location}>{location}</div>
                  <button
                    className="pl-profile-btn"
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation()
                      openProfile(person)
                    }}
                  >
                    Profile →
                  </button>
                </div>
              </div>
            </div>
          )
        })}

        {filtered.length === 0 && (
          <div className="pl-empty">
            <div className="pl-empty-title">No matching persons of interest</div>
            <div className="pl-empty-copy">Try a different name, ID, or person type.</div>
          </div>
        )}
      </div>
    </div>
  )
}
