import { useState } from 'react'
import ChartTooltip from './ChartTooltip'

// Fixed slot order, taken from the backend's own person_type ordering. Colour
// follows the type, never its rank, so filtering or a quiet week never
// repaints a category the reader has already learned.
export const PERSON_TYPES = ['staff', 'vip', 'watchlist', 'visitor']
export const TYPE_COLORS = {
  staff: 'var(--series-1)',
  vip: 'var(--series-2)',
  watchlist: 'var(--series-3)',
  visitor: 'var(--series-4)',
  unknown: 'var(--t3)',
}

const typeColor = (type) => TYPE_COLORS[type] ?? TYPE_COLORS.unknown

/**
 * Most frequently recognised people, and the enrolment mix behind them.
 *
 * The ranking is a row per person with a bar for volume -- a horizontal bar is
 * the right form when the category names are long and the reader is comparing
 * magnitude. The mix is one stacked bar rather than a pie: the segments are
 * close in size often enough that a pie would leave them incomparable.
 */
export default function PersonLeaderboard({ persons, breakdown, onSelectPerson }) {
  const [tooltip, setTooltip] = useState(null)

  const busiest = Math.max(...persons.map((person) => person.events), 1)
  const mix = PERSON_TYPES
    .map((type) => ({ type, count: breakdown[type] ?? 0 }))
    .filter((entry) => entry.count > 0)
  const mixTotal = mix.reduce((sum, entry) => sum + entry.count, 0)

  if (!persons.length) {
    return <p className="an-note">No one was recognised in this window.</p>
  }

  return (
    <div className="pl">
      <ol className="pl-list">
        {persons.map((person) => (
          <li key={person.person_id} className="pl-row">
            <button
              type="button"
              className="pl-person"
              onClick={() => onSelectPerson?.(person)}
              title="Open in POI Profiles"
            >
              <span className="pl-dot" style={{ background: typeColor(person.person_type) }} aria-hidden="true" />
              <span className="pl-name">{person.person_name ?? 'Deleted person'}</span>
              <span className="pl-type">{person.person_type ?? 'unknown'}</span>
            </button>

            <div className="pl-bar-track">
              <div
                className="pl-bar"
                style={{
                  width: `${(person.events / busiest) * 100}%`,
                  background: typeColor(person.person_type),
                }}
              />
            </div>

            <span className="pl-events">{person.events}</span>
            <span className="pl-sim">{person.avg_similarity?.toFixed(3) ?? '--'}</span>
            <span className="pl-cameras" title={person.cameras.join(', ')}>
              {person.cameras.length === 1 ? person.cameras[0] : `${person.cameras.length} cameras`}
            </span>
            <span className="pl-seen">
              {person.last_seen ? new Date(person.last_seen).toLocaleString() : '--'}
            </span>
          </li>
        ))}
      </ol>

      {mixTotal > 0 && (
        <div className="pl-mix">
          <div className="pl-mix-head">
            <span className="pl-mix-title">Recognitions by enrolment type</span>
            <span className="pl-mix-total">{mixTotal} total</span>
          </div>

          <div className="pl-mix-bar" onPointerLeave={() => setTooltip(null)}>
            {mix.map((entry) => (
              <div
                key={entry.type}
                className="pl-mix-seg"
                style={{ width: `${(entry.count / mixTotal) * 100}%`, background: typeColor(entry.type) }}
                tabIndex={0}
                role="img"
                aria-label={`${entry.type}: ${entry.count} recognitions`}
                onPointerEnter={() => setTooltip({
                  x: 50,
                  y: -68,
                  title: entry.type,
                  rows: [{ label: 'recognitions', value: entry.count, color: typeColor(entry.type) }],
                })}
                onFocus={() => setTooltip({
                  x: 50,
                  y: -68,
                  title: entry.type,
                  rows: [{ label: 'recognitions', value: entry.count, color: typeColor(entry.type) }],
                })}
                onBlur={() => setTooltip(null)}
              />
            ))}
            <ChartTooltip tooltip={tooltip} />
          </div>

          {/* Every segment is also named and counted here, so the split never
              depends on reading the bar's colours. */}
          <ul className="pl-mix-legend">
            {mix.map((entry) => (
              <li key={entry.type} className="pl-mix-key">
                <span className="cc-swatch" style={{ background: typeColor(entry.type) }} aria-hidden="true" />
                {entry.type}
                <span className="pl-mix-count">{entry.count}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
