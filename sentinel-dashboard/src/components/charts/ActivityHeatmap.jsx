import { useMemo, useState } from 'react'
import ChartTooltip from './ChartTooltip'

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const RAMP = ['var(--heat-1)', 'var(--heat-2)', 'var(--heat-3)', 'var(--heat-4)']

/**
 * Detections by weekday and hour of day.
 *
 * Magnitude on a grid, so the encoding is a single-hue ramp -- four steps
 * rather than a continuous gradient, because a reader can only place a cell on
 * a scale they can count. A cell with nothing in it stays surface-coloured, so
 * "no activity" never reads as "the lowest band of activity".
 *
 * Hours are the ones the caller asked for: the API buckets cells in the
 * viewer's timezone, not UTC.
 */
export default function ActivityHeatmap({ cells }) {
  const [tooltip, setTooltip] = useState(null)

  const { grid, max } = useMemo(() => {
    const lookup = new Map(cells.map((cell) => [`${cell.weekday}:${cell.hour}`, cell.count]))
    return {
      grid: lookup,
      max: Math.max(...cells.map((cell) => cell.count), 0),
    }
  }, [cells])

  // Equal-width bands over the observed range; the legend spells out what each
  // step covers so the ramp is never read by eye alone.
  const band = (count) => (count <= 0 ? -1 : Math.min(Math.floor(((count - 1) / max) * RAMP.length), RAMP.length - 1))
  const bandBound = (index) => Math.ceil(((index + 1) / RAMP.length) * max)

  return (
    <div className="hm">
      <div className="hm-grid-wrap">
        <div className="hm-rowlabels" aria-hidden="true">
          {WEEKDAYS.map((day) => <span key={day} className="hm-rowlabel">{day}</span>)}
        </div>

        <div className="hm-grid" onPointerLeave={() => setTooltip(null)}>
          {WEEKDAYS.map((day, weekday) => (
            <div key={day} className="hm-row">
              {Array.from({ length: 24 }, (_, hour) => {
                const count = grid.get(`${weekday}:${hour}`) ?? 0
                const index = band(count)
                return (
                  <div
                    key={hour}
                    className={`hm-cell ${count > 0 ? 'filled' : ''}`}
                    style={index >= 0 ? { background: RAMP[index] } : undefined}
                    tabIndex={0}
                    role="img"
                    aria-label={`${day} ${hour}:00 — ${count} detections`}
                    onPointerEnter={() => setTooltip({
                      x: Math.min(Math.max(((hour + 0.5) / 24) * 100, 14), 86),
                      y: 4,
                      title: `${day} ${String(hour).padStart(2, '0')}:00`,
                      rows: [{ label: 'Detections', value: count, color: index >= 0 ? RAMP[index] : 'var(--t3)' }],
                    })}
                    onFocus={() => setTooltip({
                      x: Math.min(Math.max(((hour + 0.5) / 24) * 100, 14), 86),
                      y: 4,
                      title: `${day} ${String(hour).padStart(2, '0')}:00`,
                      rows: [{ label: 'Detections', value: count, color: index >= 0 ? RAMP[index] : 'var(--t3)' }],
                    })}
                    onBlur={() => setTooltip(null)}
                  />
                )
              })}
            </div>
          ))}
          <ChartTooltip tooltip={tooltip} />
        </div>
      </div>

      <div className="hm-xaxis" aria-hidden="true">
        {Array.from({ length: 24 }, (_, hour) => (
          <span key={hour} className="hm-xtick">{hour % 6 === 0 ? String(hour).padStart(2, '0') : ''}</span>
        ))}
      </div>

      {/* A magnitude ramp needs its scale spelled out, not inferred. */}
      <div className="hm-scale">
        <span className="hm-scale-label">Detections</span>
        <span className="hm-scale-step"><span className="hm-swatch empty" />0</span>
        {RAMP.map((color, index) => (
          <span key={color} className="hm-scale-step">
            <span className="hm-swatch" style={{ background: color }} />
            {index === 0 ? `1–${bandBound(0)}` : `≤${bandBound(index)}`}
          </span>
        ))}
      </div>
    </div>
  )
}
