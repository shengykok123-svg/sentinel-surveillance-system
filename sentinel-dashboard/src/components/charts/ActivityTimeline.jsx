import { useMemo, useState } from 'react'
import ChartTooltip from './ChartTooltip'
import { countScale } from './scale'
import { ACCEPTED_COLOR, REJECTED_COLOR } from './SimilarityHistogram'

export const WATCHLIST_COLOR = 'var(--red)'

function formatBucket(ts, bucket) {
  const date = new Date(ts)
  return bucket === 'hour'
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

/**
 * Recognition activity over time.
 *
 * Recognised and unknown stack to the total, because together they are the
 * whole of what was detected. Watchlist hits do not join the stack: they are a
 * subset of the recognised events, so a third band would count them twice.
 * They ride above the column as markers instead, which also suits how rare and
 * how consequential they are.
 */
export default function ActivityTimeline({ buckets, bucket }) {
  const [tooltip, setTooltip] = useState(null)

  const { max, ticks } = useMemo(
    () => countScale(Math.max(...buckets.map((slot) => slot.recognized + slot.unknown), 1)),
    [buckets],
  )

  // Past a couple of dozen columns every label would collide, so only every
  // nth is drawn and the tooltip carries the rest.
  const labelEvery = Math.max(1, Math.ceil(buckets.length / 8))

  const showTooltip = (slot, index) => {
    setTooltip({
      x: Math.min(Math.max(((index + 0.5) / buckets.length) * 100, 14), 86),
      y: 8,
      title: new Date(slot.ts).toLocaleString(),
      rows: [
        { label: 'Recognized', value: slot.recognized, color: ACCEPTED_COLOR },
        { label: 'Unknown', value: slot.unknown, color: REJECTED_COLOR },
        { label: 'Watchlist', value: slot.watchlist, color: WATCHLIST_COLOR },
      ],
    })
  }

  return (
    <div className="tl">
      <div className="tl-main">
        <div className="tl-yaxis" aria-hidden="true">
          {ticks.map((tick) => (
            <span key={tick} className="tl-ytick">{tick}</span>
          ))}
        </div>

        <div className="tl-plot" onPointerLeave={() => setTooltip(null)}>
          <div className="tl-grid" aria-hidden="true">
            {ticks.map((tick) => <div key={tick} className="tl-gridline" />)}
          </div>

          <div className="tl-bars">
            {buckets.map((slot, index) => {
              const total = slot.recognized + slot.unknown
              return (
                <div
                  key={slot.ts}
                  className="tl-slot"
                  tabIndex={0}
                  role="img"
                  aria-label={`${new Date(slot.ts).toLocaleString()}: ${slot.recognized} recognized, ${slot.unknown} unknown, ${slot.watchlist} watchlist`}
                  onPointerEnter={() => showTooltip(slot, index)}
                  onFocus={() => showTooltip(slot, index)}
                  onBlur={() => setTooltip(null)}
                >
                  {slot.watchlist > 0 && (
                    <span
                      className="tl-alert"
                      style={{ bottom: `calc(${Math.min((total / max) * 100, 92)}% + 5px)` }}
                      title={`${slot.watchlist} watchlist hits`}
                    />
                  )}
                  <div className="tl-stack">
                    {slot.unknown > 0 && (
                      <div
                        className="tl-seg tl-seg-top"
                        style={{ height: `${(slot.unknown / max) * 100}%`, background: REJECTED_COLOR }}
                      />
                    )}
                    {slot.recognized > 0 && (
                      <div
                        className={`tl-seg ${slot.unknown > 0 ? '' : 'tl-seg-top'}`}
                        style={{ height: `${(slot.recognized / max) * 100}%`, background: ACCEPTED_COLOR }}
                      />
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          <ChartTooltip tooltip={tooltip} />
        </div>
      </div>

      <div className="tl-xaxis" aria-hidden="true">
        {buckets.map((slot, index) => (
          <span key={slot.ts} className="tl-xtick">
            {index % labelEvery === 0 ? formatBucket(slot.ts, bucket) : ''}
          </span>
        ))}
      </div>
    </div>
  )
}
