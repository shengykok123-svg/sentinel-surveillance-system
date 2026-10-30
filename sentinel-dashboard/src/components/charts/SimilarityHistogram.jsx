import { useMemo, useState } from 'react'
import ChartTooltip from './ChartTooltip'
import { countScale } from './scale'

export const ACCEPTED_COLOR = 'var(--blue)'
export const REJECTED_COLOR = 'var(--amber)'

/**
 * Distribution of match scores either side of the recognition threshold.
 *
 * The two populations are stacked per bin rather than blended, so the reader
 * sees where the classifier's decisions actually fall relative to the cut-off.
 * The simulator re-partitions the same bins at a candidate threshold; because
 * it snaps to bin edges the counts it reports are exact rather than an
 * interpolation across a bin.
 */
export default function SimilarityHistogram({ bins, threshold, simulated, onSimulate }) {
  const [tooltip, setTooltip] = useState(null)

  const { max, ticks } = useMemo(
    () => countScale(Math.max(...bins.map((bin) => bin.accepted + bin.rejected), 1)),
    [bins],
  )
  const cutoff = simulated ?? threshold

  const partition = (bin) => {
    const total = bin.accepted + bin.rejected
    // Bin edges make this exact: a bin sits wholly on one side of any
    // edge-aligned cutoff, so nothing has to be interpolated within it.
    const above = bin.lower >= cutoff ? total : 0
    return { total, above, below: total - above }
  }

  const showTooltip = (bin, index) => {
    const { total, above, below } = partition(bin)
    setTooltip({
      // Clamped so a mark at either edge keeps its readout inside the card.
      x: Math.min(Math.max(((index + 0.5) / bins.length) * 100, 14), 86),
      y: 8,
      title: `${bin.lower.toFixed(2)} – ${bin.upper.toFixed(2)}`,
      rows: [
        { label: 'Above threshold', value: above, color: ACCEPTED_COLOR },
        { label: 'Below threshold', value: below, color: REJECTED_COLOR },
        { label: 'Total', value: total, color: 'var(--t3)' },
      ],
    })
  }

  return (
    <div className="hist">
      <div className="hist-main">
        <div className="hist-yaxis" aria-hidden="true">
          {ticks.map((tick) => (
            <span key={tick} className="hist-ytick">{tick}</span>
          ))}
        </div>

        <div className="hist-plot" onPointerLeave={() => setTooltip(null)}>
          <div className="hist-grid" aria-hidden="true">
            {ticks.map((tick) => <div key={tick} className="hist-gridline" />)}
          </div>

          <div className="hist-bars">
            {bins.map((bin, index) => {
              const { above, below } = partition(bin)
              return (
                <div
                  key={bin.lower}
                  className="hist-slot"
                  tabIndex={0}
                  role="img"
                  aria-label={`Similarity ${bin.lower.toFixed(2)} to ${bin.upper.toFixed(2)}: ${above} above threshold, ${below} below`}
                  onPointerEnter={() => showTooltip(bin, index)}
                  onFocus={() => showTooltip(bin, index)}
                  onBlur={() => setTooltip(null)}
                >
                  <div className="hist-stack">
                    {above > 0 && (
                      <div
                        className="hist-seg hist-seg-top"
                        style={{ height: `${(above / max) * 100}%`, background: ACCEPTED_COLOR }}
                      />
                    )}
                    {below > 0 && (
                      <div
                        className={`hist-seg ${above > 0 ? '' : 'hist-seg-top'}`}
                        style={{ height: `${(below / max) * 100}%`, background: REJECTED_COLOR }}
                      />
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* The configured cut-off stays visible while a candidate is dragged,
              so the comparison is always against the deployed value. */}
          <div className="hist-threshold" style={{ left: `${threshold * 100}%` }}>
            <span className="hist-threshold-tag">{threshold.toFixed(2)}</span>
          </div>
          {simulated != null && simulated !== threshold && (
            <div className="hist-threshold simulated" style={{ left: `${simulated * 100}%` }}>
              <span className="hist-threshold-tag">{simulated.toFixed(2)}</span>
            </div>
          )}

          <ChartTooltip tooltip={tooltip} />
        </div>
      </div>

      <div className="hist-xaxis" aria-hidden="true">
        {[0, 0.25, 0.5, 0.75, 1].map((mark) => (
          <span key={mark} className="hist-xtick" style={{ left: `${mark * 100}%` }}>
            {mark.toFixed(2)}
          </span>
        ))}
      </div>

      <label className="hist-slider">
        <span className="hist-slider-label">Simulate threshold</span>
        <input
          type="range"
          min={0}
          max={1}
          step={bins.length ? 1 / bins.length : 0.05}
          value={cutoff}
          onChange={(event) => onSimulate(Number(event.target.value))}
          aria-label="Simulated recognition threshold"
        />
        <span className="hist-slider-value">{cutoff.toFixed(2)}</span>
        <button type="button" className="hist-slider-reset" onClick={() => onSimulate(null)}>
          Reset
        </button>
      </label>
    </div>
  )
}
