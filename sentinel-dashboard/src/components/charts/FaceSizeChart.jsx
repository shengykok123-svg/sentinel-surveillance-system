import { useMemo, useState } from 'react'
import ChartTooltip from './ChartTooltip'
import { countScale } from './scale'
import { ACCEPTED_COLOR, REJECTED_COLOR } from './SimilarityHistogram'

const percent = (value) => (value == null ? '--' : `${Math.round(value * 100)}%`)

/**
 * Recognition outcome by detected face size.
 *
 * Recognised and unrecognised detections are stacked, so the column height is
 * the sample count and the split within it *is* the recognition rate -- both
 * readable from a single count axis. Plotting the rate as a second line over
 * its own scale would invent a relationship between two arbitrary axes; the
 * rate rides the column as a label instead.
 */
export default function FaceSizeChart({ bins }) {
  const [tooltip, setTooltip] = useState(null)

  const { max, ticks } = useMemo(
    () => countScale(Math.max(...bins.map((bin) => bin.samples), 1)),
    [bins],
  )

  const showTooltip = (bin, index) => {
    setTooltip({
      // Clamped so a mark at either edge keeps its readout inside the card.
      x: Math.min(Math.max(((index + 0.5) / bins.length) * 100, 14), 86),
      y: 8,
      title: bin.label,
      rows: [
        { label: 'Recognized', value: bin.recognized, color: ACCEPTED_COLOR },
        { label: 'Unrecognized', value: bin.samples - bin.recognized, color: REJECTED_COLOR },
        { label: 'Mean similarity', value: bin.avg_similarity?.toFixed(3) ?? '--', color: 'var(--t3)' },
      ],
    })
  }

  return (
    <div className="fsz">
      <div className="fsz-main">
        <div className="fsz-yaxis" aria-hidden="true">
          {ticks.map((tick) => (
            <span key={tick} className="fsz-ytick">{tick}</span>
          ))}
        </div>

        <div className="fsz-plot" onPointerLeave={() => setTooltip(null)}>
          <div className="fsz-grid" aria-hidden="true">
            {ticks.map((tick) => <div key={tick} className="fsz-gridline" />)}
          </div>

          <div className="fsz-bars">
            {bins.map((bin, index) => {
              const unrecognized = bin.samples - bin.recognized
              const fill = (bin.samples / max) * 100
              return (
                <div
                  key={bin.label}
                  className="fsz-slot"
                  tabIndex={0}
                  role="img"
                  aria-label={`${bin.label}: ${bin.samples} detections, ${percent(bin.recognition_rate)} recognized`}
                  onPointerEnter={() => showTooltip(bin, index)}
                  onFocus={() => showTooltip(bin, index)}
                  onBlur={() => setTooltip(null)}
                >
                  {/* The rate is what the chart is about, so it rides the cap
                      of every column rather than living only in the tooltip.
                      A column too tall to clear its own cap carries the label
                      inside the top segment, where it still fits. */}
                  <span
                    className={`fsz-rate ${fill > 88 ? 'inside' : ''}`}
                    style={{ bottom: `calc(${Math.min(fill, 88)}% + 3px)` }}
                  >
                    {percent(bin.recognition_rate)}
                  </span>
                  <div className="fsz-stack">
                    {bin.recognized > 0 && (
                      <div
                        className="fsz-seg fsz-seg-top"
                        style={{ height: `${(bin.recognized / max) * 100}%`, background: ACCEPTED_COLOR }}
                      />
                    )}
                    {unrecognized > 0 && (
                      <div
                        className={`fsz-seg ${bin.recognized > 0 ? '' : 'fsz-seg-top'}`}
                        style={{ height: `${(unrecognized / max) * 100}%`, background: REJECTED_COLOR }}
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

      <div className="fsz-xaxis" aria-hidden="true">
        {bins.map((bin) => (
          <span key={bin.label} className="fsz-xtick">{bin.label}</span>
        ))}
      </div>
    </div>
  )
}
