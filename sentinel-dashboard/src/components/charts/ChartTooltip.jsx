/**
 * Floating readout for a hovered or focused mark.
 *
 * Values lead and the series name follows: the reader already knows which mark
 * they are pointing at, so the number is the thing they came for. Series are
 * keyed with a short stroke rather than a filled box, which would be
 * data-weight ink doing a label's job.
 */
export default function ChartTooltip({ tooltip }) {
  if (!tooltip) {
    return null
  }

  return (
    <div
      className="cc-tooltip"
      role="status"
      style={{ left: `${tooltip.x}%`, top: `${tooltip.y}px` }}
    >
      <div className="cc-tip-title">{tooltip.title}</div>
      {tooltip.rows.map((row) => (
        <div key={row.label} className="cc-tip-row">
          <span className="cc-tip-key" style={{ background: row.color }} aria-hidden="true" />
          <span className="cc-tip-value">{row.value}</span>
          <span className="cc-tip-label">{row.label}</span>
        </div>
      ))}
    </div>
  )
}
