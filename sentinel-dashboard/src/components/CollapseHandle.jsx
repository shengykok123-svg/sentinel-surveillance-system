import './CollapseHandle.css'

/**
 * Small arrow handle that toggles a section.
 * `axis` is 'vertical' for a section above the handle (arrow up = collapse, down = expand)
 * or 'horizontal' for a side column to the right (arrow right = collapse, left = expand).
 */
export default function CollapseHandle({ axis = 'vertical', collapsed, onToggle, label, summary = null }) {
  const direction = axis === 'vertical'
    ? (collapsed ? 'down' : 'up')
    : (collapsed ? 'left' : 'right')

  return (
    <div className={`ch-wrap ch-${axis} ${collapsed ? 'is-collapsed' : ''}`}>
      <button
        type="button"
        className="ch-btn"
        onClick={onToggle}
        aria-expanded={!collapsed}
        aria-label={collapsed ? `Show ${label}` : `Hide ${label}`}
        title={collapsed ? `Show ${label}` : `Hide ${label}`}
      >
        <svg className={`ch-icon ch-${direction}`} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 6l4 4 4-4" />
        </svg>
      </button>
      {collapsed && summary && <div className="ch-summary">{summary}</div>}
    </div>
  )
}

/** Compact "label value" pair for a collapsed summary. */
export function HandleChip({ label, value, color }) {
  return (
    <span className="ch-chip">
      <span className="ch-chip-label">{label}</span>
      <span className="ch-chip-value" style={color ? { color } : undefined}>{value}</span>
    </span>
  )
}
