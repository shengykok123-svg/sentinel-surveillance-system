import { useState } from 'react'
import './ChartCard.css'

/**
 * Shell for one analytics chart.
 *
 * Carries the title, an always-present legend for multi-series charts, and a
 * table toggle. The table is not a nicety: it is the accessible twin of the
 * chart, so every value stays reachable without relying on colour or hover.
 */
export default function ChartCard({ title, subtitle, series = [], table, stale, children, actions }) {
  const [view, setView] = useState('chart')

  return (
    <section className={`chart-card ${stale ? 'stale' : ''}`}>
      <header className="cc-head">
        <div className="cc-titles">
          <h2 className="cc-title">{title}</h2>
          {subtitle && <p className="cc-subtitle">{subtitle}</p>}
        </div>

        <div className="cc-tools">
          {actions}
          {table && (
            <div className="cc-toggle" role="group" aria-label={`${title} view`}>
              {['chart', 'table'].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`cc-toggle-btn ${view === mode ? 'active' : ''}`}
                  aria-pressed={view === mode}
                  onClick={() => setView(mode)}
                >
                  {mode === 'chart' ? 'Chart' : 'Table'}
                </button>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* Two or more series always get a legend: identity must never depend on
          colour matching alone. */}
      {series.length > 1 && (
        <ul className="cc-legend">
          {series.map((item) => (
            <li key={item.label} className="cc-legend-item">
              <span className="cc-swatch" style={{ background: item.color }} aria-hidden="true" />
              {item.label}
            </li>
          ))}
        </ul>
      )}

      <div className="cc-body">{view === 'chart' ? children : table}</div>
    </section>
  )
}

/** Simple data table used as the accessible twin of a chart. */
export function DataTable({ columns, rows, caption }) {
  return (
    <div className="cc-table-wrap">
      <table className="cc-table">
        {caption && <caption className="cc-table-caption">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column} scope="col">{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, cellIndex) => (
                cellIndex === 0
                  ? <th key={cellIndex} scope="row">{cell}</th>
                  : <td key={cellIndex}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
