import { useState } from 'react'
import { useResources } from '../hooks/useResources'
import './ResourceAnalytics.css'

const METRICS = [
  { key: 'cpu_pct',       label: 'CPU',     unit: '%',     color: '#2563eb' },
  { key: 'gpu_pct',       label: 'GPU',     unit: '%',     color: '#16a34a' },
  { key: 'ram_pct',       label: 'RAM',     unit: '%',     color: '#d97706' },
  { key: 'vram_pct',      label: 'VRAM',    unit: '%',     color: '#7c3aed' },
  { key: 'disk_pct',      label: 'Disk',    unit: '%',     color: '#dc2626' },
]

/* ── Mini sparkline (SVG) ── */
function Sparkline({ data, color, height = 50 }) {
  if (data.length < 2) return <div style={{ height }} />
  const W = 100
  const max = Math.max(...data, 1)
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W
    const y = height - (v / max) * height
    return `${x},${y}`
  }).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none"
      style={{ width: '100%', height, display: 'block' }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <polyline
        points={`0,${height} ${pts} ${W},${height}`}
        fill={color}
        fillOpacity="0.08"
        stroke="none"
      />
    </svg>
  )
}

/* ── Bar chart ── */
function BarChart({ history, metricKey, color, unit }) {
  const slice  = history.slice(-30)
  const values = slice.map(h => h[metricKey] ?? 0)
  const max    = Math.max(...values, 1)
  const labels = slice.filter((_, i) => i % 5 === 0).map(h => h.ts?.slice(0, 5) ?? '')

  return (
    <div className="bar-chart-wrap">
      <div className="bar-chart-bars">
        {values.map((v, i) => (
          <div key={i} className="bar-col">
            <div
              className="bar-fill"
              style={{
                height: `${Math.round((v / max) * 100)}%`,
                background: color,
                opacity: i === values.length - 1 ? 1 : 0.55,
              }}
              title={`${v}${unit}`}
            />
          </div>
        ))}
      </div>
      <div className="bar-chart-labels">
        {slice.map((h, i) => (
          <div key={i} className="bar-label">
            {i % 5 === 0 ? (h.ts?.slice(0, 5) ?? '') : ''}
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Single metric panel ── */
function MetricPanel({ metric, current, history }) {
  const values  = history.map(h => h[metric.key] ?? 0)
  const val     = current[metric.key] ?? 0
  const avg     = values.length
    ? Math.round(values.reduce((a, b) => a + b, 0) / values.length * 10) / 10
    : 0
  const peak    = values.length ? Math.max(...values) : 0

  return (
    <div className="metric-panel">
      <div className="mp-header">
        <div className="mp-title">
          <div className="mp-dot" style={{ background: metric.color }} />
          {metric.label}
        </div>
        <div className="mp-current" style={{ color: metric.color }}>
          {val}{metric.unit}
        </div>
      </div>

      <div className="mp-spark">
        <Sparkline data={values} color={metric.color} height={52} />
      </div>

      <div className="mp-bar-section">
        <div className="mp-section-label">Last 30 seconds</div>
        <BarChart
          history={history}
          metricKey={metric.key}
          color={metric.color}
          unit={metric.unit}
        />
      </div>

      <div className="mp-stats">
        <div className="mp-stat">
          <div className="mp-stat-label">Avg</div>
          <div className="mp-stat-val">{avg}{metric.unit}</div>
        </div>
        <div className="mp-stat">
          <div className="mp-stat-label">Peak</div>
          <div className="mp-stat-val" style={{ color: metric.color }}>{peak}{metric.unit}</div>
        </div>
        <div className="mp-stat">
          <div className="mp-stat-label">Samples</div>
          <div className="mp-stat-val">{values.length}</div>
        </div>
      </div>
    </div>
  )
}

/* ── Page ── */
export default function ResourceAnalytics() {
  const { current, history } = useResources()
  const [selected, setSelected] = useState('cpu_pct')
  const active = METRICS.find(m => m.key === selected) ?? METRICS[0]

  return (
    <div className="ra-page">
      <div className="ra-header">
        <div className="ra-title">Resource Analytics</div>
        <div className="ra-subtitle">
          System performance · {history.length} samples collected
        </div>
      </div>

      {/* Tab selector */}
      <div className="ra-tabs">
        {METRICS.map(m => (
          <div
            key={m.key}
            className={`ra-tab ${selected === m.key ? 'active' : ''}`}
            style={selected === m.key ? { borderBottomColor: m.color, color: m.color } : {}}
            onClick={() => setSelected(m.key)}
          >
            <div className="ra-tab-dot" style={{ background: m.color }} />
            {m.label}
            <span className="ra-tab-val">
              {current[m.key] ?? 0}{m.unit}
            </span>
          </div>
        ))}
      </div>

      {/* Detail panel for selected metric */}
      <div className="ra-detail">
        <MetricPanel
          metric={active}
          current={current}
          history={history}
        />
      </div>

      {/* Overview grid — all 6 metrics */}
      <div className="ra-grid">
        {METRICS.map(m => {
          const values = history.map(h => h[m.key] ?? 0)
          const val    = current[m.key] ?? 0
          return (
            <div
              key={m.key}
              className={`ra-card ${selected === m.key ? 'ra-card-active' : ''}`}
              onClick={() => setSelected(m.key)}
            >
              <div className="ra-card-header">
                <span className="ra-card-label">{m.label}</span>
                <span className="ra-card-val" style={{ color: m.color }}>
                  {val}{m.unit}
                </span>
              </div>
              <Sparkline data={values} color={m.color} height={36} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
