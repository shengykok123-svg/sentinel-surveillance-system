import { useState } from 'react'
import ActivityHeatmap from '../components/charts/ActivityHeatmap'
import ActivityTimeline, { WATCHLIST_COLOR } from '../components/charts/ActivityTimeline'
import ChartCard, { DataTable } from '../components/charts/ChartCard'
import FaceSizeChart from '../components/charts/FaceSizeChart'
import PersonLeaderboard from '../components/charts/PersonLeaderboard'
import SimilarityHistogram, { ACCEPTED_COLOR, REJECTED_COLOR } from '../components/charts/SimilarityHistogram'
import { bucketFor, useAnalytics } from '../hooks/useAnalytics'
import './Analytics.css'

const DAY_MS = 24 * 60 * 60 * 1000

const RANGES = [
  { id: '24h', label: 'Last 24h' },
  { id: '7d', label: 'Last 7 days' },
  { id: '30d', label: 'Last 30 days' },
  { id: 'all', label: 'All time' },
]

// Whether a rising value is good, so a delta can be coloured by meaning rather
// than by sign. Volume is neither good nor bad, so it stays neutral.
const KPI_DIRECTION = {
  recognition_rate: 'up',
  avg_similarity: 'up',
  unknown: 'down',
  watchlist_hits: null,
  total_events: null,
  recognized: null,
}

const percent = (value) => (value == null ? '--' : `${(value * 100).toFixed(1)}%`)
const score = (value) => (value == null ? '--' : value.toFixed(3))
const count = (value) => (value == null ? '--' : value.toLocaleString())

function formatDelta(value) {
  if (value == null) return null
  const rounded = Math.round(value * 100)
  return `${rounded > 0 ? '+' : ''}${rounded}%`
}

function deltaTone(key, value) {
  const direction = KPI_DIRECTION[key]
  if (value == null || direction == null || Math.round(value * 100) === 0) return 'neutral'
  const rising = value > 0
  return (direction === 'up') === rising ? 'good' : 'bad'
}

function StatTile({ label, value, deltaKey, delta }) {
  const text = formatDelta(delta)
  return (
    <div className="kpi-tile">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {text && <div className={`kpi-delta ${deltaTone(deltaKey, delta)}`}>{text} vs previous</div>}
    </div>
  )
}

function KpiRow({ summary }) {
  const deltas = summary.delta_vs_previous ?? {}
  return (
    <div className="kpi-row">
      {/* The recognition rate is the number this page exists to report, so it
          leads as the hero figure and the rest support it. */}
      <div className="kpi-hero">
        <div className="kpi-label">Recognition rate</div>
        <div className="kpi-hero-value">{percent(summary.recognition_rate)}</div>
        <div className="kpi-hero-sub">
          {count(summary.recognized)} matched of {count(summary.total_events)} detections
        </div>
        {formatDelta(deltas.recognition_rate) && (
          <div className={`kpi-delta ${deltaTone('recognition_rate', deltas.recognition_rate)}`}>
            {formatDelta(deltas.recognition_rate)} vs previous
          </div>
        )}
      </div>

      <div className="kpi-tiles">
        <StatTile label="Detections" value={count(summary.total_events)} deltaKey="total_events" delta={deltas.total_events} />
        <StatTile label="Unknown" value={count(summary.unknown)} deltaKey="unknown" delta={deltas.unknown} />
        <StatTile label="Mean similarity" value={score(summary.avg_similarity)} deltaKey="avg_similarity" delta={deltas.avg_similarity} />
        <StatTile label="Watchlist hits" value={count(summary.watchlist_hits)} deltaKey="watchlist_hits" delta={deltas.watchlist_hits} />
        <StatTile label="People seen" value={count(summary.active_persons)} />
        <StatTile label="Active cameras" value={count(summary.active_cameras)} />
      </div>
    </div>
  )
}

/** Exact re-partition of the binned scores at a candidate threshold. */
function simulationEffect(bins, threshold, simulated) {
  if (simulated == null || simulated === threshold) return null
  const loosening = simulated < threshold
  const [low, high] = loosening ? [simulated, threshold] : [threshold, simulated]
  const affected = bins
    .filter((bin) => bin.lower >= low && bin.lower < high)
    .reduce((sum, bin) => sum + (loosening ? bin.rejected : bin.accepted), 0)
  return { loosening, affected }
}

function CompliancePanel({ people }) {
  const now = Date.now()
  const enrolled = people.length
  const embeddingOnly = people.filter((person) => person.embedding_only_mode).length
  const inactive = people.filter((person) => !person.is_active).length
  const withRetention = people.filter((person) => person.retention_expires_at)
  const expired = withRetention.filter((person) => new Date(person.retention_expires_at) <= now)
  const expiringSoon = withRetention
    .filter((person) => {
      const due = new Date(person.retention_expires_at).getTime()
      return due > now && due - now <= 30 * DAY_MS
    })
    .sort((a, b) => new Date(a.retention_expires_at) - new Date(b.retention_expires_at))

  const rows = [
    { label: 'Enrolled', value: count(enrolled) },
    { label: 'Embeddings only', value: `${count(embeddingOnly)} · ${enrolled ? Math.round((embeddingOnly / enrolled) * 100) : 0}%` },
    { label: 'Retention set', value: count(withRetention.length) },
    { label: 'Expiring in 30 days', value: count(expiringSoon.length), warn: expiringSoon.length > 0 },
    { label: 'Past retention', value: count(expired.length), warn: expired.length > 0 },
    { label: 'Deactivated', value: count(inactive) },
  ]

  return (
    <div className="cmp">
      <dl className="cmp-stats">
        {rows.map((row) => (
          <div key={row.label} className="cmp-stat">
            <dt className="cmp-stat-label">{row.label}</dt>
            <dd className={`cmp-stat-value ${row.warn ? 'warn' : ''}`}>{row.value}</dd>
          </div>
        ))}
      </dl>

      {expiringSoon.length > 0 && (
        <div className="cmp-due">
          <div className="cmp-due-title">Next retention deadlines</div>
          <ul className="cmp-due-list">
            {expiringSoon.slice(0, 4).map((person) => (
              <li key={person.id} className="cmp-due-row">
                <span className="cmp-due-name">{person.name}</span>
                <span className="cmp-due-date">
                  {new Date(person.retention_expires_at).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {expired.length > 0 && (
        <p className="an-note">
          <strong>{count(expired.length)}</strong> enrolment
          {expired.length === 1 ? ' has' : 's have'} passed their retention deadline and should be
          reviewed for deletion.
        </p>
      )}
    </div>
  )
}

function CameraTable({ cameras }) {
  const busiest = Math.max(...cameras.map((camera) => camera.total_events), 1)

  return (
    <div className="cam-table-wrap">
      <table className="cam-table">
        <thead>
          <tr>
            <th scope="col">Camera</th>
            <th scope="col">Detections</th>
            <th scope="col" className="num">Recognized</th>
            <th scope="col" className="num">Rate</th>
            <th scope="col" className="num">Mean similarity</th>
            <th scope="col" className="num">Mean face size</th>
            <th scope="col" className="num">Last event</th>
          </tr>
        </thead>
        <tbody>
          {cameras.map((camera) => (
            <tr key={camera.camera_id} className={camera.total_events ? '' : 'idle'}>
              <th scope="row">
                <span className="cam-name">{camera.camera_name}</span>
                {camera.location && <span className="cam-location">{camera.location}</span>}
              </th>
              <td>
                <div className="cam-volume">
                  {camera.total_events > 0 && (
                    <div
                      className="cam-volume-bar"
                      style={{ width: `${(camera.total_events / busiest) * 100}%` }}
                    />
                  )}
                  <span className="cam-volume-value">{count(camera.total_events)}</span>
                </div>
              </td>
              <td className="num">{count(camera.recognized)}</td>
              <td className="num">{percent(camera.recognition_rate)}</td>
              <td className="num">{score(camera.avg_similarity)}</td>
              <td className="num">
                {camera.avg_face_size == null ? '--' : `${Math.round(camera.avg_face_size)}px`}
              </td>
              <td className="num">
                {camera.last_event_at ? new Date(camera.last_event_at).toLocaleString() : '--'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Analytics({ onNavigate }) {
  const [range, setRange] = useState('7d')
  const [simulated, setSimulated] = useState(null)
  const { data, stale, error, refresh } = useAnalytics(range)

  const changeRange = (next) => {
    setSimulated(null)
    setRange(next)
  }

  if (error) {
    return (
      <div className="an-page">
        <div className="an-empty">
          <div className="an-empty-title">Analytics unavailable</div>
          <div className="an-empty-sub">{error.message}</div>
          <button type="button" className="an-retry" onClick={refresh}>Retry</button>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="an-page">
        <div className="an-empty">
          <div className="an-empty-title">Loading recognition analytics</div>
        </div>
      </div>
    )
  }

  const { summary, timeline, histogram, faceSize, cameras, persons, people } = data
  const effect = simulationEffect(histogram.bins, histogram.threshold, simulated)
  const hasScores = histogram.total_scored > 0

  return (
    <div className="an-page">
      <header className="an-header">
        <div>
          <h1 className="an-title">Recognition Analytics</h1>
          <p className="an-subtitle">
            {summary.window.since
              ? `${new Date(summary.window.since).toLocaleString()} — now`
              : 'Full retained history'}
          </p>
        </div>

        {/* One filter row scoping everything below it, so every panel on the
            page always reports the same slice. */}
        <div className="an-ranges" role="group" aria-label="Reporting range">
          {RANGES.map((option) => (
            <button
              key={option.id}
              type="button"
              className={`an-range ${range === option.id ? 'active' : ''}`}
              aria-pressed={range === option.id}
              onClick={() => changeRange(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </header>

      <div className={`an-body ${stale ? 'stale' : ''}`}>
        <KpiRow summary={summary} />

        <ChartCard
          title="Recognition activity"
          subtitle={`Detections per ${bucketFor(range)}`}
          series={[
            { label: 'Recognized', color: ACCEPTED_COLOR },
            { label: 'Unknown', color: REJECTED_COLOR },
            { label: 'Watchlist hit', color: WATCHLIST_COLOR },
          ]}
          table={
            <DataTable
              caption="Detections per bucket"
              columns={['Bucket', 'Recognized', 'Unknown', 'Watchlist']}
              rows={timeline.buckets.map((slot) => [
                new Date(slot.ts).toLocaleString(),
                slot.recognized,
                slot.unknown,
                slot.watchlist,
              ])}
            />
          }
        >
          <ActivityTimeline buckets={timeline.buckets} bucket={timeline.bucket} />
        </ChartCard>

        <div className="an-grid">
          <ChartCard
            title="Similarity distribution"
            subtitle={
              hasScores
                ? `${count(histogram.total_scored)} scored detections · threshold ${histogram.threshold.toFixed(2)}`
                : 'No scored detections in this window'
            }
            series={[
              { label: 'Above threshold', color: ACCEPTED_COLOR },
              { label: 'Below threshold', color: REJECTED_COLOR },
            ]}
            table={
              <DataTable
                caption="Match scores by similarity bin"
                columns={['Similarity', 'Above threshold', 'Below threshold']}
                rows={histogram.bins.map((bin) => [
                  `${bin.lower.toFixed(2)} – ${bin.upper.toFixed(2)}`,
                  bin.accepted,
                  bin.rejected,
                ])}
              />
            }
          >
            {hasScores ? (
              <>
                <SimilarityHistogram
                  bins={histogram.bins}
                  threshold={histogram.threshold}
                  simulated={simulated}
                  onSimulate={setSimulated}
                />
                {effect && (
                  <p className="an-note">
                    At {simulated.toFixed(2)}, <strong>{count(effect.affected)}</strong>{' '}
                    {effect.loosening
                      ? 'currently rejected detections would be accepted.'
                      : 'currently accepted detections would be rejected.'}
                  </p>
                )}
              </>
            ) : (
              <p className="an-note">Scores appear here once detections are recorded in this window.</p>
            )}
          </ChartCard>

          <ChartCard
            title="Recognition by face size"
            subtitle="Detections grouped by the equivalent square edge of the detected face"
            series={[
              { label: 'Recognized', color: ACCEPTED_COLOR },
              { label: 'Unrecognized', color: REJECTED_COLOR },
            ]}
            table={
              <DataTable
                caption="Recognition outcome by face size band"
                columns={['Face size', 'Detections', 'Recognized', 'Rate', 'Mean similarity']}
                rows={faceSize.bins.map((bin) => [
                  bin.label,
                  bin.samples,
                  bin.recognized,
                  percent(bin.recognition_rate),
                  score(bin.avg_similarity),
                ])}
              />
            }
          >
            <FaceSizeChart bins={faceSize.bins} />
          </ChartCard>
        </div>

        <ChartCard
            title="Activity by weekday and hour"
            subtitle="Local time; bucketed against the browser's timezone rather than UTC"
            table={
              <DataTable
                caption="Detections by weekday and hour"
                columns={['Weekday', 'Hour', 'Detections']}
                rows={timeline.heatmap
                  .filter((cell) => cell.count > 0)
                  .map((cell) => [
                    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][cell.weekday],
                    `${String(cell.hour).padStart(2, '0')}:00`,
                    cell.count,
                  ])}
              />
            }
          >
            <ActivityHeatmap cells={timeline.heatmap} />
        </ChartCard>

        <ChartCard
          title="People"
          subtitle="Most frequently recognised in this window"
          table={
            <DataTable
              caption="Recognitions per person"
              columns={['Person', 'Type', 'Recognitions', 'Mean similarity', 'Cameras', 'Last seen']}
              rows={persons.persons.map((person) => [
                person.person_name ?? 'Deleted person',
                person.person_type ?? 'unknown',
                person.events,
                score(person.avg_similarity),
                person.cameras.join(', ') || '--',
                person.last_seen ? new Date(person.last_seen).toLocaleString() : '--',
              ])}
            />
          }
        >
          <PersonLeaderboard
            persons={persons.persons}
            breakdown={persons.type_breakdown}
            onSelectPerson={() => onNavigate?.('search')}
          />
        </ChartCard>

        <ChartCard title="Cameras" subtitle="Busiest first; cameras with no detections are kept as zero rows">
          <CameraTable cameras={cameras.cameras} />
        </ChartCard>

        <ChartCard title="Data retention" subtitle="Enrolment records currently on file">
          <CompliancePanel people={people} />
        </ChartCard>
      </div>
    </div>
  )
}
