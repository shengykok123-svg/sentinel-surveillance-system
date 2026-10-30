import './MetricsRow.css'

const ICONS = {
  camera: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M1 5a2 2 0 012-2h8a2 2 0 012 2v6a2 2 0 01-2 2H3a2 2 0 01-2-2V5z"/><path d="M13 7l2-1.5v5L13 9"/></svg>,
  face: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="8" cy="6" r="3"/><path d="M2 14c0-3.3 2.7-6 6-6s6 2.7 6 6" strokeLinecap="round"/></svg>,
}

function MiniCard({ icon, value, label, color }) {
  return (
    <div className="mc">
      <div className="mc-icon" style={{ color }}>{icon}</div>
      <div className="mc-right">
        <div className="mc-val" style={{ color }}>{value}</div>
        <div className="mc-label">{label}</div>
      </div>
    </div>
  )
}

export default function MetricsRow({ metrics }) {
  return (
    <div className="metrics-row">
      <MiniCard
        icon={ICONS.camera}
        value={`${metrics.camerasOnline}/${metrics.camerasTotal}`}
        label="Cameras"
        color="var(--blue)"
      />
      <MiniCard
        icon={ICONS.face}
        value={metrics.facesDetected}
        label="Tracked Faces"
        color="var(--green)"
      />
    </div>
  )
}
