import './ResourceCards.css'

const ICONS = {
  cpu:  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="3" width="10" height="10" rx="1"/><path d="M6 1v2M10 1v2M6 13v2M10 13v2M1 6h2M1 10h2M13 6h2M13 10h2"/></svg>,
  gpu:  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="1" y="4" width="14" height="8" rx="1.5"/><path d="M4 4V2M8 4V2M12 4V2M4 12v2M8 12v2M12 12v2"/></svg>,
  ram:  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="1" y="5" width="14" height="6" rx="1"/><path d="M4 5V3M8 5V3M12 5V3M4 11v2M8 11v2M12 11v2"/></svg>,
  vram: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="1" y="5" width="14" height="6" rx="1"/><path d="M5 8h6M5 8l2-2M5 8l2 2"/></svg>,
  disk: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><ellipse cx="8" cy="5" rx="6" ry="2.5"/><path d="M2 5v6c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5V5"/><path d="M2 8c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5"/></svg>,
}

function ResCard({ icon, label, value, unit, sub, color, pct }) {
  return (
    <div className="rc">
      <div className="rc-icon" style={{ color }}>{icon}</div>
      <div className="rc-body">
        <div className="rc-top">
          <span className="rc-label">{label}</span>
          <span className="rc-val" style={{ color }}>{value}<span className="rc-unit">{unit}</span></span>
        </div>
        <div className="rc-track">
          <div className="rc-fill" style={{ width: `${Math.min(pct,100)}%`, background: color }} />
        </div>
        {sub && <div className="rc-sub">{sub}</div>}
      </div>
    </div>
  )
}

export default function ResourceCards({ res }) {
  return (
    <div className="resource-cards">
      <ResCard icon={ICONS.cpu}  label="CPU"  value={res.cpu_pct}  unit="%" pct={res.cpu_pct}  color="var(--blue)" />
      <ResCard icon={ICONS.gpu}  label="GPU"  value={res.gpu_pct}  unit="%" pct={res.gpu_pct}  color="var(--green)" sub="RTX 4060" />
      <ResCard icon={ICONS.ram}  label="RAM"  value={res.ram_pct}  unit="%" pct={res.ram_pct}  color="var(--amber)" sub={`${res.ram_used_gb}/${res.ram_total_gb}GB`} />
      <ResCard icon={ICONS.vram} label="VRAM" value={res.vram_pct} unit="%" pct={res.vram_pct} color="var(--blue)"  sub={`${res.vram_used_mb}MB`} />
      <ResCard icon={ICONS.disk} label="Disk" value={res.disk_pct} unit="%" pct={res.disk_pct} color={res.disk_pct > 85 ? 'var(--red)' : 'var(--amber)'} />
    </div>
  )
}
