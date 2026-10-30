import './ToastContainer.css'

const ICONS = {
  alert: (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#dc2626" strokeWidth="2">
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 5v3.5M8 10.5v.5" strokeLinecap="round" />
    </svg>
  ),
  warn: (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#d97706" strokeWidth="2">
      <path d="M8 2L14 13H2L8 2z" />
      <path d="M8 6v3.5M8 11v.5" strokeLinecap="round" />
    </svg>
  ),
  ok: (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#16a34a" strokeWidth="2">
      <circle cx="8" cy="8" r="6.5" />
      <path d="M4.5 8l2.5 2.5L11.5 5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
}

const COLORS = { alert: '#dc2626', warn: '#d97706', ok: '#16a34a' }
const PERSON_TYPE_COLORS = {
  watchlist: '#dc2626',
  vip: '#d97706',
  staff: '#16a34a',
  visitor: '#2563eb',
}

function formatPersonType(personType) {
  const value = String(personType ?? '').trim()
  if (!value) {
    return 'POI'
  }
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function getToastTitle(toast) {
  if (toast.severity === 'alert' && toast.person_name) {
    return `${formatPersonType(toast.person_type)} recognized: ${toast.person_name}`
  }
  return toast.type
}

function getToastAccent(toast) {
  if (toast.severity === 'alert') {
    const personType = String(toast.person_type ?? '').toLowerCase()
    return PERSON_TYPE_COLORS[personType] ?? COLORS.alert
  }
  return COLORS[toast.severity] ?? '#94a3b8'
}

function getToastSub(toast) {
  const parts = [toast.camera, toast.time]
  if (toast.track_id) {
    parts.push(`Track ${toast.track_id}`)
  }
  return parts.join(' | ')
}

export default function ToastContainer({ toasts, onDismiss }) {
  return (
    <div className="toast-stack">
      {toasts.map((toast) => {
        const accent = getToastAccent(toast)

        return (
          <div key={toast.id} className={`toast toast-${toast.severity}`} style={{ '--toast-accent': accent }}>
            <div className="toast-ic" style={{ background: `${accent}20` }}>
              {ICONS[toast.severity]}
            </div>
            <div className="toast-body">
              <div className="toast-title">{getToastTitle(toast)}</div>
              <div className="toast-sub">{getToastSub(toast)}</div>
            </div>
            <button className="toast-x" onClick={() => onDismiss(toast.id)} aria-label="Dismiss alert">
              x
            </button>
          </div>
        )
      })}
    </div>
  )
}
