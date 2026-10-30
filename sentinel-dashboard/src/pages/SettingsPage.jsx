import { useEffect, useState } from 'react'
import { getCameras, createCamera, updateCamera, deleteCamera, getSettings, saveRecognitionSettings, changePassword } from '../utils/api'
import { playAlertSound } from '../hooks/usePreferences'
import './SettingsPage.css'

const blankCamera = () => ({ id: '', name: '', source_type: 'rtsp', source_uri: '', is_active: true })
export default function SettingsPage({ user, preferences, savePreferences, onRefresh }) {
  const [cameras, setCameras] = useState([])
  const [camera, setCamera] = useState(blankCamera)
  const [editing, setEditing] = useState(false)
  const [config, setConfig] = useState(null)
  const [defaults, setDefaults] = useState(null)
  const [alerts, setAlerts] = useState(preferences)
  const [password, setPassword] = useState({ current_password: '', new_password: '', confirm: '' })
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState(null)
  const [loading, setLoading] = useState(true)
  const admin = user.roles?.some(role => role.toLowerCase() === 'admin')
  async function load() {
    setLoading(true)
    try {
      const [items, settings] = await Promise.all([getCameras(), getSettings()])
      setCameras(items); setConfig(settings.recognition); setDefaults(settings.defaults)
    } catch (error) { setNotice({ error: true, text: error.message }) }
    finally { setLoading(false) }
  }
  useEffect(() => { load() }, [])
  async function run(section, action, message) {
    setBusy(section); setNotice(null)
    try { await action(); setNotice({ text: message }) }
    catch (error) { setNotice({ error: true, text: error.message }) }
    finally { setBusy('') }
  }
  function submitCamera(event) {
    event.preventDefault()
    run('camera', async () => {
      const payload = { ...camera, name: camera.name.trim(), source_uri: camera.source_uri.trim() }
      if (!payload.name || !payload.source_uri) throw new Error('Camera name and source are required.')
      if (payload.source_type === 'webcam' && !/^\d+$/.test(payload.source_uri)) throw new Error('Webcam source must be a non-negative device index.')
      if (payload.source_type === 'rtsp' && !/^rtsps?:\/\//i.test(payload.source_uri)) throw new Error('Enter an rtsp:// or rtsps:// URL.')
      if (editing) { const { id, ...changes } = payload; await updateCamera(id, changes) }
      else await createCamera({ ...payload, id: crypto.randomUUID() })
      setCameras(await getCameras()); setCamera(blankCamera()); setEditing(false); onRefresh()
    }, 'Camera saved. Enabled streams reconnect automatically.')
  }
  function removeCamera() {
    if (!window.confirm(`Delete camera "${camera.name}"? Its stream stops immediately; recorded events are kept.`)) return
    run('camera', async () => {
      await deleteCamera(camera.id)
      setCameras(await getCameras()); setCamera(blankCamera()); setEditing(false); onRefresh()
    }, 'Camera deleted.')
  }
  return <div className="settings-page">
    <header><div className="settings-eyebrow">SYSTEM / PREFERENCES</div><h1>Settings</h1><p>Manage camera sources, recognition and your operator preferences.</p></header>
    {notice && <div className={`settings-notice ${notice.error ? 'error' : ''}`} role={notice.error ? 'alert' : 'status'}>{notice.text}</div>}
    {loading && <p role="status">Loading settings…</p>}
    {!loading && !config && <button onClick={load}>Retry loading</button>}
    <div className="settings-grid">
      <section className="settings-card settings-wide"><h2>Camera management</h2><p>Changes are stored on the server. Updating a source briefly reconnects its stream.</p>
        <div className="settings-camera-list">{cameras.map(item => <div className="settings-camera" key={item.id}><div><strong>{item.name}</strong><small>{item.source_type.toUpperCase()} · {item.is_active ? (item.online ? 'Online' : 'Connecting / offline') : 'Disabled'}</small></div><button disabled={!!busy} onClick={() => { setCamera({ id: item.id, name: item.name, source_type: item.source_type, source_uri: item.source_uri, is_active: item.is_active }); setEditing(true) }}>Edit</button></div>)}{!loading && !cameras.length && <p>No cameras registered.</p>}</div>
        <form onSubmit={submitCamera}><fieldset disabled={!!busy || loading}><legend>{editing ? 'Edit camera' : 'Add camera'}</legend><div className="settings-fields">
          <label>Name<input required maxLength={255} value={camera.name} onChange={e => setCamera({ ...camera, name: e.target.value })} /></label>
          <label>Source type<select value={camera.source_type} onChange={e => setCamera({ ...camera, source_type: e.target.value })}><option value="rtsp">RTSP stream</option><option value="file">Video file</option><option value="webcam">Webcam</option></select></label>
          <label className="settings-wide">{camera.source_type === 'file' ? 'Video path on backend computer' : camera.source_type === 'webcam' ? 'Device index (e.g. 0)' : 'RTSP URL'}<input required value={camera.source_uri} onChange={e => setCamera({ ...camera, source_uri: e.target.value })} /></label>
        </div><label className="settings-check"><input type="checkbox" checked={camera.is_active} onChange={e => setCamera({ ...camera, is_active: e.target.checked })} />Enable camera</label><div className="settings-actions"><button className="primary" type="submit">{busy === 'camera' ? 'Saving…' : editing ? 'Save camera' : 'Add camera'}</button>{editing && <button type="button" onClick={() => { setEditing(false); setCamera(blankCamera()) }}>Cancel</button>}{editing && <button type="button" className="danger" onClick={removeCamera}>Delete camera</button>}</div></fieldset></form>
      </section>
      <section className="settings-card"><h2>Recognition parameters</h2><p>Applies to subsequent recognition decisions immediately. Existing confirmed tracks are re-evaluated on their normal cycle.</p>{!admin && <p>Only administrators can change these values.</p>}
        {config && <form onSubmit={e => { e.preventDefault(); run('recognition', async () => setConfig(await saveRecognitionSettings(config)), 'Recognition parameters saved.') }}><fieldset disabled={!admin || !!busy}>
          <label>Similarity threshold (0–1)<input type="number" min="0" max="1" step="0.01" required value={config.recognition_threshold} onChange={e => setConfig({ ...config, recognition_threshold: e.target.value === '' ? '' : Number(e.target.value) })} /></label><p>Higher values require a closer match and may miss more correct matches. Default: {defaults.recognition_threshold}.</p>
          <label>Minimum live face quality (0–10)<input type="number" min="0" max="10" step="any" required value={config.live_quality_threshold} onChange={e => setConfig({ ...config, live_quality_threshold: e.target.value === '' ? '' : Number(e.target.value) })} /></label><p>CR-FIQA score, not a percentage. Higher values reject more low-quality faces. Default: {defaults.live_quality_threshold}. Enrollment quality is configured separately.</p>
          <div className="settings-actions"><button className="primary">{busy === 'recognition' ? 'Saving…' : 'Save parameters'}</button><button type="button" onClick={() => setConfig({ ...defaults })}>Use defaults</button></div>
        </fieldset></form>}
      </section>
      <section className="settings-card"><h2>Alert preferences</h2><p>Saved for this account in this browser. These controls affect pop-up alerts and sounds; event recording continues.</p>
        <form onSubmit={e => { e.preventDefault(); run('alerts', () => savePreferences({ ...preferences, notifications: alerts.notifications, sound: alerts.sound, cooldown: Number(alerts.cooldown) }), 'Alert preferences saved in this browser.') }}><fieldset disabled={!!busy}>
          <label className="settings-check"><input type="checkbox" checked={alerts.notifications} onChange={e => setAlerts({ ...alerts, notifications: e.target.checked })} />Show pop-up notifications</label>
          <label className="settings-check"><input type="checkbox" checked={alerts.sound} onChange={e => { setAlerts({ ...alerts, sound: e.target.checked }); if (e.target.checked) playAlertSound().catch(() => {}) }} />Play alert sound</label>
          <label>Repeat alert cooldown (seconds)<input type="number" min="0" max="3600" step="1" required value={alerts.cooldown} onChange={e => setAlerts({ ...alerts, cooldown: e.target.value })} /></label><p>Suppresses repeated alerts for the same person on the same camera. Sound may require a click in the browser first.</p>
          <div className="settings-actions"><button className="primary">Save alerts</button><button type="button" onClick={() => run('sound', playAlertSound, 'Test sound played.')}>Test sound</button></div>
        </fieldset></form>
      </section>
      <section className="settings-card"><h2>Account & password</h2><dl><dt>Username</dt><dd>{user.username}</dd><dt>Email</dt><dd>{user.email || 'Not provided'}</dd><dt>Roles</dt><dd>{user.roles?.join(', ') || 'No role assigned'}</dd></dl>
        <form onSubmit={e => { e.preventDefault(); run('password', async () => { if (password.new_password !== password.confirm) throw new Error('New passwords do not match.'); await changePassword({ current_password: password.current_password, new_password: password.new_password }); setPassword({ current_password: '', new_password: '', confirm: '' }) }, 'Password updated successfully.') }}><fieldset disabled={!!busy}>
          <label>Current password<input type="password" autoComplete="current-password" required value={password.current_password} onChange={e => setPassword({ ...password, current_password: e.target.value })} /></label>
          <label>New password<input type="password" autoComplete="new-password" required minLength={8} maxLength={128} value={password.new_password} onChange={e => setPassword({ ...password, new_password: e.target.value })} /></label>
          <label>Confirm new password<input type="password" autoComplete="new-password" required minLength={8} maxLength={128} value={password.confirm} onChange={e => setPassword({ ...password, confirm: e.target.value })} /></label><p>Use 8–128 characters.</p><button className="primary">{busy === 'password' ? 'Updating…' : 'Update password'}</button>
        </fieldset></form>
      </section>
      <section className="settings-card"><h2>Appearance</h2><p>Your theme applies immediately and is remembered for this account in this browser.</p><label>Theme<select aria-label="Theme" value={preferences.theme} onChange={e => run('theme', () => savePreferences({ ...preferences, theme: e.target.value }), 'Theme saved.')} disabled={!!busy}><option value="dark">Dark</option><option value="light">Light</option></select></label></section>
    </div>
  </div>
}
