import { useEffect, useState } from 'react'
import CameraGrid from './components/CameraGrid'
import CollapseHandle, { HandleChip } from './components/CollapseHandle'
import EventLog from './components/EventLog'
import MetricsRow from './components/MetricsRow'
import POIPanel from './components/POIPanel'
import ResourceCards from './components/ResourceCards'
import Sidebar from './components/Sidebar'
import ToastContainer from './components/ToastContainer'
import Topbar from './components/Topbar'
import { useAuth } from './context/AuthContext'
import { useCollapsed } from './hooks/useCollapsed'
import { useResources } from './hooks/useResources'
import { useSystemStats } from './hooks/useSystemStats'
import { useToast } from './hooks/useToast'
import Analytics from './pages/Analytics'
import EnrollmentPage from './pages/EnrollmentPage'
import FaceSearch from './pages/FaceSearch'
import LoginPage from './pages/LoginPage'
import ResourceAnalytics from './pages/ResourceAnalytics'
import RecognitionRecords from './pages/RecognitionRecords'
import SettingsPage from './pages/SettingsPage'
import { usePreferences } from './hooks/usePreferences'
import './App.css'

export default function App() {
  const { initializing, isAuthenticated, user, logout } = useAuth()
  const [activePage, setActivePage] = useState('live')
  const [recordId, setRecordId] = useState(null)
  const [profileFocus, setProfileFocus] = useState(null)

  if (initializing) {
    return (
      <div className="app-loading">
        <div className="app-loading-card">
          <div className="app-loading-title">Restoring operator session</div>
          <div className="app-loading-sub">Checking bootstrap state and validating the current token.</div>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <LoginPage />
  }

  return (
    <AuthenticatedShell
      key={user.id}
      activePage={activePage}
      onNavigate={(page) => {
        // Plain navigation drops any pending profile focus so POI Profiles opens unfiltered.
        setProfileFocus(null)
        setActivePage(page)
      }}
      user={user}
      onLogout={logout}
      recordId={recordId}
      onOpenRecord={(eventId) => {
        setRecordId(eventId)
        setActivePage('records')
      }}
      profileFocus={profileFocus}
      onOpenProfile={(personId) => {
        // Fresh object each time so re-opening the same person still triggers the focus effect.
        setProfileFocus({ personId, at: Date.now() })
        setActivePage('search')
      }}
    />
  )
}

function AuthenticatedShell({ activePage, onNavigate, user, onLogout, recordId, onOpenRecord, profileFocus, onOpenProfile }) {
  const { metrics, events, cameras, people, activeTracks, refresh, clearEvents } = useSystemStats()
  const { current: res } = useResources()
  const [preferences, savePreferences] = usePreferences(user.id)
  const theme = preferences.theme
  const onToggleTheme = () => savePreferences({ ...preferences, theme: theme === 'dark' ? 'light' : 'dark' })
  useEffect(() => { document.documentElement.setAttribute('data-theme', theme) }, [theme])
  const { toasts, dismiss } = useToast(events, activeTracks, preferences)
  const [overviewCollapsed, toggleOverview] = useCollapsed('live.overview')
  const [sideCollapsed, toggleSide] = useCollapsed('live.side')

  return (
    <div className="shell">
      <Sidebar activePage={activePage} onNavigate={onNavigate} />

      <div className="main">
        <Topbar
          alertCount={metrics.alerts}
          theme={theme}
          onToggleTheme={onToggleTheme}
          cameraCount={metrics.camerasTotal}
          currentUser={user}
          onLogout={onLogout}
        />

        {activePage === 'live' && (
          <div className="content">
            {!overviewCollapsed && (
              <>
                <MetricsRow metrics={metrics} />
                <ResourceCards res={res} />
              </>
            )}
            <CollapseHandle
              axis="vertical"
              collapsed={overviewCollapsed}
              onToggle={toggleOverview}
              label="system overview"
              summary={(
                <>
                  <HandleChip label="Cam" value={`${metrics.camerasOnline}/${metrics.camerasTotal}`} color="var(--blue)" />
                  <HandleChip label="Faces" value={metrics.facesDetected} color="var(--green)" />
                  <HandleChip label="CPU" value={`${res.cpu_pct}%`} />
                  <HandleChip label="GPU" value={`${res.gpu_pct}%`} />
                  <HandleChip label="RAM" value={`${res.ram_pct}%`} />
                  <HandleChip label="VRAM" value={`${res.vram_pct}%`} />
                  <HandleChip label="Disk" value={`${res.disk_pct}%`} color={res.disk_pct > 85 ? 'var(--red)' : undefined} />
                </>
              )}
            />
            <div className={`live-body ${sideCollapsed ? 'side-collapsed' : ''}`}>
              <CameraGrid cameras={cameras} />
              <CollapseHandle
                axis="horizontal"
                collapsed={sideCollapsed}
                onToggle={toggleSide}
                label="POI and live events"
                summary={(
                  <>
                    <HandleChip label="POI" value={people.length} color="var(--blue)" />
                    <HandleChip label="Alerts" value={metrics.alerts} color={metrics.alerts > 0 ? 'var(--red)' : undefined} />
                  </>
                )}
              />
              {!sideCollapsed && (
                <div className="live-side">
                  <POIPanel people={people} onOpenProfile={onOpenProfile} />
                  <EventLog events={events} onClear={clearEvents} onOpenRecord={onOpenRecord} />
                </div>
              )}
            </div>
          </div>
        )}

        {activePage === 'search' && (
          <div className="content">
            <FaceSearch people={people} currentUser={user} onRefresh={refresh} onNavigate={onNavigate} focusPerson={profileFocus} />
          </div>
        )}
        {activePage === 'events' && <EventLog events={events} fullPage onClear={clearEvents} onOpenRecord={onOpenRecord} />}
        {activePage === 'records' && <RecognitionRecords focusEventId={recordId} />}
        {activePage === 'resources' && <ResourceAnalytics />}
        {activePage === 'enroll' && (
          <div className="content">
            <EnrollmentPage onEnroll={refresh} currentUser={user} />
          </div>
        )}
        {activePage === 'analytics' && <Analytics onNavigate={onNavigate} />}
        {activePage === 'settings' && <SettingsPage user={user} preferences={preferences} savePreferences={savePreferences} onRefresh={refresh} />}
      </div>

      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </div>
  )
}
