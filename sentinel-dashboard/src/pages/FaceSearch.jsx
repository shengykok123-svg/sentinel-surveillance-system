import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { addPOIReferences, deletePOI, generatePOIReferences, getPOIReferenceImage, getPOIReferences, requestSensitiveAccess, updatePOI } from '../utils/api'
import PhotoSearchPanel from '../components/PhotoSearchPanel'
import './FaceSearch.css'

const PERSON_TYPES = ['watchlist', 'vip', 'staff', 'visitor']
const STATUS_OPTIONS = ['all', 'active', 'inactive']

const EMPTY_EDIT_FORM = {
  name: '',
  person_type: 'watchlist',
  notes: '',
  is_active: true,
  embedding_only_mode: false,
  retention_expires_at: '',
}

function getInitials(name) {
  return String(name || '')
    .split(' ')
    .filter(Boolean)
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase() || 'PO'
}

function formatDateTime(value) {
  if (!value) {
    return 'Not available'
  }
  return new Date(value).toLocaleString()
}

function formatDateTimeLocalValue(value) {
  if (!value) {
    return ''
  }
  const date = new Date(value)
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000)
  return local.toISOString().slice(0, 16)
}

function buildEditForm(person) {
  return {
    name: person?.name ?? '',
    person_type: person?.person_type ?? person?.metadata?.person_type ?? 'watchlist',
    notes: person?.notes ?? '',
    is_active: person?.is_active ?? true,
    embedding_only_mode: person?.embedding_only_mode ?? false,
    retention_expires_at: formatDateTimeLocalValue(person?.retention_expires_at),
  }
}

function buildSearchIndex(person) {
  const metadata = person.metadata ?? {}
  return [
    person.name,
    person.id,
    person.person_type,
    getStatusLabel(person),
    metadata.alias,
    metadata.case_id,
    metadata.organization,
    metadata.risk_level,
    metadata.notes,
    person.notes,
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
}

function getStatusLabel(person) {
  return person.status === 'active' ? 'active' : 'inactive'
}

function getPersonTypeLabel(person) {
  return person.person_type ?? person.metadata?.person_type ?? 'watchlist'
}

function ReferenceImage({ personId, reference, sensitiveAccessToken, onSensitiveAccessExpired, onOpen, childrenCount = 0 }) {
  const [url, setUrl] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let active = true
    let objectUrl = null
    setUrl(null)
    setFailed(false)
    getPOIReferenceImage(personId, reference.id, sensitiveAccessToken)
      .then((blob) => {
        if (!active) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
      })
      .catch((error) => {
        if (!active) return
        if (error.status === 401 || error.status === 403) onSensitiveAccessExpired?.()
        setFailed(true)
      })
    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [personId, reference.id, sensitiveAccessToken, onSensitiveAccessExpired])

  return (
    <div className="poi-reference-image-card">
      <button type="button" className="poi-reference-open" onClick={onOpen} aria-label={`Open ${reference.original_filename}`}>
      {url
        ? <img src={url} alt={reference.original_filename} className="poi-reference-image" />
        : <div className="poi-reference-loading">{failed ? 'Image unavailable' : 'Loading image…'}</div>}
      </button>
      <div className="poi-reference-image-name" title={reference.original_filename}>{reference.original_filename}</div>
      <div className="poi-reference-image-meta">{reference.width} × {reference.height}</div>
      <div className={`poi-reference-tag ${reference.used_for_recognition === false ? 'preview' : ''}`}>
        {reference.image_kind && reference.image_kind !== 'original' ? 'Synthetic · ' : 'Original · '}
        {reference.used_for_recognition === false ? 'Preview only' : 'Recognition template'}
      </div>
      {childrenCount > 0 && <button type="button" className="poi-reference-link" onClick={onOpen}>Explore {childrenCount} generated images →</button>}
      {reference.generation?.reason && <div className="poi-reference-image-meta">{reference.generation.reason}</div>}
    </div>
  )
}

export function ReferenceGallery({ references, ...props }) {
  const [path, setPath] = useState([])
  const current = path[path.length - 1]
  const children = references.filter((r) => current ? r.parent_id === current.id : !r.parent_id)
  const groups = [
    ['original', 'Original registration photos'],
    ['view', 'Generated angles'],
    ['environment', 'Environment simulations'],
  ]
  return <div className="poi-reference-browser">
    <nav className="poi-reference-breadcrumbs" aria-label="Reference image navigation">
      <button type="button" onClick={() => setPath([])}>Original photos</button>
      {path.map((item, index) => <span key={item.id}> / <button type="button" onClick={() => setPath(path.slice(0, index + 1))}>{item.original_filename}</button></span>)}
    </nav>
    {current && <button type="button" className="poi-clear-btn" onClick={() => setPath(path.slice(0, -1))}>← Back</button>}
    {current && <div className="poi-reference-selected">
      <ReferenceImage key={current.id} {...props} reference={current} />
      {current.generation?.parameters && <dl className="poi-reference-parameters">
        {Object.entries(current.generation.parameters).map(([name, value]) => <div key={name}><dt>{({exposure:'Exposure multiplier', noise_sigma:'Noise σ (0–255)', scale:'Resolution scale'})[name] || name}</dt><dd>{Number(value).toFixed(3)}</dd></div>)}
      </dl>}
    </div>}
    {groups.map(([kind, title]) => {
      const items = children.filter((r) => (r.image_kind || 'original') === kind)
      return items.length > 0 && <section key={kind}>
        <h3>{title}</h3>
        {kind === 'view' && <p className="poi-reference-image-meta">Select an angle to explore its environment simulations. Yaw labels are generator commands.</p>}
        <div className="poi-reference-grid">{items.map((reference) => <ReferenceImage
          key={reference.id} {...props} reference={reference}
          childrenCount={references.filter((r) => r.parent_id === reference.id).length}
          onOpen={() => setPath([...path, reference])}
        />)}</div>
      </section>
    })}
    {current && !children.length && current.image_kind !== 'environment' && <p className="poi-reference-empty">No generated descendants are available for this image.</p>}
  </div>
}

export default function FaceSearch({ people = [], currentUser, onRefresh, onNavigate, focusPerson = null }) {
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [editingPerson, setEditingPerson] = useState(null)
  const [editForm, setEditForm] = useState(EMPTY_EDIT_FORM)
  const [workingId, setWorkingId] = useState('')
  const [notice, setNotice] = useState(null)
  const [referencePerson, setReferencePerson] = useState(null)
  const [references, setReferences] = useState([])
  const [referencesLoading, setReferencesLoading] = useState(false)
  const [referencesError, setReferencesError] = useState('')
  const [referencesGenerating, setReferencesGenerating] = useState(false)
  const [generationNotice, setGenerationNotice] = useState('')
  const [sensitiveAccess, setSensitiveAccess] = useState(null)
  const [sensitivePerson, setSensitivePerson] = useState(null)
  const [sensitivePassword, setSensitivePassword] = useState('')
  const [sensitiveError, setSensitiveError] = useState('')
  const [sensitiveWorking, setSensitiveWorking] = useState(false)
  const referenceInputRef = useRef(null)
  const [referenceFiles, setReferenceFiles] = useState([])
  const [referenceUploading, setReferenceUploading] = useState(false)

  const isAdmin = (currentUser?.roles ?? []).some((role) => String(role).toLowerCase() === 'admin')

  useEffect(() => {
    if (!editingPerson) {
      setEditForm(EMPTY_EDIT_FORM)
      return
    }
    setEditForm(buildEditForm(editingPerson))
  }, [editingPerson])

  // Opened from the live POI panel: filter the registry down to that person and scroll to it.
  useEffect(() => {
    if (!focusPerson?.personId) {
      return
    }
    setQuery(String(focusPerson.personId))
    setTypeFilter('all')
    setStatusFilter('all')
    const frame = requestAnimationFrame(() => {
      document.getElementById('poi-registry')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
    return () => cancelAnimationFrame(frame)
  }, [focusPerson])

  const filteredPeople = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    return people
      .filter((person) => {
        if (typeFilter !== 'all' && getPersonTypeLabel(person) !== typeFilter) {
          return false
        }
        if (statusFilter !== 'all' && getStatusLabel(person) !== statusFilter) {
          return false
        }
        if (!normalizedQuery) {
          return true
        }
        return buildSearchIndex(person).includes(normalizedQuery)
      })
      .sort((left, right) => {
        const leftTime = left.last_seen_time ? new Date(left.last_seen_time).getTime() : 0
        const rightTime = right.last_seen_time ? new Date(right.last_seen_time).getTime() : 0
        return rightTime - leftTime || left.name.localeCompare(right.name)
      })
  }, [people, query, statusFilter, typeFilter])

  const summary = useMemo(() => ({
    total: people.length,
    watchlist: people.filter((person) => getPersonTypeLabel(person) === 'watchlist').length,
    liveNow: people.filter((person) => getStatusLabel(person) === 'active').length,
    embeddings: people.reduce((sum, person) => sum + (person.total_embeddings ?? 0), 0),
  }), [people])

  const updateEditForm = (field, value) => {
    setEditForm((current) => ({ ...current, [field]: value }))
  }

  const closeEditor = () => {
    setEditingPerson(null)
    setWorkingId('')
    setReferenceFiles([])
  }

  const handleSave = async () => {
    if (!editingPerson) {
      return
    }
    const trimmedName = editForm.name.trim()
    if (!trimmedName) {
      setNotice({ type: 'error', text: 'POI name is required before saving.' })
      return
    }

    setWorkingId(editingPerson.id)
    try {
      await updatePOI(editingPerson.id, {
        name: trimmedName,
        person_type: editForm.person_type,
        notes: editForm.notes.trim() || null,
        is_active: editForm.is_active,
        embedding_only_mode: editForm.embedding_only_mode,
        retention_expires_at: editForm.retention_expires_at
          ? new Date(editForm.retention_expires_at).toISOString()
          : null,
      })
      await onRefresh?.()
      setNotice({ type: 'success', text: `Updated ${trimmedName}.` })
      closeEditor()
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
    } finally {
      setWorkingId('')
    }
  }

  const handleDelete = async (person) => {
    const confirmed = window.confirm(`Delete POI profile "${person.name}"? This keeps existing audit history but removes the profile from active use.`)
    if (!confirmed) {
      return
    }

    setWorkingId(person.id)
    try {
      await deletePOI(person.id)
      await onRefresh?.()
      if (editingPerson?.id === person.id) {
        closeEditor()
      }
      setNotice({ type: 'success', text: `Deleted ${person.name}.` })
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
    } finally {
      setWorkingId('')
    }
  }

  const loadReferences = async (person, token) => {
    setReferencePerson(person)
    setReferences([])
    setReferencesError('')
    setReferencesLoading(true)
    try {
      setReferences(await getPOIReferences(person.id, token))
    } catch (error) {
      if (error.status === 401 || error.status === 403) {
        setSensitiveAccess(null)
        setReferencePerson(null)
        setSensitivePerson(person)
        setSensitiveError('Sensitive access has expired. Re-enter your password to continue.')
        return
      }
      setReferencesError(error.message || 'Reference images could not be loaded.')
    } finally {
      setReferencesLoading(false)
    }
  }

  const openReferences = (person) => {
    setGenerationNotice('')
    if (!isAdmin) {
      setNotice({ type: 'error', text: 'Only administrators can view retained reference images.' })
      return
    }
    if (!sensitiveAccess || Date.parse(sensitiveAccess.expiresAt) <= Date.now()) {
      setSensitiveAccess(null)
      setSensitivePerson(person)
      setSensitivePassword('')
      setSensitiveError('')
      return
    }
    void loadReferences(person, sensitiveAccess.token)
  }

  const confirmSensitiveAccess = async (event) => {
    event.preventDefault()
    if (!sensitivePerson || !sensitivePassword) return
    setSensitiveWorking(true)
    setSensitiveError('')
    try {
      const response = await requestSensitiveAccess(sensitivePassword)
      const nextAccess = { token: response.sensitive_access_token, expiresAt: response.expires_at }
      const person = sensitivePerson
      setSensitiveAccess(nextAccess)
      setSensitivePerson(null)
      setSensitivePassword('')
      await loadReferences(person, nextAccess.token)
    } catch (error) {
      setSensitiveError(error.detail || 'Password verification failed.')
    } finally {
      setSensitiveWorking(false)
    }
  }

  const handleSensitiveAccessExpired = useCallback(() => {
    if (!referencePerson) return
    const person = referencePerson
    setSensitiveAccess(null)
    setReferencePerson(null)
    setReferences([])
    setSensitivePerson(person)
    setSensitivePassword('')
    setSensitiveError('Sensitive access has expired. Re-enter your password to continue.')
  }, [referencePerson])

  useEffect(() => {
    if (!referencePerson || !sensitiveAccess?.expiresAt) return
    const timer = window.setTimeout(handleSensitiveAccessExpired,
      Math.max(0, Date.parse(sensitiveAccess.expiresAt) - Date.now()))
    return () => window.clearTimeout(timer)
  }, [referencePerson, sensitiveAccess?.expiresAt, handleSensitiveAccessExpired])

  const generateReferences = async () => {
    if (!referencePerson || referencesGenerating) return
    const person = referencePerson
    setReferencesGenerating(true)
    setGenerationNotice('Generating angles and environment simulations… This can take a few minutes.')
    try {
      const result = await generatePOIReferences(person.id, sensitiveAccess.token)
      setGenerationNotice(`${result.generated_images || 0} images generated; ${result.embeddings_added || 0} recognition templates added. ${(result.warnings || []).join(' ')}`)
      await loadReferences(person, sensitiveAccess.token)
      onRefresh?.()
    } catch (error) {
      if (error.status === 401 || error.status === 403) handleSensitiveAccessExpired()
      else setGenerationNotice(error.message || 'Generation failed. Try again.')
    } finally {
      setReferencesGenerating(false)
    }
  }

  const selectReferenceFiles = (incoming) => {
    const images = Array.from(incoming ?? []).filter((file) => file.type.startsWith('image/'))
    setReferenceFiles((current) => {
      const known = new Set(current.map((file) => `${file.name}-${file.size}-${file.lastModified}`))
      return [...current, ...images.filter((file) => !known.has(`${file.name}-${file.size}-${file.lastModified}`))]
    })
  }

  const addReferences = async () => {
    if (!editingPerson || !referenceFiles.length) return
    setReferenceUploading(true)
    try {
      const result = await addPOIReferences(editingPerson.id, referenceFiles)
      await onRefresh?.()
      setReferenceFiles([])
      const rejected = result.rejected_images?.length ?? 0
      setNotice({
        type: 'success',
        text: `${result.original_images_added ?? result.embeddings_added} original image(s) added; ${result.embeddings_added} recognition templates stored.${rejected ? ` ${rejected} rejected.` : ''} ${(result.enhancement?.warnings || []).join(' ')}`,
      })
    } catch (error) {
      const detail = error.detail
      const rejected = Array.isArray(detail?.rejected_images) ? ` ${detail.rejected_images.join('; ')}` : ''
      setNotice({ type: 'error', text: `${detail?.message || error.message}${rejected}` })
    } finally {
      setReferenceUploading(false)
    }
  }

  return (
    <div className="poi-page">
      <div className="poi-header-card">
        <div>
          <div className="poi-eyebrow">POI Profile Management</div>
          <h2 className="poi-title">Manage enrolled persons of interest</h2>
          <p className="poi-subtitle">
            Search the current POI registry by text or by photograph, edit profile details, remove outdated
            entries, or jump directly into the enrollment flow to create a new profile.
          </p>
        </div>

        <div className="poi-header-actions">
          <button className="poi-create-btn" type="button" onClick={() => onNavigate?.('enroll')}>
            Create POI
          </button>
        </div>
      </div>

      <div className="poi-summary-grid">
        <div className="poi-summary-card">
          <span className="poi-summary-label">Profiles</span>
          <span className="poi-summary-value">{summary.total}</span>
        </div>
        <div className="poi-summary-card">
          <span className="poi-summary-label">Watchlist</span>
          <span className="poi-summary-value">{summary.watchlist}</span>
        </div>
        <div className="poi-summary-card">
          <span className="poi-summary-label">Live now</span>
          <span className="poi-summary-value">{summary.liveNow}</span>
        </div>
        <div className="poi-summary-card">
          <span className="poi-summary-label">Embeddings</span>
          <span className="poi-summary-value">{summary.embeddings}</span>
        </div>
      </div>

      <PhotoSearchPanel
        people={people}
        onShowProfile={(personId) => {
          setQuery(personId)
          setTypeFilter('all')
          setStatusFilter('all')
          document.getElementById('poi-registry')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }}
      >
        <label className="poi-search-field">
          <span className="poi-field-label">Search</span>
          <input
            className="poi-input"
            type="search"
            placeholder="Search by name, ID, alias, case ID, organization..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <div className="poi-find-filters">
          <label className="poi-filter-field">
            <span className="poi-field-label">Type</span>
            <select className="poi-select" value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
              <option value="all">All types</option>
              {PERSON_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>

          <label className="poi-filter-field">
            <span className="poi-field-label">Status</span>
            <select className="poi-select" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
              {STATUS_OPTIONS.map((status) => (
                <option key={status} value={status}>
                  {status === 'all' ? 'All status' : `${status.charAt(0).toUpperCase()}${status.slice(1)}`}
                </option>
              ))}
            </select>
          </label>

          <button
            className="poi-clear-btn"
            type="button"
            onClick={() => {
              setQuery('')
              setTypeFilter('all')
              setStatusFilter('all')
            }}
          >
            Reset
          </button>
        </div>
      </PhotoSearchPanel>

      {notice && (
        <div className={`poi-notice ${notice.type}`}>
          {notice.text}
        </div>
      )}

      <div className="poi-results-head" id="poi-registry">
        <div className="poi-results-title">Profiles</div>
        <div className="poi-results-meta">
          {filteredPeople.length} of {people.length} shown
        </div>
      </div>

      <div className="poi-grid">
        {filteredPeople.map((person) => {
          const metadata = person.metadata ?? {}
          const personType = getPersonTypeLabel(person)
          const status = getStatusLabel(person)
          const notesPreview =
            typeof metadata.notes === 'string'
              ? metadata.notes
              : typeof person.notes === 'string'
                ? person.notes
                : ''

          return (
            <article key={person.id} className="poi-card">
              <div className="poi-card-top">
                <div className="poi-avatar">{getInitials(person.name)}</div>
                <div className="poi-card-heading">
                  <div className="poi-card-name">{person.name}</div>
                  <div className="poi-card-id">{person.id}</div>
                </div>
                <div className="poi-card-badges">
                  <span className={`poi-pill type-${personType}`}>{personType}</span>
                  <span className={`poi-pill status-${status}`}>{status}</span>
                </div>
              </div>

              <div className="poi-card-meta">
                <div className="poi-meta-row">
                  <span className="poi-meta-label">Embeddings</span>
                  <span className="poi-meta-value">{person.total_embeddings ?? 0}</span>
                </div>
                <div className="poi-meta-row">
                  <span className="poi-meta-label">Enrolled</span>
                  <span className="poi-meta-value">{formatDateTime(person.enrolled_at)}</span>
                </div>
                <div className="poi-meta-row">
                  <span className="poi-meta-label">Last seen</span>
                  <span className="poi-meta-value">
                    {person.last_seen_camera
                      ? `${person.last_seen_camera} | ${formatDateTime(person.last_seen_time)}`
                      : 'No recognition event yet'}
                  </span>
                </div>
                <div className="poi-meta-row">
                  <span className="poi-meta-label">Mode</span>
                  <span className="poi-meta-value">{person.embedding_only_mode ? 'Embedding only' : 'Full profile'}</span>
                </div>
                <div className="poi-meta-row">
                  <span className="poi-meta-label">Profile</span>
                  <span className="poi-meta-value">{person.is_active ? 'Enabled' : 'Disabled'}</span>
                </div>
                {metadata.alias && (
                  <div className="poi-meta-row">
                    <span className="poi-meta-label">Alias</span>
                    <span className="poi-meta-value">{metadata.alias}</span>
                  </div>
                )}
                {metadata.case_id && (
                  <div className="poi-meta-row">
                    <span className="poi-meta-label">Case ID</span>
                    <span className="poi-meta-value">{metadata.case_id}</span>
                  </div>
                )}
              </div>

              <div className="poi-notes-preview">
                {notesPreview || 'No additional notes recorded for this profile.'}
              </div>

              <div className="poi-card-actions">
                {isAdmin && (
                  <button
                    className="poi-action-btn view"
                    type="button"
                    onClick={() => openReferences(person)}
                  >
                    References
                  </button>
                )}
                <button
                  className="poi-action-btn primary"
                  type="button"
                  onClick={() => setEditingPerson(person)}
                >
                  Edit
                </button>
                <button
                  className="poi-action-btn danger"
                  type="button"
                  disabled={workingId === person.id}
                  onClick={() => handleDelete(person)}
                >
                  {workingId === person.id ? 'Deleting...' : 'Delete'}
                </button>
              </div>
            </article>
          )
        })}

        {filteredPeople.length === 0 && (
          <div className="poi-empty-card">
            <div className="poi-empty-title">No matching POI profiles</div>
            <div className="poi-empty-copy">
              Adjust the search or filters, or create a new POI profile from the enrollment page.
            </div>
            <button className="poi-create-btn secondary" type="button" onClick={() => onNavigate?.('enroll')}>
              Go to Enrollment
            </button>
          </div>
        )}
      </div>

      {editingPerson && (
        <div className="poi-modal-backdrop" onClick={closeEditor}>
          <div
            className="poi-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <div className="poi-modal-head">
              <div>
                <div className="poi-modal-title">Edit POI profile</div>
                <div className="poi-modal-subtitle">{editingPerson.id}</div>
              </div>
              <button className="poi-icon-btn" type="button" onClick={closeEditor}>
                X
              </button>
            </div>

            <div className="poi-modal-grid">
              <label className="poi-form-field">
                <span className="poi-field-label">Full name</span>
                <input
                  className="poi-input"
                  type="text"
                  value={editForm.name}
                  onChange={(event) => updateEditForm('name', event.target.value)}
                />
              </label>

              <label className="poi-form-field">
                <span className="poi-field-label">Person type</span>
                <select
                  className="poi-select"
                  value={editForm.person_type}
                  onChange={(event) => updateEditForm('person_type', event.target.value)}
                >
                  {PERSON_TYPES.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </label>

              <label className="poi-form-field">
                <span className="poi-field-label">Retention expiry</span>
                <input
                  className="poi-input"
                  type="datetime-local"
                  value={editForm.retention_expires_at}
                  onChange={(event) => updateEditForm('retention_expires_at', event.target.value)}
                />
              </label>

              <div className="poi-switches">
                <label className="poi-switch">
                  <input
                    type="checkbox"
                    checked={editForm.is_active}
                    onChange={(event) => updateEditForm('is_active', event.target.checked)}
                  />
                  <span>Active profile</span>
                </label>

                <label className="poi-switch">
                  <input
                    type="checkbox"
                    checked={editForm.embedding_only_mode}
                    onChange={(event) => updateEditForm('embedding_only_mode', event.target.checked)}
                  />
                  <span>Embedding-only mode</span>
                </label>
              </div>

              <label className="poi-form-field full">
                <span className="poi-field-label">Notes</span>
                <textarea
                  className="poi-textarea"
                  value={editForm.notes}
                  onChange={(event) => updateEditForm('notes', event.target.value)}
                  placeholder="Plain text or JSON stored in the notes field"
                />
              </label>

              <div className="poi-reference-upload full">
                <div>
                  <div className="poi-field-label">Add reference images</div>
                  <div className="poi-reference-upload-copy">
                    Upload clear single-person images. Each one is checked for one face, size, confidence, blur and duplicate embeddings.
                  </div>
                  {referenceFiles.length > 0 && (
                    <div className="poi-reference-file-list">
                      {referenceFiles.map((file) => <span key={`${file.name}-${file.size}-${file.lastModified}`}>{file.name}</span>)}
                    </div>
                  )}
                </div>
                <div className="poi-reference-upload-actions">
                  <input
                    ref={referenceInputRef}
                    className="poi-file-input"
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(event) => {
                      selectReferenceFiles(event.target.files)
                      event.target.value = ''
                    }}
                  />
                  <button className="poi-clear-btn" type="button" disabled={referenceUploading} onClick={() => referenceInputRef.current?.click()}>
                    Choose images
                  </button>
                  <button className="poi-create-btn" type="button" disabled={referenceUploading || !referenceFiles.length} onClick={addReferences}>
                    {referenceUploading ? 'Adding…' : `Add ${referenceFiles.length || ''} reference${referenceFiles.length === 1 ? '' : 's'}`}
                  </button>
                </div>
              </div>
            </div>

            <div className="poi-modal-actions">
              <button className="poi-clear-btn" type="button" onClick={closeEditor}>
                Cancel
              </button>
              <button
                className="poi-create-btn"
                type="button"
                disabled={workingId === editingPerson.id}
                onClick={handleSave}
              >
                {workingId === editingPerson.id ? 'Saving...' : 'Save changes'}
              </button>
            </div>
          </div>
        </div>
      )}

      {referencePerson && (
        <div className="poi-modal-backdrop" onClick={() => !referencesGenerating && setReferencePerson(null)}>
          <div className="poi-modal poi-reference-modal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <div className="poi-modal-head">
              <div>
                <div className="poi-modal-title">Reference images</div>
                <div className="poi-modal-subtitle">{referencePerson.name} · {references.length} retained image(s)</div>
              </div>
              <button className="poi-icon-btn" type="button" disabled={referencesGenerating} onClick={() => setReferencePerson(null)}>X</button>
            </div>
            {referencesLoading && <div className="poi-reference-empty">Loading reference images…</div>}
            {referencesError && <div className="poi-reference-empty">{referencesError}</div>}
            {!referencesLoading && references.some((r) => (r.image_kind || 'original') === 'original' && !references.some((child) => child.original_reference_id === r.id)) && <div className="poi-reference-generation">
              <button type="button" className="poi-create-btn" disabled={referencesGenerating} onClick={generateReferences}>{referencesGenerating ? 'Generating…' : 'Generate missing images'}</button>
              <p>Generate angles and environment simulations for original photos without generated images. Eligible templates will be added to recognition.</p>
            </div>}
            {generationNotice && <div className="poi-reference-empty" role="status">{generationNotice}</div>}
            {!referencesLoading && !referencesError && !references.length && (
              <div className="poi-reference-empty">
                No viewable reference images were retained for this profile. Images uploaded before this feature cannot be restored; enrol the profile again to store them.
              </div>
            )}
            {!referencesLoading && !referencesError && references.length > 0 && (
              <ReferenceGallery
                    key={`${referencePerson.id}-${sensitiveAccess?.token}`}
                    references={references}
                    personId={referencePerson.id}
                    sensitiveAccessToken={sensitiveAccess?.token}
                    onSensitiveAccessExpired={handleSensitiveAccessExpired}
                  />
            )}
          </div>
        </div>
      )}

      {sensitivePerson && (
        <div className="poi-modal-backdrop" onClick={() => !sensitiveWorking && setSensitivePerson(null)}>
          <form className="poi-modal poi-sensitive-modal" onClick={(event) => event.stopPropagation()} onSubmit={confirmSensitiveAccess} role="dialog" aria-modal="true">
            <div className="poi-modal-head">
              <div>
                <div className="poi-modal-title">Confirm sensitive access</div>
                <div className="poi-modal-subtitle">Re-enter your password to view {sensitivePerson.name}'s reference images.</div>
              </div>
              <button className="poi-icon-btn" type="button" disabled={sensitiveWorking} onClick={() => setSensitivePerson(null)}>X</button>
            </div>
            <div className="poi-sensitive-copy">Access is limited to administrators, expires after 5 minutes, and each image view is recorded in the audit log.</div>
            <label className="poi-form-field full">
              <span className="poi-field-label">Your password</span>
              <input className="poi-input" type="password" autoComplete="current-password" autoFocus value={sensitivePassword} onChange={(event) => setSensitivePassword(event.target.value)} />
            </label>
            {sensitiveError && <div className="poi-sensitive-error">{sensitiveError}</div>}
            <div className="poi-modal-actions">
              <button className="poi-clear-btn" type="button" disabled={sensitiveWorking} onClick={() => setSensitivePerson(null)}>Cancel</button>
              <button className="poi-create-btn" type="submit" disabled={sensitiveWorking || !sensitivePassword}>{sensitiveWorking ? 'Verifying…' : 'Continue'}</button>
            </div>
          </form>
        </div>
      )}
    </div>
  )
}
