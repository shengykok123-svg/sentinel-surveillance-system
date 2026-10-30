import { useEffect, useRef, useState } from 'react'
import { enrollPerson, assessImageQuality, getEnrollmentJob } from '../utils/api'
import './EnrollmentPage.css'

const PERSON_TYPES = ['watchlist', 'vip', 'staff', 'visitor']

const EMPTY_FORM = {
  fullName: '',
  personType: 'watchlist',
  isActive: true,
  embeddingOnlyMode: false,
  retentionExpiresAt: '',
  notes: '',
}

function buildFileKey(file) {
  return `${file.name}-${file.size}-${file.lastModified}`
}

function buildEnrollmentFailure(error) {
  const detail = error?.detail
  const message =
    typeof detail === 'string'
      ? detail
      : detail?.message || error?.message || 'Profile creation failed.'

  const code = typeof detail === 'object' && detail?.code ? detail.code : 'profile_creation_failed'
  const rejectedImages =
    typeof detail === 'object' && Array.isArray(detail?.rejected_images) ? detail.rejected_images : []

  let title = 'Profile not created'
  if (code === 'face_not_detected') {
    title = 'Face not detected'
  } else if (code === 'landmark_extraction_failed') {
    title = 'Landmark extraction failed'
  }

  return { code, title, message, rejectedImages }
}

function IconUpload() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 16V7" strokeLinecap="round" />
      <path d="M8.5 10.5L12 7l3.5 3.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5 17.5h14" strokeLinecap="round" />
    </svg>
  )
}

export default function EnrollmentPage({ onEnroll, currentUser }) {
  const storageKey = `sentinel.enrollment-job.${currentUser?.id ?? 'preview'}`
  const [jobId, setJobId] = useState(() => sessionStorage.getItem(storageKey))
  const [job, setJob] = useState(null)
  const [pollError, setPollError] = useState(null)
  const onEnrollRef = useRef(onEnroll)
  onEnrollRef.current = onEnroll
  const inputRef = useRef(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [loading, setLoading] = useState(() => Boolean(sessionStorage.getItem(storageKey)))
  const [result, setResult] = useState(null)
  const [failure, setFailure] = useState(null)
  const [toast, setToast] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [quality, setQuality] = useState({})
  const [qualityRetry, setQualityRetry] = useState(0)

  useEffect(() => {
    if (!jobId) return
    const controller = new AbortController()
    let timer
    let missingCount = 0
    setLoading(true)
    async function poll() {
      try {
        const state = await getEnrollmentJob(jobId, controller.signal)
        if (controller.signal.aborted) return
        missingCount = 0
        setJob(state)
        setPollError(null)
        if (state.status === 'completed' || state.status === 'failed') {
          sessionStorage.removeItem(storageKey)
          setJobId(null)
          setLoading(false)
          if (state.status === 'completed') {
            setResult(state.result)
            setFailure(null)
            setFiles([])
            setForm((current) => ({ ...current, fullName: '', notes: '', retentionExpiresAt: '' }))
            setToast({ type: 'success', title: 'Enrollment complete', message: `${state.result.full_name} is ready for recognition.` })
            onEnrollRef.current?.()
          } else {
            setFailure(buildEnrollmentFailure({ detail: state.error }))
          }
          return
        }
      } catch (error) {
        if (controller.signal.aborted) return
        if (error.status === 404 && ++missingCount >= 5) {
          sessionStorage.removeItem(storageKey)
          setJobId(null)
          setLoading(false)
          setFailure({ title: 'Enrollment status unavailable', message: 'The backend may have restarted or the upload may not have arrived. Check the person list before submitting again.', rejectedImages: [] })
          return
        }
        if (error.status === 401) return
        setPollError('Connection interrupted. Rechecking enrollment status…')
      }
      timer = setTimeout(poll, 2000)
    }
    poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [jobId, storageKey])

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setQuality({})
    // Sequential requests avoid flooding the detector when many images are selected.
    async function assessFiles() {
      for (const file of files) {
        if (!active) return
        const key = buildFileKey(file)
        try {
          const assessment = await assessImageQuality(file, controller.signal)
          if (active) setQuality((current) => ({ ...current, [key]: assessment }))
        } catch (error) {
          if (!active) return
          setQuality((current) => ({ ...current, [key]: {
            status: 'error', message: 'Quality check unavailable. Please retry.',
          } }))
        }
      }
    }
    assessFiles()
    return () => { active = false; controller.abort() }
  }, [files, qualityRetry])

  useEffect(() => {
    const next = files.map((file) => ({
      key: buildFileKey(file),
      name: file.name,
      sizeLabel: `${Math.max(file.size / 1024, 1).toFixed(1)} KB`,
      url: URL.createObjectURL(file),
    }))
    setPreviews(next)
    return () => next.forEach((preview) => URL.revokeObjectURL(preview.url))
  }, [files])

  useEffect(() => {
    if (!toast) {
      return undefined
    }
    const id = window.setTimeout(() => setToast(null), 4500)
    return () => window.clearTimeout(id)
  }, [toast])

  const updateForm = (field, value) => {
    setForm((current) => ({ ...current, [field]: value }))
  }

  const appendFiles = (incoming) => {
    if (loading) return
    const images = Array.from(incoming ?? []).filter((file) => file.type.startsWith('image/'))
    if (!images.length) {
      return
    }

    setFiles((current) => {
      const seen = new Set(current.map(buildFileKey))
      return [...current, ...images.filter((file) => {
        const key = buildFileKey(file)
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })]
    })
    setResult(null)
    setFailure(null)
    setToast(null)
  }

  const removeFile = (key) => {
    if (loading) return
    setFiles((current) => current.filter((file) => buildFileKey(file) !== key))
    setResult(null)
    setFailure(null)
    setToast(null)
  }

  const clearForm = () => {
    setForm(EMPTY_FORM)
    setFiles([])
    setResult(null)
    setFailure(null)
    setToast(null)
  }

  const submit = async () => {
    if (loading) return
    if (!qualityReady) return
    if (!form.fullName.trim()) {
      setToast({ type: 'error', title: 'Validation error', message: 'Full name is required.' })
      return
    }
    if (!files.length) {
      setToast({ type: 'error', title: 'Validation error', message: 'Upload at least one face image.' })
      return
    }

    const formData = new FormData()
    formData.append('full_name', form.fullName.trim())
    formData.append('person_type', form.personType)
    formData.append('is_active', String(form.isActive))
    formData.append('embedding_only_mode', String(form.embeddingOnlyMode))
    if (form.retentionExpiresAt) {
      formData.append('retention_expires_at', new Date(form.retentionExpiresAt).toISOString())
    }
    if (form.notes.trim()) {
      formData.append('notes', form.notes.trim())
    }
    files.forEach((file) => formData.append('images', file))

    setResult(null)
    setFailure(null)
    setLoading(true)
    setJob(null)
    const nextJobId = crypto.randomUUID()
    sessionStorage.setItem(storageKey, nextJobId)
    try {
      const data = await enrollPerson(formData, nextJobId)
      setJob(data)
      setJobId(nextJobId)
    } catch (error) {
      if (!error.status) {
        setJobId(nextJobId)
        return
      }
      sessionStorage.removeItem(storageKey)
      setLoading(false)
      const nextFailure = buildEnrollmentFailure(error)
      setResult(null)
      setFailure(nextFailure)
      setToast({
        type: 'error',
        title: nextFailure.title,
        message: nextFailure.message,
      })
    }
  }

  const qualityReady = files.length > 0 && files.every((file) => {
    const state = quality[buildFileKey(file)]?.status
    return state && state !== 'error'
  }) && files.some((file) => quality[buildFileKey(file)]?.status === 'passed')
  const canSubmit = !loading && form.fullName.trim() && qualityReady

  return (
    <div className="ep">
      {loading && <div className="ep-quality-note" role="status" aria-live="polite">
        {pollError || (job?.status === 'queued' ? 'Waiting for the enrollment worker…' : job ? 'Processing reference images and preparing recognition…' : 'Uploading reference images…')}
        {job && ` Elapsed: ${Math.floor(job.elapsed_seconds ?? 0)} seconds.`}
        {' You can switch pages after the upload finishes; return here to check the result.'}
      </div>}
      {toast && (
        <div className={`ep-toast ${toast.type}`}>
          <div className="ep-toast-icon">{toast.type === 'success' ? 'OK' : '!'}</div>
          <div className="ep-toast-body">
            <div className="ep-toast-title">{toast.title}</div>
            <div className="ep-toast-msg">{toast.message}</div>
          </div>
        </div>
      )}

      <header className="ep-header">
        <div>
          <div className="ep-eyebrow">POI Enrollment</div>
          <h2 className="ep-title">Create and index a person profile</h2>
          <p className="ep-desc">
            Add reference photos to create a person profile and make it available for live recognition.
          </p>
        </div>
        <div className="ep-chips">
          <div className="ep-chip">
            <span className="ep-chip-label">Images queued</span>
            <span className="ep-chip-val">{files.length}</span>
          </div>
          <div className="ep-chip">
            <span className="ep-chip-label">Type</span>
            <span className="ep-chip-val sm">{form.personType}</span>
          </div>
          <div className="ep-chip">
            <span className="ep-chip-label">Mode</span>
            <span className="ep-chip-val sm">{form.embeddingOnlyMode ? 'Embedding' : 'Full profile'}</span>
          </div>
        </div>
      </header>

      <div className="ep-shell">
        <section className="ep-card">
          <div className="ep-card-head">
            <div>
              <div className="ep-card-title">Profile details</div>
              <div className="ep-card-sub">Identity and recognition preferences</div>
            </div>
            <div className="ep-badge">Operator flow</div>
          </div>

          <hr className="ep-divider" />

          <div className="ep-grid">
            <div className="ep-field">
              <label className="ep-label" htmlFor="ep-fullname">Full Name *</label>
              <input
                id="ep-fullname"
                className="ep-input"
                type="text"
                value={form.fullName}
                onChange={(event) => updateForm('fullName', event.target.value)}
                placeholder="e.g. Alice Tan"
                autoComplete="off"
              />
            </div>

            <div className="ep-field">
              <label className="ep-label" htmlFor="ep-type">Person Type</label>
              <select
                id="ep-type"
                className="ep-select"
                value={form.personType}
                onChange={(event) => updateForm('personType', event.target.value)}
              >
                {PERSON_TYPES.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>

            <div className="ep-field full">
              <label className="ep-label" htmlFor="ep-retention">
                Retention Expiry <span style={{ color: 'var(--t3)', fontWeight: 400 }}>(optional)</span>
              </label>
              <input
                id="ep-retention"
                className="ep-input"
                type="datetime-local"
                value={form.retentionExpiresAt}
                onChange={(event) => updateForm('retentionExpiresAt', event.target.value)}
              />
            </div>

            <div className="ep-toggles">
              <label className="ep-toggle">
                <input
                  type="checkbox"
                  checked={form.isActive}
                  onChange={(event) => updateForm('isActive', event.target.checked)}
                />
                <div className="ep-toggle-copy">
                  <div className="ep-toggle-title">Active profile</div>
                  <div className="ep-toggle-sub">Available for live recognition immediately after enrollment.</div>
                </div>
              </label>

              <label className="ep-toggle">
                <input
                  type="checkbox"
                  checked={form.embeddingOnlyMode}
                  onChange={(event) => updateForm('embeddingOnlyMode', event.target.checked)}
                />
                <div className="ep-toggle-copy">
                  <div className="ep-toggle-title">Embedding-only mode</div>
                  <div className="ep-toggle-sub">Store vectors only - raw images are not retained.</div>
                </div>
              </label>
            </div>

            <div className="ep-field full">
              <label className="ep-label" htmlFor="ep-notes">
                Notes / Extended JSON <span style={{ color: 'var(--t3)', fontWeight: 400 }}>(optional)</span>
              </label>
              <textarea
                id="ep-notes"
                className="ep-textarea"
                value={form.notes}
                onChange={(event) => updateForm('notes', event.target.value)}
                placeholder={'Plain text or JSON, e.g. {"alias":"Night Fox","risk_level":"high"}'}
              />
            </div>
          </div>
        </section>

        <aside className="ep-aside">
          <section className="ep-card">
            <div className="ep-card-head">
              <div>
                <div className="ep-card-title">Face image intake</div>
                <div className="ep-card-sub">One or more frontal face crops</div>
              </div>
              <div className="ep-badge">{files.length} selected</div>
            </div>

            <hr className="ep-divider" />

            <div
              className={`ep-drop${dragging ? ' drag-over' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                appendFiles(event.dataTransfer.files)
              }}
            >
              <div className="ep-drop-icon"><IconUpload /></div>
              <div className="ep-drop-title">Drop face images here</div>
              <div className="ep-drop-sub">Duplicates are filtered before FAISS indexing. JPG, PNG, WEBP accepted.</div>
              <button
                type="button"
                className="ep-upload-btn"
                onClick={(event) => {
                  event.stopPropagation()
                  inputRef.current?.click()
                }}
              >
                Select images
              </button>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                multiple
                className="ep-file-input"
                onChange={(event) => {
                  appendFiles(event.target.files)
                  event.target.value = ''
                }}
              />
            </div>

            {previews.length > 0 ? (
              <div className="ep-preview-grid">
                {previews.map((preview) => (
                  <div className="ep-preview-card" key={preview.key}>
                    <img src={preview.url} alt={preview.name} className="ep-preview-img" />
                    <div>
                      <div className="ep-preview-name" title={preview.name}>{preview.name}</div>
                      <div className="ep-preview-size">{preview.sizeLabel}</div>
                      <div className={`ep-quality ${quality[preview.key]?.status || 'checking'}`} role="status">
                        <strong>{({ passed: 'Quality passed', low_quality: 'Below threshold', invalid_image: 'Cannot assess', error: 'Check failed' })[quality[preview.key]?.status] || 'Checking quality…'}</strong>
                        <span>CR-FIQA: {Number.isFinite(quality[preview.key]?.score) ? quality[preview.key].score.toFixed(4) : '—'}</span>
                        {Number.isFinite(quality[preview.key]?.threshold) && <span>Required ≥ {quality[preview.key].threshold.toFixed(4)}</span>}
                        {quality[preview.key]?.message && <span>{quality[preview.key].message}</span>}
                      </div>
                    </div>
                    <button type="button" className="ep-remove-btn" disabled={loading} onClick={() => removeFile(preview.key)}>
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="ep-empty">
                Uploaded images appear here before AdaFace embedding extraction runs.
              </div>
            )}

            <hr className="ep-divider" />
            <div className="ep-actions">
              <button
                type="button"
                className="ep-submit-btn"
                onClick={submit}
                disabled={!canSubmit}
              >
                {loading ? 'Generating images & enrolling…' : 'Enroll POI'}
              </button>
              <button
                type="button"
                className="ep-clear-btn"
                onClick={clearForm}
                disabled={loading}
              >
                Clear
              </button>
            </div>
            {files.length > 0 && <div className="ep-quality-note">
              Enrollment generates angle views and environment simulations, then adds eligible recognition templates. This can take a few minutes. After upload, processing continues when you switch pages.
              Quality scores are not percentages. All photos meeting the threshold can be enrolled; duplicate checks run during enrollment.
              {Object.values(quality).some((item) => item.status === 'error') && <button type="button" className="ep-clear-btn" disabled={loading} onClick={() => setQualityRetry((value) => value + 1)}>Retry quality checks</button>}
            </div>}
          </section>
        </aside>
      </div>

      <section className="ep-card">
        <div className="ep-card-head">
          <div>
            <div className="ep-card-title">Enrollment output</div>
            <div className="ep-card-sub">Profile and reference-image results</div>
          </div>
        </div>

        <hr className="ep-divider" />

        {result ? (
          <div className="ep-output">
            <div className="ep-result-banner">
              <div className="ep-result-status">{result.status?.toUpperCase()}</div>
              <div className="ep-result-name">{result.full_name}</div>
            </div>
            <div className="ep-result-metrics">
              <div className="ep-metric">
                <div className="ep-metric-label">Person ID</div>
                <div className="ep-metric-val">{result.person_id}</div>
              </div>
              <div className="ep-metric">
                <div className="ep-metric-label">Embeddings stored</div>
                <div className="ep-metric-val">{result.embeddings_count}</div>
              </div>
            </div>
            {result.rejected_images?.length ? (
              result.rejected_images.map((item) => (
                <div key={item} className="ep-alert warn">{item}</div>
              ))
            ) : (
              <div className="ep-alert ok">
                All uploaded images produced usable embeddings and were added to the search index.
              </div>
            )}
            {result.enhancement?.status && <div className="ep-alert ok">
              Enhancement: {result.enhancement.status}. {result.enhancement.generated_images || 0} generated images; {result.enhancement.embeddings_added || 0} additional recognition templates.
              {result.enhancement.retained_images > 0 && ' Open POI → References to browse original photos, generated angles and environment simulations.'}
              {form.embeddingOnlyMode && ' Embedding-only mode: generated images are not retained.'}
            </div>}
            {result.enhancement?.warnings?.map((warning) => <div key={warning} className="ep-alert warn">{warning}</div>)}
          </div>
        ) : failure ? (
          <div className="ep-output">
            <div className="ep-result-banner ep-result-banner-failure">
              <div className="ep-result-status ep-result-status-failure">FAILED</div>
              <div className="ep-result-name">{failure.title}</div>
            </div>
            <div className="ep-alert error">{failure.message}</div>
            {failure.rejectedImages?.length ? (
              failure.rejectedImages.map((item) => (
                <div key={item} className="ep-alert warn">{item}</div>
              ))
            ) : (
              <div className="ep-alert warn">
                Update the reference image and try again.
              </div>
            )}
          </div>
        ) : (
          <div className="ep-summary">
            <div className="ep-summary-banner">
              <div className="ep-summary-title">Current request</div>
              <div className="ep-summary-meta">
                {form.fullName.trim() || 'Unnamed profile'} | {form.personType} | {files.length} image{files.length !== 1 ? 's' : ''}
              </div>
            </div>
            <div className="ep-hint-list">
              {[
                'Face detection runs on each image before embedding extraction.',
                'Notes accept plain text or structured JSON stored in the notes field.',
                'Embeddings are written to PostgreSQL then indexed into FAISS.',
              ].map((hint) => (
                <div className="ep-hint" key={hint}>
                  <span className="ep-hint-dot" />
                  {hint}
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
