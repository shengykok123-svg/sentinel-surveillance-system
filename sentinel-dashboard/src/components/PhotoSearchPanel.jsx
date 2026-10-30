import { useEffect, useMemo, useRef, useState } from 'react'
import { getSettings, searchFace } from '../utils/api'
import './PhotoSearchPanel.css'

const TOP_K_OPTIONS = [5, 10, 20]

function formatDateTime(value) {
  return value ? new Date(value).toLocaleString() : ''
}

// Search the enrolled gallery with a photograph (use case UC06). The backend
// returns the closest enrolled people ranked by similarity and applies no
// threshold, so every candidate is shown with its score and marked against the
// live acceptance threshold for the operator to judge.
//
// Anything passed as children is laid out beside the photograph search, in the
// same card, so the page's text search and filters sit in the space the upload
// controls leave free.
export default function PhotoSearchPanel({ people = [], onShowProfile, children }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(null)
  const [searching, setSearching] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    let active = true
    getSettings()
      .then((settings) => {
        const value = settings?.recognition?.recognition_threshold
        if (active && typeof value === 'number') setThreshold(value)
      })
      .catch(() => {})
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const peopleById = useMemo(() => new Map(people.map((person) => [person.id, person])), [people])

  const choose = (files) => {
    const image = Array.from(files ?? []).find((candidate) => candidate.type.startsWith('image/'))
    if (!image) {
      setError('Choose an image file (JPG, PNG or WEBP).')
      return
    }
    setFile(image)
    setResult(null)
    setError('')
  }

  const runSearch = async () => {
    if (!file || searching) return
    setSearching(true)
    setError('')
    setResult(null)
    try {
      setResult(await searchFace(file, topK))
    } catch (err) {
      const detail = typeof err.detail === 'string' ? err.detail : err.detail?.message
      setError(err.status === 400
        ? (detail || 'No face could be found in this photograph.')
        : (detail || err.message || 'The search could not be completed.'))
    } finally {
      setSearching(false)
    }
  }

  const clear = () => {
    setFile(null)
    setResult(null)
    setError('')
  }

  const openPicker = () => {
    if (!searching) inputRef.current?.click()
  }

  const results = result?.results ?? []
  const atThreshold = threshold == null ? 0 : results.filter((match) => match.similarity >= threshold).length

  const photoSearch = (
    <div className="poi-find-photo">
      <div>
        <div className="poi-results-title" id="poi-photo-title">Search by photograph</div>
        <p className="poi-photo-copy">
          Upload a photograph to find the closest enrolled people. Candidates are ranked by similarity and listed
          even when none reaches the acceptance threshold{threshold != null ? ` of ${threshold.toFixed(2)}` : ''},
          so judge each one against the photograph.
        </p>
      </div>

      <div className="poi-photo-body">
        <div
          className={`poi-photo-drop ${dragging ? 'dragging' : ''}`}
          role="button"
          tabIndex={0}
          aria-label="Choose a photograph to search with"
          onClick={openPicker}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              openPicker()
            }
          }}
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragging(false)
            choose(event.dataTransfer.files)
          }}
        >
          {previewUrl
            ? <img src={previewUrl} alt="Photograph to search with" className="poi-photo-preview" />
            : <div className="poi-photo-placeholder">Drop a photograph here, or click to choose one</div>}
          <input
            ref={inputRef}
            className="poi-file-input"
            type="file"
            accept="image/*"
            onChange={(event) => {
              choose(event.target.files)
              event.target.value = ''
            }}
          />
        </div>

        <div className="poi-photo-controls">
          <label className="poi-filter-field">
            <span className="poi-field-label">Candidates</span>
            <select className="poi-select" value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {TOP_K_OPTIONS.map((k) => <option key={k} value={k}>Top {k}</option>)}
            </select>
          </label>
          {file && <div className="poi-photo-file" title={file.name}>{file.name}</div>}
          <div className="poi-photo-buttons">
            <button type="button" className="poi-clear-btn" disabled={searching} onClick={openPicker}>
              {file ? 'Choose another' : 'Choose photograph'}
            </button>
            <button type="button" className="poi-create-btn" disabled={!file || searching} onClick={runSearch}>
              {searching ? 'Searching…' : 'Search'}
            </button>
            {(file || result) && (
              <button type="button" className="poi-clear-btn" disabled={searching} onClick={clear}>Clear</button>
            )}
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <section className="poi-photo-card" aria-labelledby="poi-photo-title">
      {children
        ? (
          <div className="poi-find-grid">
            {photoSearch}
            <div className="poi-find-registry">
              <div>
                <div className="poi-results-title">Search the registry</div>
                <p className="poi-photo-copy">
                  Filter the enrolled profiles below by name, ID, alias, case ID or organisation, and by type and status.
                </p>
              </div>
              {children}
            </div>
          </div>
        )
        : photoSearch}

      {error && <div className="poi-notice error" role="alert">{error}</div>}

      {result && (
        <div className="poi-photo-results" aria-live="polite">
          <div className="poi-results-head">
            <div className="poi-results-meta">
              {results.length} candidate{results.length === 1 ? '' : 's'}
              {threshold != null ? ` · ${atThreshold} at or above ${threshold.toFixed(2)}` : ''}
            </div>
            <div className="poi-results-meta">
              Embedding and index search {Number(result.search_ms ?? 0).toFixed(1)} ms
            </div>
          </div>

          {results.length === 0
            ? <div className="poi-photo-empty">No enrolled people to compare against.</div>
            : (
              <ol className="poi-photo-list">
                {results.map((match, index) => {
                  const above = threshold != null && match.similarity >= threshold
                  const width = Math.max(0, Math.min(1, match.similarity)) * 100
                  return (
                    <li key={`${match.person_id}-${match.embedding_id}`} className={`poi-photo-row ${above ? 'above' : ''}`}>
                      <span className="poi-photo-rank">{index + 1}</span>
                      <div className="poi-photo-who">
                        <div className="poi-card-name">{match.name}</div>
                        <div className="poi-card-id">{match.person_id}</div>
                        <div className="poi-photo-seen">
                          {match.camera
                            ? `Last seen on ${match.camera}, ${formatDateTime(match.timestamp)}`
                            : 'No recognition event yet'}
                        </div>
                      </div>
                      <div className="poi-photo-score">
                        <div className="poi-photo-sim">{match.similarity.toFixed(3)}</div>
                        <div className="poi-photo-bar" aria-hidden="true">
                          <span style={{ width: `${width}%` }} />
                          {threshold != null && <i style={{ left: `${threshold * 100}%` }} />}
                        </div>
                        <span className={`poi-pill ${above ? 'status-active' : 'status-inactive'}`}>
                          {above ? 'At or above threshold' : 'Below threshold'}
                        </span>
                      </div>
                      {peopleById.has(match.person_id) && (
                        <button type="button" className="poi-photo-show" onClick={() => onShowProfile?.(match.person_id)}>
                          Show profile
                        </button>
                      )}
                    </li>
                  )
                })}
              </ol>
            )}
        </div>
      )}
    </section>
  )
}
