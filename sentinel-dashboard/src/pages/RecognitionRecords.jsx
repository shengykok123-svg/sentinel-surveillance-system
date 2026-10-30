import { useEffect, useMemo, useState } from 'react'
import { getRecognitionRecords, getRecognitionRecordSnapshot } from '../utils/api'
import './RecognitionRecords.css'

function titleCase(value) {
  const text = String(value ?? '').trim()
  return text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : '--'
}

function formatTimestamp(value) {
  if (!value) return '--'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? '--' : date.toLocaleString()
}

function Snapshot({ record }) {
  const [imageUrl, setImageUrl] = useState(null)
  const [unavailable, setUnavailable] = useState(false)

  useEffect(() => {
    let active = true
    let objectUrl = null
    setImageUrl(null)
    setUnavailable(false)
    if (!record?.has_snapshot) {
      setUnavailable(true)
      return undefined
    }
    getRecognitionRecordSnapshot(record.id)
      .then((blob) => {
        if (!active) return
        objectUrl = URL.createObjectURL(blob)
        setImageUrl(objectUrl)
      })
      .catch(() => {
        if (active) setUnavailable(true)
      })
    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [record?.has_snapshot, record?.id])

  if (unavailable) {
    return <div className="rr-image-empty">Snapshot unavailable for this record.</div>
  }
  if (!imageUrl) {
    return <div className="rr-image-empty">Loading evidence image…</div>
  }
  return <img className="rr-evidence-image" src={imageUrl} alt={`Recognition evidence for ${record.person_name ?? 'person'}`} />
}

export default function RecognitionRecords({ focusEventId }) {
  const [records, setRecords] = useState([])
  const [selectedId, setSelectedId] = useState(focusEventId ?? null)
  const [query, setQuery] = useState('')
  const [camera, setCamera] = useState('all')
  const [type, setType] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    setLoading(true)
    getRecognitionRecords({ limit: 200 })
      .then((payload) => {
        if (!active) return
        setRecords(payload.records ?? [])
        setError('')
      })
      .catch(() => {
        if (active) setError('Recognition records could not be loaded.')
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (focusEventId) setSelectedId(focusEventId)
  }, [focusEventId])

  const cameras = useMemo(
    () => [...new Set(records.map((record) => record.camera).filter(Boolean))].sort(),
    [records],
  )
  const types = useMemo(
    () => [...new Set(records.map((record) => record.person_type).filter(Boolean))].sort(),
    [records],
  )
  const filteredRecords = useMemo(() => {
    const search = query.trim().toLowerCase()
    return records.filter((record) => {
      const text = [record.person_name, record.person_type, record.camera, record.track_id, record.time]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
      return (!search || text.includes(search))
        && (camera === 'all' || record.camera === camera)
        && (type === 'all' || record.person_type === type)
    })
  }, [records, query, camera, type])
  const selectedRecord = records.find((record) => record.id === selectedId) ?? filteredRecords[0] ?? null

  const resetFilters = () => {
    setQuery('')
    setCamera('all')
    setType('all')
  }

  return (
    <main className="rr-page">
      <section className="rr-header-card">
        <div>
          <div className="rr-eyebrow">Evidence archive</div>
          <h1 className="rr-title">Recognition Records</h1>
          <p className="rr-subtitle">Review each successful recognition with its time, location, identity data and annotated camera evidence.</p>
        </div>
        <div className="rr-total"><strong>{records.length}</strong><span>records loaded</span></div>
      </section>

      <section className="rr-toolbar" aria-label="Recognition record filters">
        <label className="rr-search-field">
          <span>Search</span>
          <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name, camera, type or track…" />
        </label>
        <label>
          <span>Camera</span>
          <select value={camera} onChange={(event) => setCamera(event.target.value)}>
            <option value="all">All cameras</option>
            {cameras.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>Type</span>
          <select value={type} onChange={(event) => setType(event.target.value)}>
            <option value="all">All types</option>
            {types.map((item) => <option key={item} value={item}>{titleCase(item)}</option>)}
          </select>
        </label>
        <button className="rr-reset" type="button" onClick={resetFilters}>Reset</button>
      </section>

      {error && <div className="rr-notice">{error}</div>}
      {loading ? (
        <div className="rr-empty">Loading recognition records…</div>
      ) : !records.length ? (
        <div className="rr-empty">No recognition records yet. New successful recognitions will appear here with their evidence image.</div>
      ) : (
        <section className="rr-workspace">
          <aside className="rr-list-panel">
            <div className="rr-list-head">{filteredRecords.length} of {records.length} shown</div>
            <div className="rr-list">
              {filteredRecords.map((record) => (
                <button
                  key={record.id}
                  className={`rr-record-card ${selectedRecord?.id === record.id ? 'selected' : ''}`}
                  type="button"
                  onClick={() => setSelectedId(record.id)}
                >
                  <span className="rr-record-name">{record.person_name ?? 'Recognized person'}</span>
                  <span className="rr-record-meta">{titleCase(record.person_type)} · {record.camera}</span>
                  <span className="rr-record-time">{formatTimestamp(record.timestamp)}</span>
                </button>
              ))}
              {!filteredRecords.length && <div className="rr-list-empty">No records match your filters.</div>}
            </div>
          </aside>

          <section className="rr-detail-panel">
            {selectedRecord ? (
              <>
                <div className="rr-detail-head">
                  <div>
                    <div className="rr-eyebrow">Recognition detail</div>
                    <h2>{selectedRecord.person_name ?? 'Recognized person'}</h2>
                  </div>
                  <span className={`rr-type rr-type-${selectedRecord.person_type ?? 'default'}`}>{titleCase(selectedRecord.person_type)}</span>
                </div>
                <Snapshot record={selectedRecord} />
                <dl className="rr-details">
                  <div><dt>Time</dt><dd>{formatTimestamp(selectedRecord.timestamp)}</dd></div>
                  <div><dt>Location</dt><dd>{selectedRecord.camera ?? '--'}</dd></div>
                  <div><dt>Track</dt><dd>{selectedRecord.track_id ? `Track ${selectedRecord.track_id}` : '--'}</dd></div>
                  <div><dt>Similarity</dt><dd>{selectedRecord.conf ?? '--'}</dd></div>
                  <div className="rr-bbox"><dt>Bounding box</dt><dd>{selectedRecord.bbox?.join(', ') ?? '--'}</dd></div>
                </dl>
              </>
            ) : <div className="rr-empty">Select a record to review its evidence.</div>}
          </section>
        </section>
      )}
    </main>
  )
}
