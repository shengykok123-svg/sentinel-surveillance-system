const API_BASE = import.meta.env.VITE_API_URL ?? '/api'
const WS_BASE =
  import.meta.env.VITE_WS_URL ??
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`
const TOKEN_STORAGE_KEY = 'sentinel.access_token'

let accessToken = window.localStorage.getItem(TOKEN_STORAGE_KEY)
let unauthorizedHandler = null

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export function getAccessToken() {
  return accessToken
}

export function setAccessToken(token) {
  accessToken = token
  if (token) {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, token)
    return
  }
  window.localStorage.removeItem(TOKEN_STORAGE_KEY)
}

export function clearAccessToken() {
  setAccessToken('')
}

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler
  return () => {
    if (unauthorizedHandler === handler) {
      unauthorizedHandler = null
    }
  }
}

async function request(path, options = {}) {
  // Bound reads and stream-ticket waits so a stalled connection cannot leave
  // polling or camera setup pending forever. Long-running writes keep their
  // existing lifetime; aborting a fetch would not cancel server-side work.
  const timeoutMs = options.timeoutMs ?? ((options.method ?? 'GET') === 'GET' || path.endsWith('/ticket') ? 15000 : 0)
  const controller = new AbortController()
  const abort = () => controller.abort(options.signal?.reason)
  if (options.signal?.aborted) abort()
  else options.signal?.addEventListener('abort', abort, { once: true })
  const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(new DOMException('Request timed out', 'TimeoutError')), timeoutMs) : null
  try {
  const headers = new Headers(options.headers ?? {})
  if (accessToken && !options.skipAuth) {
    headers.set('Authorization', `Bearer ${accessToken}`)
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    signal: controller.signal,
  })

  const contentType = res.headers.get('content-type') ?? ''
  let payload = null
  if (res.status !== 204) {
    if (contentType.includes('application/json')) {
      payload = await res.json()
    } else {
      payload = await res.text()
    }
  }

  if (!res.ok) {
    if (res.status === 401 && typeof unauthorizedHandler === 'function' && !options.skipUnauthorizedHandler) {
      unauthorizedHandler()
    }
    const detail = payload?.detail ?? payload ?? res.statusText
    const message = typeof detail === 'string'
      ? detail
      : typeof detail?.message === 'string'
        ? detail.message
        : JSON.stringify(detail)
    throw new ApiError(`[${res.status}] ${path}: ${message}`, res.status, detail)
  }

  if (res.status === 204) {
    return null
  }

  return payload
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', abort)
  }
}

function jsonRequest(method, path, body, options = {}) {
  return request(path, {
    ...options,
    method,
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
}

export function getStreamUrl(cameraId) {
  return `${API_BASE}/stream/${encodeURIComponent(cameraId)}`
}

export async function getAuthenticatedStreamUrl(cameraId, signal) {
  const { ticket } = await request(`/stream/${encodeURIComponent(cameraId)}/ticket`, { method: 'POST', signal })
  return `${getStreamUrl(cameraId)}?ticket=${encodeURIComponent(ticket)}`
}

export function reportExpiredSession(token) {
  if (token === accessToken) unauthorizedHandler?.()
}

export function getAuthSetupState() {
  return request('/auth/setup-state', { skipAuth: true, skipUnauthorizedHandler: true })
}

export async function login(identifier, password) {
  const response = await jsonRequest(
    'POST',
    '/auth/login',
    { identifier, password },
    { skipAuth: true, skipUnauthorizedHandler: true }
  )
  setAccessToken(response.access_token)
  return response
}

export function getCurrentUser() {
  return request('/auth/me')
}

export function logout() {
  return request('/auth/logout', { method: 'POST' })
}

export function requestSensitiveAccess(password) {
  return jsonRequest('POST', '/auth/sensitive-access', { password }, { skipUnauthorizedHandler: true })
}

export function bootstrapAdmin(payload) {
  return jsonRequest('POST', '/auth/bootstrap-admin', payload, {
    skipAuth: true,
    skipUnauthorizedHandler: true,
  })
}

export function getEvents(params = {}) {
  const qs = new URLSearchParams(params).toString()
  return request(`/events${qs ? `?${qs}` : ''}`)
}

export function clearEvents() {
  return request('/events', { method: 'DELETE' })
}

export function getRecognitionRecords(params = {}) {
  const qs = new URLSearchParams(params).toString()
  return request(`/recognition-records${qs ? `?${qs}` : ''}`)
}

export async function getRecognitionRecordSnapshot(eventId) {
  const headers = new Headers()
  if (accessToken) {
    headers.set('Authorization', `Bearer ${accessToken}`)
  }
  const res = await fetch(`${API_BASE}/recognition-records/${eventId}/snapshot`, { headers })
  if (!res.ok) {
    if (res.status === 401 && typeof unauthorizedHandler === 'function') {
      unauthorizedHandler()
    }
    throw new ApiError(`[${res.status}] recognition record image unavailable`, res.status, await res.text())
  }
  return res.blob()
}

export function getResources() {
  return request('/resources')
}

export function getCameras() {
  return request('/cameras')
}

export function createCamera(payload) {
  return jsonRequest('POST', '/cameras', payload)
}

export function updateCamera(cameraId, payload) {
  return jsonRequest('PUT', `/cameras/${cameraId}`, payload)
}

export function deleteCamera(cameraId) {
  return request(`/cameras/${cameraId}`, { method: 'DELETE' })
}

export function getPOI() {
  return request('/poi')
}

export function enrollPerson(formData, jobId) {
  const suffix = jobId ? `?job_id=${encodeURIComponent(jobId)}` : ''
  return request(`/poi/enroll${suffix}`, { method: 'POST', body: formData })
}

export function getEnrollmentJob(jobId, signal) {
  return request(`/poi/enrollment-jobs/${encodeURIComponent(jobId)}`, { signal })
}

export function deletePOI(personId) {
  return request(`/poi/${personId}`, { method: 'DELETE' })
}

export function updatePOI(personId, payload) {
  return jsonRequest('PUT', `/poi/${personId}`, payload)
}

export function assessImageQuality(file, signal) {
  const body = new FormData()
  body.append('image', file)
  return request('/poi/assess-quality', { method: 'POST', body, signal })
}

export function getPOIReferences(personId, sensitiveAccessToken) {
  return request(`/poi/${personId}/references`, {
    headers: { 'X-Sensitive-Access': sensitiveAccessToken },
    skipUnauthorizedHandler: true,
  })
}

export function generatePOIReferences(personId, sensitiveAccessToken) {
  return request(`/poi/${personId}/generate-references`, {
    method: 'POST', headers: { 'X-Sensitive-Access': sensitiveAccessToken },
    skipUnauthorizedHandler: true,
  })
}

export async function addPOIReferences(personId, files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('images', file))
  return request(`/poi/${personId}/references`, { method: 'POST', body: formData })
}

export async function getPOIReferenceImage(personId, referenceId, sensitiveAccessToken) {
  const headers = new Headers()
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  if (sensitiveAccessToken) headers.set('X-Sensitive-Access', sensitiveAccessToken)
  const res = await fetch(`${API_BASE}/poi/${personId}/references/${referenceId}/image`, { headers })
  if (!res.ok) {
    throw new ApiError(`[${res.status}] reference image unavailable`, res.status, await res.text())
  }
  return res.blob()
}

export async function searchFace(imageFile, topK = 5) {
  const form = new FormData()
  form.append('image', imageFile)
  return request(`/search?top_k=${topK}`, { method: 'POST', body: form })
}

export function getEmbeddingStats() {
  return request('/embeddings/stats')
}

export function getActiveTracks() {
  return request('/tracks/active')
}

export function getEventsWebSocketUrl() {
  return `${WS_BASE}/events`
}

// ── Analytics ────────────────────────────────────────────────────────
//
// Events are stored in UTC, so every analytics request carries the viewer's
// offset from UTC in minutes. Without it, "activity by hour of day" would be
// reported against UTC boundaries and a dashboard in UTC+8 would show its
// peaks eight hours off.

function localTzOffset() {
  // getTimezoneOffset() counts minutes *west* of UTC; the API wants east.
  return -new Date().getTimezoneOffset()
}

function analyticsRequest(path, params = {}) {
  const qs = new URLSearchParams({ tz_offset: localTzOffset(), ...params }).toString()
  return request(`/analytics${path}?${qs}`)
}

export function getAnalyticsSummary(range = '24h') {
  return analyticsRequest('/summary', { range })
}

export function getAnalyticsTimeline(range = '7d', bucket = 'hour') {
  return analyticsRequest('/timeline', { range, bucket })
}

export function getSimilarityHistogram(range = '7d', bins = 20) {
  return analyticsRequest('/similarity-histogram', { range, bins })
}

export function getFaceSizeQuality(range = '7d') {
  return analyticsRequest('/face-size-quality', { range })
}

export function getCameraAnalytics(range = '7d') {
  return analyticsRequest('/cameras', { range })
}

export function getTopPersons(range = '7d', limit = 10) {
  return analyticsRequest('/persons/top', { range, limit })
}

export const getSettings = () => request('/settings')
export const saveRecognitionSettings = (payload) => jsonRequest('PUT', '/settings/recognition', payload)
export const changePassword = (payload) => jsonRequest('POST', '/auth/change-password', payload)

// ── Live frame stream ───────────────────────────────────────────────
//
// Frames travel over a WebSocket so they do not compete with API calls for
// the browser's six-per-origin HTTP/1.1 connections the way MJPEG did.
export async function getStreamTicket(cameraId, signal) {
  const { ticket } = await request(`/stream/${encodeURIComponent(cameraId)}/ticket`, { method: 'POST', signal })
  return ticket
}

export function getStreamWebSocketUrl(cameraId, ticket) {
  return `${WS_BASE}/stream/${encodeURIComponent(cameraId)}?ticket=${encodeURIComponent(ticket)}`
}
