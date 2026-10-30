// Pure helpers behind useSystemStats.
//
// The POI list is fetched rarely, so its live status and last sighting are
// rebuilt in the browser from the pushed active tracks and recognition
// events. The rules mirror the backend's _build_live_track_lookup and
// _build_person_read: a recognised live track makes a person active and
// supplies the last sighting; otherwise the newest known sighting is used.

// People and embedding stats are refetched at most this often while the
// live socket is connected.
export const SLOW_REFRESH_MS = 60000

const SLOW_ENDPOINTS = ['people', 'embeddings']

function toTime(value) {
  if (value == null || value === '') return NaN
  return new Date(value).getTime()
}

// True when `candidate` is a valid time later than `current` (an invalid or
// missing `current` loses to any valid candidate).
function isNewer(candidate, current) {
  const candidateTime = toTime(candidate)
  if (Number.isNaN(candidateTime)) return false
  const currentTime = toTime(current)
  return Number.isNaN(currentTime) || candidateTime > currentTime
}

/** Map of person_id -> newest recognised live track for that person. */
export function liveTrackLookup(activeTracks) {
  const lookup = new Map()
  for (const track of activeTracks ?? []) {
    if (track?.status !== 'recognized' || !track.person_id) continue
    const key = String(track.person_id)
    const current = lookup.get(key)
    if (!current || isNewer(track.last_seen, current.last_seen)) lookup.set(key, track)
  }
  return lookup
}

function withLiveFields(person, status, camera, time) {
  if (
    person.status === status &&
    (person.last_seen_camera ?? null) === camera &&
    (person.last_seen_time ?? null) === time
  ) {
    return person
  }
  return { ...person, status, last_seen_camera: camera, last_seen_time: time }
}

/**
 * Apply live status and the latest sighting to the fetched people.
 * `sightings` is the Map kept by recordSightings. Unchanged people are
 * returned as the same object.
 */
export function mergeLivePeople(people, activeTracks, sightings) {
  const live = liveTrackLookup(activeTracks)
  return (people ?? []).map((person) => {
    const key = String(person.id)
    const track = live.get(key)
    if (track) {
      return withLiveFields(person, 'active', track.camera_name ?? null, track.last_seen ?? null)
    }
    let camera = person.last_seen_camera ?? null
    let time = person.last_seen_time ?? null
    const sighting = sightings?.get(key)
    if (sighting && isNewer(sighting.time, time)) {
      camera = sighting.camera ?? null
      time = sighting.time
    }
    return withLiveFields(person, 'inactive', camera, time)
  })
}

/**
 * Record the newest sighting per person from recognised live tracks and/or a
 * recognised pushed event. Returns the same Map when nothing got newer.
 */
export function recordSightings(sightings, { activeTracks, event } = {}) {
  const base = sightings ?? new Map()
  let next = base
  const note = (personId, camera, time) => {
    if (!personId || Number.isNaN(toTime(time))) return
    const key = String(personId)
    const current = next.get(key)
    if (current && !isNewer(time, current.time)) return
    if (next === base) next = new Map(base)
    next.set(key, { camera: camera ?? null, time })
  }
  for (const track of activeTracks ?? []) {
    if (track?.status === 'recognized') note(track.person_id, track.camera_name, track.last_seen)
  }
  if (event?.status === 'recognized') note(event.person_id, event.camera, event.timestamp)
  return next
}

/**
 * Decide which endpoints one refresh should request.
 *
 * - Socket down: everything, as the pushed data is missing.
 * - Socket up: cameras and tracks never (pushed every second); events only
 *   on a forced refresh or the first refresh since the socket came up
 *   (catch-up); people and embedding stats then too, or once each has gone
 *   SLOW_REFRESH_MS without a successful load.
 *
 * `lastLoad` holds the Date.now() of each slow endpoint's last success.
 */
export function planRefresh({ connected, wasConnected = false, now = Date.now(), lastLoad = {}, force = false }) {
  if (!connected) {
    return { cameras: true, tracks: true, events: true, people: true, embeddings: true }
  }
  const full = force || !wasConnected
  const plan = { cameras: false, tracks: false, events: full }
  for (const name of SLOW_ENDPOINTS) {
    const last = lastLoad?.[name]
    plan[name] = full || last == null || now - last >= SLOW_REFRESH_MS
  }
  return plan
}
