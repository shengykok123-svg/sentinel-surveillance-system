-- =============================================================================
-- Intelligent Surveillance System: Database Schema

-- ---------------------------------------------------------------------------
-- EXTENSIONS
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
-- pg_partman is optional; enables automated partition management.
-- Install: https://github.com/pgpartman/pg_partman
-- CREATE EXTENSION IF NOT EXISTS "pg_partman";


-- ===========================================================================
-- SECTION 1: AUTHENTICATION & ACCESS CONTROL
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 1.1 USERS
-- ---------------------------------------------------------------------------
CREATE TABLE users (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    username            VARCHAR(64)     NOT NULL,
    email               VARCHAR(255)    NOT NULL,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    is_locked           BOOLEAN         NOT NULL DEFAULT FALSE,
    account_locked_reason VARCHAR(128)  NULL,                   -- #8
    failed_login_count  INT             NOT NULL DEFAULT 0,
    locked_until        TIMESTAMPTZ     NULL,
    last_login_at       TIMESTAMPTZ     NULL,
    deleted_at          TIMESTAMPTZ     NULL,                   -- #2
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- #1: Case-insensitive uniqueness (only among non-deleted rows)
CREATE UNIQUE INDEX idx_users_email_ci
    ON users (LOWER(email))
    WHERE deleted_at IS NULL;

CREATE UNIQUE INDEX idx_users_username_ci
    ON users (LOWER(username))
    WHERE deleted_at IS NULL;

-- ---------------------------------------------------------------------------
-- 1.2 USER_CREDENTIALS
-- ---------------------------------------------------------------------------
CREATE TABLE user_credentials (
    id                      UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id                 UUID            NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    password_hash           TEXT            NOT NULL,
    hash_algorithm          VARCHAR(32)     NOT NULL DEFAULT 'argon2id',
    password_changed_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    must_change_password    BOOLEAN         NOT NULL DEFAULT FALSE
);

-- ---------------------------------------------------------------------------
-- 1.3 ROLES
-- ---------------------------------------------------------------------------
CREATE TABLE roles (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(64)     NOT NULL UNIQUE,
    description     TEXT            NULL,
    is_system_role  BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO roles (name, description, is_system_role) VALUES
    ('admin',    'Full system access including user management and configuration', TRUE),
    ('operator', 'Can perform searches, view detections, and manage enrolled persons', TRUE),
    ('viewer',   'Read-only access to detection events and camera feeds', TRUE);

-- ---------------------------------------------------------------------------
-- 1.4 PERMISSIONS
-- resource examples : 'search_query', 'enrolled_person', 'camera', 'user'
-- action examples   : 'create', 'read', 'update', 'delete', 'export'
-- ---------------------------------------------------------------------------
CREATE TABLE permissions (
    id          UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource    VARCHAR(64)     NOT NULL,
    action      VARCHAR(32)     NOT NULL,
    description TEXT            NULL,
    UNIQUE (resource, action)
);

INSERT INTO permissions (resource, action, description) VALUES
    ('enrolled_person',  'create', 'Enroll a new person into the system'),
    ('enrolled_person',  'read',   'View enrolled person records'),
    ('enrolled_person',  'update', 'Edit enrolled person details'),
    ('enrolled_person',  'delete', 'Remove an enrolled person'),
    ('search_query',     'create', 'Initiate a person-of-interest search'),
    ('search_query',     'read',   'View past search queries and results'),
    ('detection_event',  'read',   'View face detection events'),
    ('detection_event',  'delete', 'Delete detection event records'),
    ('camera',           'create', 'Add a new camera to the system'),
    ('camera',           'read',   'View camera feeds and configurations'),
    ('camera',           'update', 'Edit camera settings'),
    ('camera',           'delete', 'Remove a camera'),
    ('user',             'create', 'Create new user accounts'),
    ('user',             'read',   'View user accounts'),
    ('user',             'update', 'Edit user account details'),
    ('user',             'delete', 'Deactivate or remove user accounts'),
    ('audit_log',        'read',   'View system audit logs'),
    ('report',           'export', 'Export system reports');

-- ---------------------------------------------------------------------------
-- 1.5 ROLE_PERMISSIONS
-- ---------------------------------------------------------------------------
CREATE TABLE role_permissions (
    id              UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_id         UUID    NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id   UUID    NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    UNIQUE (role_id, permission_id)
);

-- ---------------------------------------------------------------------------
-- 1.6 USER_ROLES
-- ---------------------------------------------------------------------------
CREATE TABLE user_roles (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id         UUID            NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_by     UUID            NOT NULL REFERENCES users(id),
    assigned_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ     NULL,
    UNIQUE (user_id, role_id)
);

-- ---------------------------------------------------------------------------
-- 1.7 USER_SESSIONS
-- ---------------------------------------------------------------------------
CREATE TABLE user_sessions (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token_hash  TEXT            NOT NULL UNIQUE,
    ip_address          INET            NULL,
    user_agent          TEXT            NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_active_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ     NOT NULL,
    is_revoked          BOOLEAN         NOT NULL DEFAULT FALSE,
    revoked_at          TIMESTAMPTZ     NULL,                   -- #7
    revoked_reason      VARCHAR(128)    NULL,                   -- #7
    invalidated_by      UUID            NULL REFERENCES users(id), -- #7
    CONSTRAINT chk_session_expiry CHECK (expires_at > created_at) -- #8
);

-- ---------------------------------------------------------------------------
-- 1.8 REFRESH_TOKENS
-- ---------------------------------------------------------------------------
CREATE TABLE refresh_tokens (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id      UUID            NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    token_hash      TEXT            NOT NULL UNIQUE,
    issued_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ     NOT NULL,
    is_revoked      BOOLEAN         NOT NULL DEFAULT FALSE,
    revoked_at      TIMESTAMPTZ     NULL,
    revoked_reason  VARCHAR(128)    NULL,                       -- #7
    CONSTRAINT chk_refresh_token_expiry CHECK (expires_at > issued_at) -- #8
);

-- #8: Prevent UPDATE that un-revokes a refresh token
CREATE OR REPLACE FUNCTION trg_prevent_token_unrevoke()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.is_revoked = TRUE AND NEW.is_revoked = FALSE THEN
        RAISE EXCEPTION 'A revoked refresh token cannot be reinstated.';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_refresh_tokens_no_unrevoke
    BEFORE UPDATE ON refresh_tokens
    FOR EACH ROW EXECUTE FUNCTION trg_prevent_token_unrevoke();

-- ---------------------------------------------------------------------------
-- 1.9 LOGIN_ATTEMPTS
-- status         : 'success', 'failed', 'locked'
-- failure_reason : 'wrong_password', 'account_locked', 'mfa_failed', etc.
-- ---------------------------------------------------------------------------
CREATE TABLE login_attempts (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NULL REFERENCES users(id) ON DELETE SET NULL,
    ip_address      INET            NULL,
    status          VARCHAR(16)     NOT NULL CHECK (status IN ('success', 'failed', 'locked')),
    failure_reason  VARCHAR(64)     NULL,
    attempted_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 1.10 PASSWORD_RESET_TOKENS
-- ---------------------------------------------------------------------------
CREATE TABLE password_reset_tokens (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      TEXT            NOT NULL UNIQUE,
    issued_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ     NOT NULL,
    is_used         BOOLEAN         NOT NULL DEFAULT FALSE,
    used_at         TIMESTAMPTZ     NULL,
    invalidated_by  UUID            NULL REFERENCES users(id), -- #7
    CONSTRAINT chk_reset_token_expiry CHECK (expires_at > issued_at) -- #8
);

-- #8: Prevent replay — once a reset token is used, block re-use at DB level
CREATE OR REPLACE FUNCTION trg_prevent_reset_token_reuse()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.is_used = TRUE AND NEW.is_used = FALSE THEN
        RAISE EXCEPTION 'A used password reset token cannot be reactivated.';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_reset_tokens_no_reuse
    BEFORE UPDATE ON password_reset_tokens
    FOR EACH ROW EXECUTE FUNCTION trg_prevent_reset_token_reuse();

-- ---------------------------------------------------------------------------
-- 1.11 MFA_CONFIGS
-- ---------------------------------------------------------------------------
CREATE TABLE mfa_configs (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mfa_type            VARCHAR(16)     NOT NULL CHECK (mfa_type IN ('totp', 'sms', 'email_otp')),
    secret_encrypted    TEXT            NOT NULL,
    is_enabled          BOOLEAN         NOT NULL DEFAULT FALSE,
    enabled_at          TIMESTAMPTZ     NULL,
    last_used_at        TIMESTAMPTZ     NULL,
    UNIQUE (user_id, mfa_type)
);


-- ===========================================================================
-- SECTION 2: SURVEILLANCE CORE
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 2.1 ENROLLED_PERSONS

-- DELETION STRATEGY (important — read before deleting any row):
--   Hard delete  : DELETE FROM enrolled_persons WHERE id = ?
--                  This cascades to face_embeddings (embeddings lost) and
--                  sets detection_events.person_id = NULL, tracks.person_id = NULL,
--                  search_queries.target_person_id = NULL.
--                  Use only when required by privacy/right-to-erasure requests.
--   Soft delete  : UPDATE enrolled_persons SET deleted_at = NOW(), is_active = FALSE
--                  WHERE id = ?
--                  Preserves embeddings and detection history for audit purposes.
--                  Recommended for routine deactivation.
--   Before hard-deleting, ensure FAISS index entry is also removed
--   (see Section 6 for the FAISS sync trigger hook).
-- ---------------------------------------------------------------------------
CREATE TABLE enrolled_persons (
    id                      UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name               VARCHAR(255)    NOT NULL,
    person_type             VARCHAR(32)     NOT NULL CHECK (person_type IN ('staff', 'vip', 'watchlist', 'visitor')),
    notes                   TEXT            NULL,
    is_active               BOOLEAN         NOT NULL DEFAULT TRUE,
    enrolled_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    enrolled_by             UUID            NOT NULL REFERENCES users(id),
    retention_expires_at    TIMESTAMPTZ     NULL,
    embedding_only_mode     BOOLEAN         NOT NULL DEFAULT TRUE,
    deleted_at              TIMESTAMPTZ     NULL                    -- #2
);

-- ---------------------------------------------------------------------------
-- 2.2 FACE_EMBEDDINGS
-- ---------------------------------------------------------------------------
CREATE TABLE face_embeddings (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id           UUID            NOT NULL REFERENCES enrolled_persons(id) ON DELETE CASCADE,
    faiss_index_id      INT             NULL UNIQUE,
    embedding_vector    vector(512)     NOT NULL,
    source_type         VARCHAR(32)     NOT NULL CHECK (source_type IN ('manual_upload', 'camera_capture', 'batch_import')),
    quality_score       FLOAT           NULL CHECK (quality_score BETWEEN 0.0 AND 1.0),
    is_primary          BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ     NULL
);

CREATE UNIQUE INDEX idx_face_embeddings_primary
    ON face_embeddings (person_id)
    WHERE is_primary = TRUE;

-- #4: pgvector IVFFlat index — tune lists parameter per note above
CREATE INDEX idx_face_embeddings_vector
    ON face_embeddings USING ivfflat (embedding_vector vector_cosine_ops)
    WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- 2.3 CAMERAS
--   last_frame_at  : Timestamp of the last successfully decoded frame
--   stream_fps     : Observed FPS from the RTSP stream (updated periodically)
--   error_message  : Last error reported by the stream consumer (null = healthy)
-- ---------------------------------------------------------------------------
CREATE TABLE cameras (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(128)    NOT NULL,
    location        VARCHAR(255)    NULL,
    rtsp_url        TEXT            NOT NULL,
    status          VARCHAR(16)     NOT NULL DEFAULT 'inactive'
                        CHECK (status IN ('active', 'inactive', 'error')),
    config          JSONB           NULL DEFAULT '{}',
    last_active_at  TIMESTAMPTZ     NULL,
    last_frame_at   TIMESTAMPTZ     NULL,                       -- #5
    stream_fps      FLOAT           NULL,                       -- #5
    error_message   TEXT            NULL,                       -- #5
    deleted_at      TIMESTAMPTZ     NULL,                       -- #2
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ===========================================================================
-- SECTION 3: DETECTION EVENTS (PARTITIONED)
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 3.1 DETECTION_EVENTS (range-partitioned by detected_at)
--
--   Partition management options:
--     Manual  : Create partitions by hand each month (see examples below).
--     Automated: Use pg_partman with run_maintenance() on a cron job.
--       SELECT partman.create_parent(
--           p_parent_table  => 'public.detection_events',
--           p_control       => 'detected_at',
--           p_interval      => '1 month',
--           p_premake       => 3
--       );
-- ---------------------------------------------------------------------------
CREATE TABLE detection_events (
    id                      UUID            NOT NULL DEFAULT uuid_generate_v4(),
    camera_id               UUID            NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    person_id               UUID            NULL REFERENCES enrolled_persons(id) ON DELETE SET NULL,
    face_embedding          vector(512)     NULL,
    confidence_score        FLOAT           NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    detection_confidence    FLOAT           NOT NULL CHECK (detection_confidence BETWEEN 0.0 AND 1.0),

    -- #9: Bounding box as typed integer columns
    bbox_x                  INT             NOT NULL CHECK (bbox_x >= 0),
    bbox_y                  INT             NOT NULL CHECK (bbox_y >= 0),
    bbox_w                  INT             NOT NULL CHECK (bbox_w > 0),
    bbox_h                  INT             NOT NULL CHECK (bbox_h > 0),

    detected_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    is_anonymized           BOOLEAN         NOT NULL DEFAULT FALSE,
    expires_at              TIMESTAMPTZ     NULL,

    PRIMARY KEY (id, detected_at)           -- partition key must be in PK
) PARTITION BY RANGE (detected_at);

-- Create initial partitions (add more monthly via cron or pg_partman)
CREATE TABLE detection_events_y2025m01
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE detection_events_y2025m02
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE detection_events_y2025m03
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

CREATE TABLE detection_events_y2025m04
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

CREATE TABLE detection_events_y2025m05
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

CREATE TABLE detection_events_y2025m06
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

CREATE TABLE detection_events_default
    PARTITION OF detection_events DEFAULT;

-- Indexes on partitioned table (PostgreSQL propagates to child partitions)
CREATE INDEX idx_detection_events_camera_time
    ON detection_events (camera_id, detected_at DESC);

CREATE INDEX idx_detection_events_person_id
    ON detection_events (person_id)
    WHERE person_id IS NOT NULL;

CREATE INDEX idx_detection_events_expires_at
    ON detection_events (expires_at)
    WHERE expires_at IS NOT NULL;

-- NOTE (#4): pgvector IVFFlat indexes cannot be created on partitioned tables
-- directly in PostgreSQL. Create the vector index on each child partition:
--   CREATE INDEX ON detection_events_y2025m01
--       USING ivfflat (face_embedding vector_cosine_ops) WITH (lists = 100);
-- Or use the FAISS index (via faiss_index_id in face_embeddings) as the
-- primary ANN lookup and only use pgvector for small ad-hoc searches.


-- ===========================================================================
-- SECTION 4: TRACKING
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 4.1 TRACKS
-- status          : 'tentative', 'confirmed', 'deleted'
-- trajectory      : JSON array [{frame, x, y, w, h}, ...] for movement path
-- ---------------------------------------------------------------------------
CREATE TABLE tracks (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id           UUID            NULL REFERENCES enrolled_persons(id) ON DELETE SET NULL,
    camera_id           UUID            NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    deepsort_track_id   INT             NOT NULL,
    status              VARCHAR(16)     NOT NULL DEFAULT 'tentative'
                            CHECK (status IN ('tentative', 'confirmed', 'deleted')),
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    frame_count         INT             NOT NULL DEFAULT 0,
    trajectory          JSONB           NULL DEFAULT '[]',
    UNIQUE (camera_id, deepsort_track_id)
);

-- ---------------------------------------------------------------------------
-- 4.2 TRACK_DETECTION_EVENTS
-- Junction table: links detection events to their parent track.
-- ---------------------------------------------------------------------------
CREATE TABLE track_detection_events (
    id          UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    track_id    UUID    NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    event_id    UUID    NOT NULL,                   -- FK to detection_events (partitioned)
    event_time  TIMESTAMPTZ NOT NULL,               -- required for partition routing
    UNIQUE (track_id, event_id)
    -- Note: FK to partitioned detection_events is not directly supported in
    -- PostgreSQL. Enforce referential integrity in the backend service layer.
);

-- ---------------------------------------------------------------------------
-- 4.3 CROSS_CAMERA_EVENTS
-- Records person transitions detected by the re-ID network across cameras.
-- similarity_score is cosine similarity output from the re-ID model.
-- ---------------------------------------------------------------------------
CREATE TABLE cross_camera_events (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_track_id       UUID            NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    to_track_id         UUID            NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    from_camera_id      UUID            NOT NULL REFERENCES cameras(id),
    to_camera_id        UUID            NOT NULL REFERENCES cameras(id),
    similarity_score    FLOAT           NOT NULL CHECK (similarity_score BETWEEN 0.0 AND 1.0),
    transition_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);


-- ===========================================================================
-- SECTION 5: PERSON-OF-INTEREST SEARCH
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 5.1 SEARCH_QUERIES
-- ---------------------------------------------------------------------------
CREATE TABLE search_queries (
    id                      UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id                 UUID            NOT NULL REFERENCES users(id),
    target_person_id        UUID            NULL REFERENCES enrolled_persons(id) ON DELETE SET NULL,
    target_person_name      VARCHAR(255)    NULL,               -- #11: snapshot at search time
    status                  VARCHAR(16)     NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    cameras_searched        INT             NOT NULL DEFAULT 0,
    results_count           INT             NOT NULL DEFAULT 0,
    threshold_used          FLOAT           NOT NULL CHECK (threshold_used BETWEEN 0.0 AND 1.0),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ     NULL,
    expires_at              TIMESTAMPTZ     NULL                -- #10
);

-- ---------------------------------------------------------------------------
-- 5.2 SEARCH_RESULTS
-- ---------------------------------------------------------------------------
CREATE TABLE search_results (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id            UUID            NOT NULL REFERENCES search_queries(id) ON DELETE CASCADE,
    event_id            UUID            NOT NULL,               -- FK to detection_events (partitioned)
    event_time          TIMESTAMPTZ     NOT NULL,               -- required for partition routing
    similarity_score    FLOAT           NOT NULL CHECK (similarity_score BETWEEN 0.0 AND 1.0),
    found_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);


-- ===========================================================================
-- SECTION 6: AUDIT LOGGING
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 6.1 AUDIT_LOGS
--
-- Improvement #10 : expires_at for configurable log retention
--   Recommended retention: 90 days for session logs, 1 year for security events.
--   Purge job: DELETE FROM audit_logs WHERE expires_at < NOW();
--
-- Audit logs are append-only (no UPDATE, no DELETE).
-- ---------------------------------------------------------------------------
CREATE TABLE audit_logs (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NULL REFERENCES users(id) ON DELETE SET NULL,
    session_id      UUID            NULL REFERENCES user_sessions(id) ON DELETE SET NULL,
    action_type     VARCHAR(64)     NOT NULL,
    resource_type   VARCHAR(64)     NULL,
    resource_id     UUID            NULL,
    metadata        JSONB           NULL DEFAULT '{}',
    ip_address      INET            NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ     NULL                        -- #10
);

CREATE RULE audit_logs_no_update AS ON UPDATE TO audit_logs DO INSTEAD NOTHING;
CREATE RULE audit_logs_no_delete AS ON DELETE TO audit_logs DO INSTEAD NOTHING;

-- ---------------------------------------------------------------------------
-- 6.2 FAISS SYNC NOTIFICATION HOOK  (#14)
--
-- When a face_embeddings row is deleted, the backend FAISS sync service must
-- remove the corresponding vector from the FAISS index using faiss_index_id.
-- This trigger fires pg_notify so the sync service can react asynchronously.
-- The backend should LISTEN on channel 'faiss_sync' and process the payload.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION trg_faiss_sync_on_delete()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.faiss_index_id IS NOT NULL THEN
        PERFORM pg_notify(
            'faiss_sync',
            json_build_object(
                'event',          'delete',
                'faiss_index_id', OLD.faiss_index_id,
                'person_id',      OLD.person_id
            )::text
        );
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_face_embeddings_faiss_delete
    AFTER DELETE ON face_embeddings
    FOR EACH ROW EXECUTE FUNCTION trg_faiss_sync_on_delete();


-- ===========================================================================
-- SECTION 7: INDEXES
-- ===========================================================================

-- Auth & sessions
CREATE INDEX idx_user_sessions_user_id          ON user_sessions (user_id);
CREATE INDEX idx_user_sessions_expires_active    ON user_sessions (expires_at) WHERE is_revoked = FALSE;
CREATE INDEX idx_refresh_tokens_session_id       ON refresh_tokens (session_id);
CREATE INDEX idx_login_attempts_user_time        ON login_attempts (user_id, attempted_at DESC);
CREATE INDEX idx_login_attempts_ip_time          ON login_attempts (ip_address, attempted_at DESC);
CREATE INDEX idx_password_reset_active           ON password_reset_tokens (user_id) WHERE is_used = FALSE;
CREATE INDEX idx_user_roles_user_id              ON user_roles (user_id);
CREATE INDEX idx_user_roles_role_id              ON user_roles (role_id);

-- Enrolled persons
CREATE INDEX idx_enrolled_persons_active         ON enrolled_persons (is_active) WHERE deleted_at IS NULL;

-- Face embeddings
CREATE INDEX idx_face_embeddings_person_id       ON face_embeddings (person_id);
CREATE INDEX idx_face_embeddings_faiss_id        ON face_embeddings (faiss_index_id);

-- Cameras
CREATE INDEX idx_cameras_status                  ON cameras (status) WHERE deleted_at IS NULL;

-- Tracking
CREATE INDEX idx_tracks_camera_status            ON tracks (camera_id, status);
CREATE INDEX idx_tracks_person_id                ON tracks (person_id) WHERE person_id IS NOT NULL;
CREATE INDEX idx_cross_camera_from_track         ON cross_camera_events (from_track_id);
CREATE INDEX idx_cross_camera_to_track           ON cross_camera_events (to_track_id);

-- Search
CREATE INDEX idx_search_queries_user_time        ON search_queries (user_id, created_at DESC);
CREATE INDEX idx_search_results_query_id         ON search_results (query_id);

-- Audit
CREATE INDEX idx_audit_logs_user_time            ON audit_logs (user_id, created_at DESC);
CREATE INDEX idx_audit_logs_session_id           ON audit_logs (session_id);
CREATE INDEX idx_audit_logs_resource             ON audit_logs (resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at           ON audit_logs (created_at DESC);

-- #12: Operational index guidance
-- Do NOT add further indexes speculatively. After deployment, identify
-- slow queries using EXPLAIN ANALYZE and pg_stat_statements, then add
-- targeted indexes only for confirmed high-frequency query patterns.
-- Excess indexes degrade INSERT/UPDATE performance on high-write tables
-- like detection_events and tracks.


-- ===========================================================================
-- SECTION 8: UTILITY TRIGGERS
-- ===========================================================================

-- Auto-update updated_at on users
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();

-- #8: Prevent locking a user with no reason recorded
CREATE OR REPLACE FUNCTION trg_require_lock_reason()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_locked = TRUE AND NEW.account_locked_reason IS NULL THEN
        RAISE EXCEPTION 'account_locked_reason must be set when locking a user account.';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_lock_reason
    BEFORE UPDATE ON users
    FOR EACH ROW
    WHEN (NEW.is_locked = TRUE AND OLD.is_locked = FALSE)
    EXECUTE FUNCTION trg_require_lock_reason();

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
