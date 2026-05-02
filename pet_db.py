"""PetDB — pet plugin's SQLite layer.

Extends CortexDB with the 6 pet-specific tables (pet_state,
pet_interactions, pet_vitals_log, pet_coma_log, pet_training_history,
heartbeat_log) and the helper methods that operate on them.

Slice 2c2d — extracted from cortex-core/src/cortex_db.py so cortex.db
no longer carries pet schema. Plugin uses PetDB against its own pet.db;
core uses CortexDB (pet-free) against cortex.db.
"""

import json

from cortex_db import CortexDB


PET_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pet_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pet_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT DEFAULT '',
    sentiment_score REAL DEFAULT 0.0,
    inference_time_ms INTEGER DEFAULT 0,
    tokens_generated INTEGER DEFAULT 0,
    stage INTEGER DEFAULT 0,
    mood TEXT DEFAULT 'neutral',
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pet_interactions_created ON pet_interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_pet_interactions_session ON pet_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_pet_state_key ON pet_state(key);

CREATE TABLE IF NOT EXISTS pet_vitals_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hunger REAL DEFAULT 1.0,
    cleanliness REAL DEFAULT 1.0,
    energy REAL DEFAULT 1.0,
    happiness REAL DEFAULT 0.5,
    intelligence REAL DEFAULT 0.0,
    is_coma INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_vitals_log_created ON pet_vitals_log(created_at);

CREATE TABLE IF NOT EXISTS pet_coma_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entered_at TEXT NOT NULL DEFAULT (datetime('now')),
    exited_at TEXT,
    trigger_vitals TEXT DEFAULT '',
    duration_hours REAL DEFAULT 0,
    revival_method TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pet_training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lora_version TEXT NOT NULL,
    final_loss REAL,
    perplexity_base REAL,
    perplexity_finetuned REAL,
    intelligence_score REAL,
    training_time_s REAL,
    dataset_size INTEGER,
    deployed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS heartbeat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_type TEXT NOT NULL DEFAULT 'reflection',
    prompt TEXT NOT NULL,
    response TEXT DEFAULT '',
    sentiment_score REAL DEFAULT 0.0,
    inference_time_ms INTEGER DEFAULT 0,
    tokens_generated INTEGER DEFAULT 0,
    battery_pct INTEGER DEFAULT -1,
    is_charging INTEGER DEFAULT 0,
    hunger REAL DEFAULT 0.0,
    cleanliness REAL DEFAULT 0.0,
    energy REAL DEFAULT 0.0,
    happiness REAL DEFAULT 0.0,
    shell_commands TEXT DEFAULT '',
    shell_results TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_log_created ON heartbeat_log(created_at);
CREATE INDEX IF NOT EXISTS idx_heartbeat_log_type ON heartbeat_log(prompt_type);
"""


class PetDB(CortexDB):
    """CortexDB plus pet-specific schema and helper methods.

    The pet plugin instantiates PetDB(pet_db_path) and replaces
    self.api.db with it during on_load(). All pet runtime code
    (PetEngine, Heartbeat, plugin HTTP handlers) calls helpers like
    get_pet_state() / insert_pet_interaction() directly through this.
    """

    def __init__(self, db_path):
        super().__init__(db_path)
        self._conn.executescript(PET_SCHEMA_SQL)
        self._conn.commit()

    # --- Pet State ---

    def get_pet_state(self, key, default=None):
        """Get a pet state value by key."""
        row = self._conn.execute(
            "SELECT value FROM pet_state WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return row["value"]
        return default

    def set_pet_state(self, key, value):
        """Set a pet state value (upsert)."""
        self._conn.execute(
            "INSERT INTO pet_state (key, value, updated_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=datetime('now')",
            (key, str(value)),
        )
        self._conn.commit()

    def get_all_pet_state(self):
        """Get all pet state key-value pairs."""
        rows = self._conn.execute(
            "SELECT key, value, updated_at FROM pet_state"
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # --- Pet Interactions ---

    def insert_pet_interaction(self, prompt, response="", sentiment_score=0.0,
                               inference_time_ms=0, tokens_generated=0,
                               stage=0, mood="neutral", session_id=None):
        """Log a pet chat interaction."""
        cur = self._conn.execute(
            "INSERT INTO pet_interactions (prompt, response, sentiment_score, "
            "inference_time_ms, tokens_generated, stage, mood, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (prompt, response, sentiment_score, inference_time_ms,
             tokens_generated, stage, mood, session_id),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_pet_interaction(self, interaction_id, response,
                               inference_time_ms=0, tokens_generated=0):
        """Update a pet interaction with the model response."""
        self._conn.execute(
            "UPDATE pet_interactions SET response=?, inference_time_ms=?, "
            "tokens_generated=? WHERE id=?",
            (response, inference_time_ms, tokens_generated, interaction_id),
        )
        self._conn.commit()

    def get_recent_pet_interactions(self, limit=20):
        """Get recent pet interactions for mood calculation and history."""
        rows = self._conn.execute(
            "SELECT * FROM pet_interactions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pet_mood_score(self, window=20):
        """Calculate rolling mood score from recent sentiment scores.
        Returns float in range [-1.0, 1.0] where positive = happy."""
        rows = self._conn.execute(
            "SELECT sentiment_score FROM pet_interactions "
            "ORDER BY created_at DESC LIMIT ?",
            (window,),
        ).fetchall()
        if not rows:
            return 0.0
        scores = [r["sentiment_score"] for r in rows]
        return sum(scores) / len(scores)

    def get_pet_interaction_count(self):
        """Get total number of pet interactions (drives evolution stage)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM pet_interactions"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_pet_stats(self):
        """Composite pet statistics."""
        count = self.get_pet_interaction_count()
        mood_score = self.get_pet_mood_score()
        state = self.get_all_pet_state()
        recent = self.get_recent_pet_interactions(5)
        return {
            "total_interactions": count,
            "mood_score": round(mood_score, 3),
            "state": state,
            "recent_interactions": recent,
        }

    # --- Pet Analytics ---

    def get_pet_analytics(self, days=7):
        """Get pet analytics: mood trends, interaction frequency, stage progress."""
        # Daily mood averages
        mood_trend = self._conn.execute(
            "SELECT date(created_at) AS day, "
            "ROUND(AVG(sentiment_score), 3) AS avg_sentiment, "
            "COUNT(*) AS interactions, "
            "MIN(mood) AS mood_sample "
            "FROM pet_interactions "
            "WHERE created_at >= datetime('now', ?)"
            "GROUP BY date(created_at) ORDER BY day",
            ("-{} days".format(days),),
        ).fetchall()

        # Mood distribution
        mood_dist = self._conn.execute(
            "SELECT mood, COUNT(*) AS count "
            "FROM pet_interactions "
            "WHERE created_at >= datetime('now', ?) "
            "GROUP BY mood ORDER BY count DESC",
            ("-{} days".format(days),),
        ).fetchall()

        # Average inference time
        perf = self._conn.execute(
            "SELECT ROUND(AVG(inference_time_ms)) AS avg_ms, "
            "ROUND(AVG(tokens_generated), 1) AS avg_tokens, "
            "MAX(inference_time_ms) AS max_ms "
            "FROM pet_interactions "
            "WHERE inference_time_ms > 0 AND created_at >= datetime('now', ?)",
            ("-{} days".format(days),),
        ).fetchone()

        # Stage progression (first interaction at each stage)
        stages = self._conn.execute(
            "SELECT stage, MIN(created_at) AS reached_at, COUNT(*) AS interactions "
            "FROM pet_interactions GROUP BY stage ORDER BY stage"
        ).fetchall()

        # Total stats
        total = self.get_pet_interaction_count()
        current_mood = self.get_pet_mood_score()

        return {
            "period_days": days,
            "total_interactions": total,
            "current_mood_score": round(current_mood, 3),
            "daily_trend": [dict(r) for r in mood_trend],
            "mood_distribution": [dict(r) for r in mood_dist],
            "performance": dict(perf) if perf else {},
            "stage_progression": [dict(r) for r in stages],
        }

    # --- Pet Vitals (Tamagotchi System) ---

    def log_pet_vitals(self, hunger, cleanliness, energy, happiness,
                       intelligence, is_coma=False):
        """Snapshot current vitals to the log table for historical graphing."""
        self._conn.execute(
            "INSERT INTO pet_vitals_log (hunger, cleanliness, energy, "
            "happiness, intelligence, is_coma) VALUES (?, ?, ?, ?, ?, ?)",
            (hunger, cleanliness, energy, happiness, intelligence,
             1 if is_coma else 0),
        )
        self._conn.commit()

    def get_pet_vitals_history(self, hours=24, max_points=500):
        """Get vitals snapshots for charting.

        For short ranges (<= 48h), returns all data points.
        For longer ranges, downsamples by taking every Nth row to stay
        under max_points, keeping the chart responsive.
        """
        # Count total rows in range first
        count = self._conn.execute(
            "SELECT COUNT(*) FROM pet_vitals_log "
            "WHERE created_at >= datetime('now', ?)",
            ("-{} hours".format(hours),),
        ).fetchone()[0]

        if count <= max_points:
            # Return all rows
            rows = self._conn.execute(
                "SELECT * FROM pet_vitals_log "
                "WHERE created_at >= datetime('now', ?) "
                "ORDER BY created_at ASC",
                ("-{} hours".format(hours),),
            ).fetchall()
        else:
            # Downsample: use modulo on rowid to evenly space points
            step = max(1, count // max_points)
            rows = self._conn.execute(
                "SELECT * FROM ("
                "  SELECT *, ROW_NUMBER() OVER (ORDER BY created_at ASC) AS rn "
                "  FROM pet_vitals_log "
                "  WHERE created_at >= datetime('now', ?)"
                ") WHERE rn % ? = 1 OR rn = ("
                "  SELECT COUNT(*) FROM pet_vitals_log "
                "  WHERE created_at >= datetime('now', ?)"
                ")",
                ("-{} hours".format(hours), step, "-{} hours".format(hours)),
            ).fetchall()

        return [dict(r) for r in rows]

    # --- Pet Coma Log ---

    def log_pet_coma(self, trigger_vitals):
        """Log coma entry. Returns the coma log id."""
        trigger_json = json.dumps(trigger_vitals) if trigger_vitals else ""
        cur = self._conn.execute(
            "INSERT INTO pet_coma_log (trigger_vitals) VALUES (?)",
            (trigger_json,),
        )
        self._conn.commit()
        return cur.lastrowid

    def end_pet_coma(self, coma_id, revival_method="manual"):
        """Log coma exit with duration."""
        self._conn.execute(
            "UPDATE pet_coma_log SET exited_at=datetime('now'), "
            "revival_method=?, "
            "duration_hours=ROUND((julianday(datetime('now')) - "
            "julianday(entered_at)) * 24, 2) "
            "WHERE id=?",
            (revival_method, coma_id),
        )
        self._conn.commit()

    def get_pet_coma_history(self, limit=10):
        """Get past coma events."""
        rows = self._conn.execute(
            "SELECT * FROM pet_coma_log ORDER BY entered_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pet_coma_count(self):
        """Total number of coma events."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM pet_coma_log"
        ).fetchone()
        return row["cnt"] if row else 0

    # --- Pet Training History ---

    def log_pet_training(self, lora_version, final_loss=None,
                         perplexity_base=None, perplexity_finetuned=None,
                         intelligence_score=None, training_time_s=None,
                         dataset_size=None):
        """Log a training deployment event."""
        cur = self._conn.execute(
            "INSERT INTO pet_training_history (lora_version, final_loss, "
            "perplexity_base, perplexity_finetuned, intelligence_score, "
            "training_time_s, dataset_size) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (lora_version, final_loss, perplexity_base, perplexity_finetuned,
             intelligence_score, training_time_s, dataset_size),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_pet_training_history(self, limit=10):
        """Get LoRA deployment + intelligence history."""
        rows = self._conn.execute(
            "SELECT * FROM pet_training_history "
            "ORDER BY deployed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Pet Bad Interactions (for cleaning) ---

    def get_worst_pet_interactions(self, limit=10):
        """Get interactions with worst sentiment for cleanup review."""
        rows = self._conn.execute(
            "SELECT id, prompt, response, sentiment_score, mood, created_at "
            "FROM pet_interactions "
            "WHERE sentiment_score < -0.3 "
            "ORDER BY sentiment_score ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_interactions_discarded(self, interaction_ids):
        """Mark bad interactions as discarded (set mood to 'discarded').
        Returns number of rows updated."""
        if not interaction_ids:
            return 0
        placeholders = ",".join("?" * len(interaction_ids))
        cur = self._conn.execute(
            "UPDATE pet_interactions SET mood='discarded' WHERE id IN ({})".format(
                placeholders
            ),
            interaction_ids,
        )
        self._conn.commit()
        return cur.rowcount

    # --- Heartbeat Log ---

    def insert_heartbeat(self, prompt_type, prompt, response="",
                         sentiment_score=0.0, inference_time_ms=0,
                         tokens_generated=0, battery_pct=-1, is_charging=False,
                         hunger=0.0, cleanliness=0.0, energy=0.0, happiness=0.0,
                         shell_commands="", shell_results=""):
        """Log a heartbeat reflection."""
        cur = self._conn.execute(
            "INSERT INTO heartbeat_log (prompt_type, prompt, response, "
            "sentiment_score, inference_time_ms, tokens_generated, "
            "battery_pct, is_charging, hunger, cleanliness, energy, "
            "happiness, shell_commands, shell_results) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (prompt_type, prompt, response, sentiment_score,
             inference_time_ms, tokens_generated, battery_pct,
             1 if is_charging else 0, hunger, cleanliness, energy,
             happiness, shell_commands, shell_results),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_recent_heartbeats(self, limit=20):
        """Get recent heartbeat reflections."""
        rows = self._conn.execute(
            "SELECT * FROM heartbeat_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_heartbeat_count_since(self, since_iso):
        """Count heartbeats since a given ISO timestamp."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM heartbeat_log WHERE created_at >= ?",
            (since_iso,),
        ).fetchone()
        return row[0] if row else 0

    def get_interactions_since(self, since_iso):
        """Count pet interactions since a given ISO timestamp."""
        # Normalize: created_at uses "YYYY-MM-DD HH:MM:SS" (space separator,
        # no timezone) but since_iso may be full ISO with "T" and "+00:00".
        # SQLite string comparison fails when formats differ (space < 'T').
        normalized = since_iso.replace("T", " ")
        # Strip timezone offset if present (e.g. "+00:00", "Z")
        if "+" in normalized and normalized.index("+") > 10:
            normalized = normalized[:normalized.index("+")]
        normalized = normalized.rstrip("Z")
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pet_interactions WHERE created_at >= ?",
            (normalized,),
        ).fetchone()
        return row[0] if row else 0
