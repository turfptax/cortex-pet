"""One-time migration of pet rows from cortex.db into the plugin's pet.db.

Runs on plugin on_load(). Idempotent — checks a flag in pet_state to skip
on subsequent boots. Pet rows in cortex.db are left intact (slice 2c
removes them); the only writes go to pet.db.

This file becomes dead code after slice 2c removes the pet tables from
cortex.db. Slice 2c should delete it.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("plugin.pet.migrate")

PET_TABLES = [
    "pet_state",
    "pet_interactions",
    "pet_vitals_log",
    "pet_coma_log",
    "pet_training_history",
    "heartbeat_log",
]

MIGRATION_FLAG_KEY = "migrated_from_cortex_db"


def migrate_if_needed(cortex_db_path, pet_db_path) -> dict:
    """Copy pet table rows from cortex.db into pet.db. Idempotent.

    Args:
        cortex_db_path: filesystem path to cortex.db (opened read-only)
        pet_db_path: filesystem path to pet.db (modified in place)

    Returns:
        dict mapping table name -> rows copied. Empty dict if migration
        was skipped (flag already set, or cortex.db not found).
    """
    cortex_path = str(Path(cortex_db_path).resolve())
    pet_path = str(Path(pet_db_path).resolve())

    if not Path(cortex_path).is_file():
        log.warning("cortex.db not found at %s; skipping migration", cortex_path)
        return {}

    dst = sqlite3.connect(pet_path)
    try:
        # ── Idempotency check ──────────────────────────────────────────
        cursor = dst.execute(
            "SELECT value FROM pet_state WHERE key = ?",
            (MIGRATION_FLAG_KEY,),
        )
        row = cursor.fetchone()
        if row:
            log.info("pet.db migration already done at %s; skipping", row[0])
            return {}

        log.info("migration starting: %s -> %s", cortex_path, pet_path)
        t0 = time.monotonic()

        # ── Open source read-only ──────────────────────────────────────
        src = sqlite3.connect(f"file:{cortex_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row

        try:
            counts = {}
            for table in PET_TABLES:
                # Wipe pet.db's rows in this table so re-runs converge to
                # an identical state. Slice 2b1's parallel-running plugin
                # PetEngine left a few decay rows in pet.db that we want
                # to discard in favour of cortex.db's authoritative copy.
                dst.execute(f"DELETE FROM {table}")

                rows = src.execute(f"SELECT * FROM {table}").fetchall()
                if not rows:
                    counts[table] = 0
                    log.info("  %-22s     0 rows", table)
                    continue

                cols = list(rows[0].keys())
                placeholders = ", ".join(["?"] * len(cols))
                col_names = ", ".join(cols)
                sql = (
                    f"INSERT INTO {table} ({col_names}) "
                    f"VALUES ({placeholders})"
                )
                dst.executemany(sql, [tuple(r) for r in rows])
                counts[table] = len(rows)
                log.info("  %-22s %5d rows copied", table, len(rows))
        finally:
            src.close()

        # ── Set the migration flag ─────────────────────────────────────
        # We just DELETE'd all of pet_state so any prior flag is gone;
        # use INSERT OR REPLACE to be defensive against a partial re-run.
        timestamp = datetime.now(timezone.utc).isoformat()
        dst.execute(
            "INSERT OR REPLACE INTO pet_state (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (MIGRATION_FLAG_KEY, timestamp, timestamp),
        )
        dst.commit()

        elapsed_ms = (time.monotonic() - t0) * 1000
        total = sum(counts.values())
        log.info(
            "migration complete: %d rows across %d tables in %.0f ms",
            total, len(counts), elapsed_ms,
        )
        return counts
    finally:
        dst.close()
