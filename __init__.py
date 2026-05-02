"""Cortex pet plugin — slice 2b2.

The pet plugin owns the runtime now. main.py no longer creates its own
PetEngine or Heartbeat; instead it pulls the references off the loaded
PetPlugin and passes them into CortexProtocol and StateContext.

on_load():
  1. Migrate existing pet rows from cortex.db into pet.db (one-time,
     gated by a flag in pet_state — safe to re-run).
  2. Construct PetEngine against pet.db, which now holds the migrated
     history (so vitals load with the real feeds=82 / cleans=360 etc.).
  3. Construct Heartbeat and START it. This is now the only autonomous
     reflection loop on the device.

Slice 2c removes the dead pet tables from cortex.db (and their CREATE
statements + the ~85 PET_*/HEARTBEAT_* config keys that pet.py still
imports from cortex's config).
"""

import logging

from plugin_api import Plugin, Route
from pet import PetEngine
from heartbeat import Heartbeat
from migrate import migrate_if_needed


log = logging.getLogger("plugin.pet")


class PetPlugin(Plugin):
    """Pet plugin owning PetEngine + Heartbeat against pet.db."""

    def __init__(self, api):
        super().__init__(api)
        self.pet_engine = None
        self.heartbeat = None

    def http_routes(self):
        """Routes mounted by the core HTTP server.

        Slice 2c2b — only a status stub for now to verify the route
        mounting framework works. Slice 2c2c migrates the full surface
        (feed, clean, chat, vitals, etc.) over from cortex_protocol.py's
        24 pet/heartbeat CMD handlers.

        All routes mount under /plugins/pet/<path> with handler signature:
            (payload: dict) -> dict
        """
        return [
            Route("GET", "/status", self._http_status),
        ]

    def _http_status(self, payload):
        """GET /plugins/pet/status — basic plugin liveness + stats snapshot."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        try:
            stats = self.pet_engine.get_stats()
        except Exception as e:
            return {"ok": False, "error": "get_stats failed: {}".format(e)}
        return {
            "ok": True,
            "plugin": "pet",
            "version": "0.1.0",
            "engine_loaded": True,
            "heartbeat_running": (
                self.heartbeat is not None
                and getattr(self.heartbeat, "_running", False)
            ),
            "stats": stats,
        }

    def on_load(self) -> None:
        # ── Step 1: one-time migration from cortex.db ───────────────
        try:
            cortex_db_path = self.api.core_db_path
            pet_db_path = self.api.plugin_data / "pet.db"
            counts = migrate_if_needed(cortex_db_path, pet_db_path)
            if counts:
                total = sum(counts.values())
                self.api.log.info(
                    "pet.db migration ok: %d rows across %d tables",
                    total, len(counts),
                )
        except Exception as e:
            # Don't take down core if migration hits a snag — let
            # PetEngine come up against whatever's in pet.db today.
            self.api.log.exception(
                "migration failed: %s — continuing with current pet.db state", e
            )

        # ── Step 2: PetEngine against pet.db (now holds the history) ──
        self.pet_engine = PetEngine(self.api.db, battery=self.api.battery)
        self.api.log.info("pet engine ready")

        # ── Step 3: Heartbeat owns the autonomous loop ──────────────
        self.heartbeat = Heartbeat(
            self.api.db, self.pet_engine, battery=self.api.battery
        )
        self.heartbeat.start()
        self.api.log.info("heartbeat started (plugin owns the autonomous loop now)")

    def on_unload(self) -> None:
        if self.heartbeat is not None and hasattr(self.heartbeat, "stop"):
            try:
                self.heartbeat.stop()
            except Exception:
                pass
        if self.pet_engine is not None and hasattr(self.pet_engine, "cleanup"):
            try:
                self.pet_engine.cleanup()
            except Exception:
                pass
        self.api.log.info("plugin pet unloaded")


def register(api):
    """Entry point invoked by plugins_runtime._load_plugin()."""
    return PetPlugin(api)
