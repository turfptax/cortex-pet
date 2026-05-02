"""Cortex pet plugin.

Owns the pet runtime (PetEngine + Heartbeat) and exposes the full pet
HTTP surface under /plugins/pet/* via the plugin route framework
(slice 2c2c1). cortex_protocol.py no longer carries any pet/heartbeat
CMD handlers.

on_load():
  1. Migrate existing pet rows from cortex.db into pet.db (one-time,
     gated by a flag in pet_state — safe to re-run).
  2. Construct PetEngine against pet.db.
  3. Construct Heartbeat and START it.

Slice 2c2d will drop the dead pet tables from cortex_db.py + cortex.db.
"""

from datetime import datetime, timezone, timedelta
import logging

from plugin_api import Plugin, Route
from pet import PetEngine
from heartbeat import Heartbeat
from migrate import migrate_if_needed
from pet_db import PetDB
from pet_config import DREAM_MIN_INTERACTIONS, DREAM_COOLDOWN_HOURS


log = logging.getLogger("plugin.pet")


def _strip_meta(payload):
    """Drop framework metadata keys (those starting with __) from payload."""
    return {k: v for k, v in payload.items() if not k.startswith("__")}


def _as_int(payload, key, default, max_value=None):
    """Coerce payload[key] to int with default + optional clamp."""
    val = payload.get(key, default)
    try:
        n = int(val)
    except (TypeError, ValueError):
        n = default
    if max_value is not None:
        n = min(n, max_value)
    return n


def _as_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)


class PetPlugin(Plugin):
    """Pet plugin owning PetEngine + Heartbeat against pet.db."""

    def __init__(self, api):
        super().__init__(api)
        self.pet_engine = None
        self.heartbeat = None

    # ── HTTP routes ─────────────────────────────────────────────────

    def http_routes(self):
        """All 23 pet routes mounted under /plugins/pet/*.

        Migrated from cortex_protocol.py's pet/heartbeat CMD handlers in
        slice 2c2c1. Handler signature: (payload: dict) -> dict.

        See cortex-core/notes/ (or git log for slice 2c2c) for the route
        table. force_train was dropped because the Hub bypasses it.
        """
        return [
            # ── Pet / LLM ──────────────────────────────────────────
            Route("POST", "/chat",                 self._http_chat),
            Route("GET",  "/responses",            self._http_responses),
            Route("GET",  "/status",               self._http_status),
            Route("GET",  "/mood",                 self._http_mood),
            Route("GET",  "/history",              self._http_history),
            Route("GET",  "/analytics",            self._http_analytics),
            # ── Vitals + actions ───────────────────────────────────
            Route("GET",  "/vitals",               self._http_vitals),
            Route("POST", "/feed",                 self._http_feed),
            Route("POST", "/clean",                self._http_clean),
            Route("POST", "/rest",                 self._http_rest),
            Route("GET",  "/intelligence",         self._http_intelligence_get),
            Route("POST", "/intelligence",         self._http_intelligence_post),
            Route("GET",  "/vitals/history",       self._http_vitals_history),
            Route("GET",  "/coma/status",          self._http_coma_status),
            Route("GET",  "/coma/history",         self._http_coma_history),
            Route("GET",  "/training-history",     self._http_training_history),
            # ── Heartbeat / Autonomous life ────────────────────────
            Route("GET",  "/heartbeat",            self._http_heartbeat_status),
            Route("POST", "/heartbeat",            self._http_heartbeat_config),
            Route("GET",  "/heartbeat/log",        self._http_heartbeat_log),
            Route("POST", "/sleep",                self._http_sleep),
            Route("POST", "/wake",                 self._http_wake),
            Route("POST", "/tuck-in",              self._http_tuck_in),
            Route("POST", "/dream-complete",       self._http_dream_complete),
        ]

    # ── Pet / LLM handlers ──────────────────────────────────────────

    def _http_chat(self, payload):
        """POST /plugins/pet/chat — send a prompt for inference (async).

        Body: {"prompt": "..."} -> {"ok": true, "interaction_id": N}
        Caller polls /plugins/pet/responses for the completed response.
        """
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        prompt = payload.get("prompt", "")
        if not prompt:
            return {"ok": False, "error": "missing 'prompt' field"}
        # session_id used to come from CortexProtocol's _active_session_id;
        # plugin doesn't track sessions yet (slice 2c3 may add via core_memory).
        interaction_id = self.pet_engine.ask(prompt, session_id=None)
        if interaction_id is None:
            return {"ok": False, "error": "pet disabled or queue full"}
        return {"ok": True, "interaction_id": interaction_id}

    def _http_responses(self, payload):
        """GET /plugins/pet/responses?since_id=N — drain queue + recent."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        responses = self.pet_engine.get_all_responses()
        if not responses:
            since_id = payload.get("since_id")
            if since_id is not None:
                try:
                    since_id = int(since_id)
                except (TypeError, ValueError):
                    since_id = None
            responses = self.pet_engine.get_recent_responses(since_id=since_id)
        return {"ok": True, "responses": responses}

    def _http_status(self, payload):
        """GET /plugins/pet/status — plugin liveness + pet stats snapshot."""
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

    def _http_mood(self, payload):
        """GET /plugins/pet/mood — detailed mood info."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        return {
            "ok": True,
            "mood": self.pet_engine.mood,
            "mood_score": round(self.pet_engine.mood_score, 3),
            "stage": self.pet_engine.stage,
            "stage_name": self.pet_engine.stage_name,
            "interaction_count": self.pet_engine.interaction_count,
        }

    def _http_history(self, payload):
        """GET /plugins/pet/history?limit=10 — recent interactions."""
        limit = _as_int(payload, "limit", 10, max_value=50)
        return {
            "ok": True,
            "interactions": self.api.db.get_recent_pet_interactions(limit),
        }

    def _http_analytics(self, payload):
        """GET /plugins/pet/analytics?days=7 — mood/perf/stage trends."""
        days = _as_int(payload, "days", 7, max_value=365)
        return {"ok": True, **self.api.db.get_pet_analytics(days)}

    # ── Vitals / care actions ──────────────────────────────────────

    def _http_vitals(self, payload):
        """GET /plugins/pet/vitals — current vitals + bloom number."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        vitals = self.pet_engine.get_vitals()
        vitals["bloom"] = self.pet_engine.bloom
        return {"ok": True, **vitals}

    def _http_feed(self, payload):
        """POST /plugins/pet/feed — body {type?: chat_snack|data_meal|training_feast}."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        feed_type = payload.get("type", "chat_snack")
        new_hunger = self.pet_engine.feed(feed_type)
        return {"ok": True, "hunger": round(new_hunger, 3), "type": feed_type}

    def _http_clean(self, payload):
        """POST /plugins/pet/clean — body {discard_ids?: [int]}."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        discard_ids = payload.get("discard_ids", [])
        new_cleanliness, discarded = self.pet_engine.clean(discard_ids)
        return {
            "ok": True,
            "cleanliness": round(new_cleanliness, 3),
            "discarded": discarded,
        }

    def _http_rest(self, payload):
        """POST /plugins/pet/rest — instant +10% energy."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        new_energy = self.pet_engine.rest()
        return {"ok": True, "energy": round(new_energy, 3)}

    def _http_intelligence_get(self, payload):
        """GET /plugins/pet/intelligence — IQ score breakdown."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        return {"ok": True, **self.pet_engine.get_intelligence()}

    def _http_intelligence_post(self, payload):
        """POST /plugins/pet/intelligence — push training metrics, recompute IQ."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        self.pet_engine.on_training_complete(_strip_meta(payload))
        return {"ok": True, "intelligence": round(self.pet_engine.intelligence, 1)}

    def _http_vitals_history(self, payload):
        """GET /plugins/pet/vitals/history?hours=24 — historical vitals."""
        hours = _as_int(payload, "hours", 24, max_value=8760)
        return {
            "ok": True,
            "history": self.api.db.get_pet_vitals_history(hours),
        }

    def _http_coma_status(self, payload):
        """GET /plugins/pet/coma/status — coma status + revival progress."""
        if self.pet_engine is None:
            return {"ok": False, "error": "pet engine not initialized"}
        return {"ok": True, **self.pet_engine.get_coma_status()}

    def _http_coma_history(self, payload):
        """GET /plugins/pet/coma/history — past coma events."""
        return {"ok": True, "history": self.api.db.get_pet_coma_history()}

    def _http_training_history(self, payload):
        """GET /plugins/pet/training-history — LoRA deployment history."""
        return {"ok": True, "history": self.api.db.get_pet_training_history()}

    # ── Heartbeat / autonomous life ────────────────────────────────

    def _http_heartbeat_status(self, payload):
        """GET /plugins/pet/heartbeat — heartbeat system status."""
        if self.heartbeat is None:
            return {"ok": False, "enabled": False}
        return {"ok": True, **self.heartbeat.get_heartbeat_stats()}

    def _http_heartbeat_log(self, payload):
        """GET /plugins/pet/heartbeat/log?limit=20 — recent reflections."""
        limit = _as_int(payload, "limit", 20, max_value=100)
        return {"ok": True, "entries": self.api.db.get_recent_heartbeats(limit)}

    def _http_heartbeat_config(self, payload):
        """POST /plugins/pet/heartbeat — body {interval_s?, enabled?}."""
        updates = []
        if "interval_s" in payload:
            interval = max(300, min(7200, _as_int(payload, "interval_s", 1800)))
            self.api.db.set_pet_state("heartbeat_interval_s", str(interval))
            updates.append("interval={}".format(interval))
        if "enabled" in payload:
            enabled = _as_bool(payload["enabled"])
            self.api.db.set_pet_state("heartbeat_enabled",
                                       "1" if enabled else "0")
            updates.append("enabled={}".format(enabled))
        return {"ok": True, "updates": updates or ["no changes"]}

    def _http_sleep(self, payload):
        """POST /plugins/pet/sleep — body {reason?: 'user'}."""
        if self.heartbeat is None:
            return {"ok": False, "error": "heartbeat not available"}
        reason = payload.get("reason", "user")
        self.heartbeat.enter_sleep(reason)
        return {"ok": True, "reason": reason}

    def _http_wake(self, payload):
        """POST /plugins/pet/wake — wake from sleep."""
        if self.heartbeat is None:
            return {"ok": False, "error": "heartbeat not available"}
        self.heartbeat.wake_up()
        return {"ok": True}

    def _http_tuck_in(self, payload):
        """POST /plugins/pet/tuck-in — sleep + dream readiness probe.

        Also registers the requesting client's IP as the Hub's IP so the
        Pi knows where to call back for dream training. (Replaces the
        per-request tracking that lived in http_server.py before 2c2c1.)
        """
        if self.heartbeat is None:
            return {"ok": False, "error": "heartbeat not available"}

        # Pet-side responsibility: capture Hub's IP for dream callbacks.
        client_ip = payload.get("__client_ip__")
        if client_ip and client_ip not in ("127.0.0.1", "::1"):
            try:
                self.heartbeat.register_hub_ip(client_ip)
            except Exception:
                pass

        # Put pet to sleep
        self.heartbeat.enter_sleep("tuck_in")

        hub_ip = self.heartbeat._known_hub_ip
        hub_available = False
        if hub_ip:
            hub_available = self.heartbeat._check_hub(hub_ip)

        # Dream readiness check
        last_dream = self.heartbeat._last_dream_time
        last_dream_iso = (last_dream.isoformat()
                          if last_dream else "2000-01-01T00:00:00")
        new_interactions = self.api.db.get_interactions_since(last_dream_iso)

        cooldown_ok = True
        if last_dream:
            elapsed = datetime.now(timezone.utc) - last_dream
            cooldown_ok = elapsed >= timedelta(hours=DREAM_COOLDOWN_HOURS)

        return {
            "ok": True,
            "sleeping": True,
            "hub_ip": hub_ip,
            "hub_available": hub_available,
            "new_interactions": new_interactions,
            "min_interactions": DREAM_MIN_INTERACTIONS,
            "interactions_ready": new_interactions >= DREAM_MIN_INTERACTIONS,
            "cooldown_ok": cooldown_ok,
            "dream_ready": (hub_available
                            and new_interactions >= DREAM_MIN_INTERACTIONS
                            and cooldown_ok),
        }

    def _http_dream_complete(self, payload):
        """POST /plugins/pet/dream-complete — Hub reports training done."""
        if self.heartbeat is None:
            return {"ok": False, "error": "heartbeat not available"}
        self.heartbeat.on_dream_complete(training_metrics=_strip_meta(payload))
        return {"ok": True}

    # ── Lifecycle ───────────────────────────────────────────────────

    def on_load(self) -> None:
        pet_db_path = self.api.plugin_data / "pet.db"

        # ── Step 1: open PetDB first so pet tables exist ──────────────
        # cortex_db.py (post-2c2d) no longer carries pet schema, so the
        # generic CortexDB the runtime opened against pet.db has no pet
        # tables. PetDB extends CortexDB and adds them. Must happen
        # BEFORE migrate_if_needed which reads/writes pet_state.
        if self.api.db is not None:
            try:
                self.api.db.close()
            except Exception:
                pass
        self.api.db = PetDB(str(pet_db_path))
        self.api.log.info("pet.db opened via PetDB (pet schema + helpers)")

        # ── Step 2: one-time migration from cortex.db ───────────────
        try:
            cortex_db_path = self.api.core_db_path
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

        # ── Step 3: PetEngine against pet.db (now holds the history) ──
        self.pet_engine = PetEngine(self.api.db, battery=self.api.battery)
        self.api.log.info("pet engine ready")

        # ── Step 4: Heartbeat owns the autonomous loop ──────────────
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
