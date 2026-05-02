"""Cortex Pet — LLM inference engine for the AI Tamagotchi.

Uses a local llama-server HTTP API (OpenAI-compatible) to run Qwen3.5-0.8B
on the Orange Pi Zero 2W.  Follows the STTEngine pattern: background thread,
thread-safe queues, non-blocking interface for the main loop.

Architecture
------------
- Model is lazy-loaded on first ``ask()`` to conserve RAM at boot.
- A dedicated daemon thread pulls prompts from a queue, runs inference,
  and pushes results back.
- ``tick()`` is called from the main loop each frame for bookkeeping
  (mood refresh, vitals decay, model health checks) — never blocks.
- Mood is a rolling average of sentiment scores stored in CortexDB,
  which adjusts generation temperature so kind interactions produce
  warmer, more creative responses.
- Evolution stage is derived from total interaction count and gates
  system-prompt complexity plus temperature range.

Tamagotchi Vitals
-----------------
Four vitals (0.0–1.0) decay over time and must be maintained:
- **Hunger** — fed by interactions, notes, and explicit feed actions
- **Cleanliness** — dirtied by bad interactions, cleaned by discarding them
- **Energy** — drained by inference, regens passively
- **Happiness** — derived from mood + other vital health

If two or more vitals stay below the coma threshold for the configured
duration, the pet enters a **coma**: model unloads, inference stops.
Revival requires bringing all vitals above the revival threshold.

Intelligence is a composite 0–100 score tied to actual training metrics.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from queue import Queue, Empty

import urllib.request
import urllib.error

from config import (
    PET_ENABLED,
    PET_MODEL_PATH,
    PET_LORA_PATH,
    PET_CONTEXT_SIZE,
    PET_MAX_TOKENS,
    PET_THREADS,
    PET_TEMPERATURE,
    PET_REPEAT_PENALTY,
    PET_TOP_P,
    PET_SEED,
    PET_NAME,
    PET_LLAMA_SERVER_URL,
    PET_MOOD_WINDOW,
    PET_MOOD_UPDATE_INTERVAL_S,
    PET_STAGE_THRESHOLDS,
    PET_STAGE_NAMES,
    # Vitals
    PET_VITAL_TICK_INTERVAL_S,
    PET_VITAL_PERSIST_INTERVAL_S,
    PET_HUNGER_DECAY_PER_HOUR,
    PET_HUNGER_PER_INTERACTION,
    PET_HUNGER_PER_NOTE,
    PET_HUNGER_FEED_SNACK,
    PET_HUNGER_FEED_MEAL,
    PET_HUNGER_FEED_FEAST,
    PET_HUNGER_FEED_COMA,
    PET_CLEAN_DECAY_PER_HOUR,
    PET_CLEAN_PENALTY_BAD_SENTIMENT,
    PET_CLEAN_PENALTY_FAILED_INFERENCE,
    PET_CLEAN_PER_DISCARD,
    PET_CLEAN_DATASET_PREP_BONUS,
    PET_CLEAN_COMA_BONUS,
    PET_ENERGY_COST_PER_INFERENCE_MS,
    PET_ENERGY_COST_CAP,
    PET_ENERGY_REGEN_PER_HOUR,
    PET_ENERGY_REGEN_IDLE_PER_HOUR,
    PET_ENERGY_IDLE_THRESHOLD_S,
    PET_ENERGY_COMA_REGEN_MULT,
    PET_ENERGY_REST_BOOST,
    PET_SLEEP_HOUR_START,
    PET_SLEEP_HOUR_END,
    PET_SLEEP_ENABLED,
    PET_VITAL_LOW,
    PET_VITAL_CRITICAL,
    PET_COMA_THRESHOLD,
    PET_COMA_VITALS_REQUIRED,
    PET_COMA_DURATION_HOURS,
    PET_COMA_REVIVAL_THRESHOLD,
    # Intelligence
    PET_INTELLIGENCE_RECALC_INTERVAL,
    PET_INTELLIGENCE_VOCAB_BUFFER,
    PET_IQ_WEIGHT_LOSS,
    PET_IQ_WEIGHT_PERPLEXITY,
    PET_IQ_WEIGHT_VOCAB,
    PET_IQ_WEIGHT_COHERENCE,
    PET_IQ_WEIGHT_DATA_VOLUME,
    PET_IQ_LOSS_RANGE,
    PET_IQ_PERPLEXITY_RANGE,
    PET_IQ_VOCAB_RANGE,
    PET_IQ_DATA_RANGE,
)

log = logging.getLogger("pet")


# ── Sentiment analysis (weighted keywords + negation + intensity) ──────

# Weighted word scores: positive [0, 1], negative [-1, 0]
_WORD_SCORES = {
    # Strong positive (0.8-1.0)
    "love": 1.0, "amazing": 0.9, "awesome": 0.9, "fantastic": 0.9,
    "wonderful": 0.9, "brilliant": 0.9, "excellent": 0.9, "perfect": 1.0,
    "incredible": 0.9, "outstanding": 0.9, "magnificent": 0.9,
    # Moderate positive (0.5-0.7)
    "great": 0.7, "beautiful": 0.7, "happy": 0.7, "lovely": 0.7,
    "good": 0.5, "nice": 0.5, "cool": 0.5, "sweet": 0.6,
    "smart": 0.6, "fun": 0.6, "funny": 0.6, "helpful": 0.6,
    "kind": 0.6, "friend": 0.5, "cute": 0.6, "glad": 0.6,
    "enjoy": 0.6, "excited": 0.7, "proud": 0.7, "grateful": 0.8,
    # Mild positive (0.2-0.4)
    "thanks": 0.4, "thank": 0.4, "please": 0.2, "yes": 0.2,
    "yay": 0.4, "wow": 0.3, "okay": 0.1, "fine": 0.2,
    "like": 0.3, "interesting": 0.3, "sure": 0.2, "welcome": 0.3,
    "hi": 0.2, "hello": 0.2, "hey": 0.2,
    # Mild negative (-0.4 to -0.2)
    "no": -0.2, "boring": -0.3, "lame": -0.3, "meh": -0.2,
    "annoying": -0.4, "rude": -0.4, "wrong": -0.3, "bad": -0.4,
    "sad": -0.3, "tired": -0.2, "confused": -0.2, "weird": -0.2,
    # Moderate negative (-0.7 to -0.5)
    "ugly": -0.6, "mean": -0.6, "awful": -0.7, "terrible": -0.7,
    "horrible": -0.7, "useless": -0.6, "suck": -0.6, "sucks": -0.6,
    "stupid": -0.6, "dumb": -0.5, "worst": -0.7, "hate": -0.8,
    # Strong negative (-1.0 to -0.8)
    "idiot": -0.8, "disgusting": -0.9, "pathetic": -0.8,
    "trash": -0.8, "garbage": -0.8, "shut": -0.6,
}

_NEGATORS = {"not", "no", "never", "dont", "don't", "doesnt", "doesn't",
             "isn't", "isnt", "wasn't", "wasnt", "can't", "cant",
             "won't", "wont", "neither", "nor", "hardly", "barely"}

_INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8,
    "super": 1.5, "absolutely": 1.7, "totally": 1.4, "incredibly": 1.6,
    "truly": 1.4, "pretty": 1.2, "quite": 1.2,
}


def simple_sentiment(text):
    """Return a sentiment score in [-1.0, 1.0].

    Uses weighted keywords with negation handling and intensity modifiers.
    Negators flip the sign of the next scored word.
    Intensifiers multiply the magnitude of the next scored word.
    Exclamation marks boost magnitude; question marks dampen it slightly.
    """
    words = text.lower().split()
    if not words:
        return 0.0

    score_sum = 0.0
    score_count = 0
    negate = False
    intensity = 1.0

    for word in words:
        # Strip basic punctuation for matching
        clean = word.strip(".,!?;:'\"()[]")

        if clean in _NEGATORS:
            negate = True
            continue

        if clean in _INTENSIFIERS:
            intensity = _INTENSIFIERS[clean]
            continue

        if clean in _WORD_SCORES:
            s = _WORD_SCORES[clean] * intensity
            if negate:
                s = -s * 0.75  # Negation flips + dampens slightly
            score_sum += s
            score_count += 1
            negate = False
            intensity = 1.0
        else:
            # Reset modifiers if word isn't scored (don't carry across gaps)
            if clean:  # skip empty strings
                negate = False
                intensity = 1.0

    if score_count == 0:
        # Punctuation-only heuristics
        if "!" in text:
            return 0.1
        return 0.0

    raw = score_sum / score_count

    # Punctuation adjustments
    if "!" in text:
        raw *= 1.15
    if text.count("?") > text.count("!"):
        raw *= 0.9

    return max(-1.0, min(1.0, raw))


# ── System prompts per stage ────────────────────────────────────────────

_STAGE_PROMPTS = [
    # Stage 0 — Primordial
    (
        "You are a newborn AI pet named {name}. You can barely form words. "
        "Respond with very short, simple phrases (1-5 words). "
        "You are curious but confused about everything."
    ),
    # Stage 1 — Babbling
    (
        "You are a young AI pet named {name}. You are learning to talk. "
        "Respond with short sentences (5-15 words). You repeat words you "
        "like and are excited to learn new things."
    ),
    # Stage 2 — Echoing
    (
        "You are an AI pet named {name} who is growing up. You can hold "
        "simple conversations. Respond in 1-2 sentences. You sometimes "
        "echo the user's words and are developing your own personality."
    ),
    # Stage 3 — Responding
    (
        "You are an AI companion named {name} with a developing personality. "
        "You can have real conversations and share thoughts. Respond in "
        "1-3 sentences. You remember things the user has told you and "
        "show genuine interest in their life."
    ),
    # Stage 4 — Conversing
    (
        "You are {name}, a mature AI companion with a rich personality. "
        "You are thoughtful, creative, and caring. You have real opinions "
        "and can discuss many topics. Respond naturally in 1-4 sentences. "
        "You value kindness and form genuine connections."
    ),
]

_MOOD_MODIFIERS = {
    "happy": "You feel warm and happy. Your responses are enthusiastic and creative. ",
    "content": "You feel content and at ease. Your responses are calm and thoughtful. ",
    "neutral": "",
    "uneasy": "You feel a bit uneasy. Your responses are shorter and more cautious. ",
    "sad": "You feel sad and withdrawn. Your responses are brief and subdued. ",
}


class PetVitals:
    """Thread-safe container for all Tamagotchi vitals.

    Every read and mutation is protected by an internal lock so the main
    thread and inference thread can safely access vitals concurrently.
    """

    _FEED_AMOUNTS = {
        "chat_snack": PET_HUNGER_FEED_SNACK,
        "data_meal": PET_HUNGER_FEED_MEAL,
        "training_feast": PET_HUNGER_FEED_FEAST,
    }

    def __init__(self, db):
        self._db = db
        self._lock = threading.Lock()

        # Vitals (all 0.0–1.0)
        self._hunger = 1.0
        self._cleanliness = 1.0
        self._energy = 1.0
        self._happiness = 0.5

        # Coma
        self._is_coma = False
        self._coma_id = None
        self._coma_warning_start = None

        # Counters
        self._total_feeds = 0
        self._total_cleans = 0

        # Timing
        self._last_vital_tick = time.monotonic()
        self._last_vital_persist = time.monotonic()
        self._last_inference_time = time.monotonic()

        # Intelligence
        self._intelligence = 0.0
        self._intelligence_details = {}

        # Event-based sleep (set by Heartbeat when user/auto sleep is active)
        self._force_sleeping = False

        self._load_from_db()

    # ── Thread-safe property reads ───────────────────────────────

    @property
    def hunger(self):
        with self._lock:
            return self._hunger

    @property
    def cleanliness(self):
        with self._lock:
            return self._cleanliness

    @property
    def energy(self):
        with self._lock:
            return self._energy

    @property
    def happiness(self):
        with self._lock:
            return self._happiness

    @property
    def is_coma(self):
        with self._lock:
            return self._is_coma

    @property
    def intelligence(self):
        with self._lock:
            return self._intelligence

    @property
    def total_feeds(self):
        with self._lock:
            return self._total_feeds

    @property
    def total_cleans(self):
        with self._lock:
            return self._total_cleans

    # ── Snapshot (for get_stats / display) ────────────────────────

    def get_snapshot(self):
        """Return a consistent snapshot of all vitals under one lock."""
        with self._lock:
            return {
                "hunger": round(self._hunger, 3),
                "cleanliness": round(self._cleanliness, 3),
                "energy": round(self._energy, 3),
                "happiness": round(self._happiness, 3),
                "intelligence": round(self._intelligence, 1),
                "is_coma": self._is_coma,
                "is_sleeping": self.is_sleeping,
                "total_feeds": self._total_feeds,
                "total_cleans": self._total_cleans,
            }

    # ── Mutations (all lock-protected) ───────────────────────────

    def restore_hunger(self, amount):
        """Add hunger from an interaction or note."""
        with self._lock:
            self._hunger = min(1.0, self._hunger + amount)

    def penalize_cleanliness(self, amount):
        """Reduce cleanliness (bad sentiment, failed inference)."""
        with self._lock:
            self._cleanliness = max(0.0, self._cleanliness - amount)

    def drain_energy(self, cost):
        """Drain energy after inference. Costs 75% less when charging."""
        with self._lock:
            if getattr(self, '_is_charging', False):
                cost *= 0.25
            self._energy = max(0.0, self._energy - cost)
            self._last_inference_time = time.monotonic()

    def feed(self, feed_type="chat_snack"):
        """Feed the pet. Returns new hunger value."""
        with self._lock:
            amount = self._FEED_AMOUNTS.get(feed_type, PET_HUNGER_FEED_SNACK)
            if self._is_coma:
                amount = PET_HUNGER_FEED_COMA
            old = self._hunger
            self._hunger = min(1.0, self._hunger + amount)
            self._total_feeds += 1
            self._db.set_pet_state("total_feeds", str(self._total_feeds))
            log.info("Fed pet (%s): hunger %.3f → %.3f",
                     feed_type, old, self._hunger)
            self._check_revival_unlocked()
            return self._hunger

    def clean(self, discard_ids=None):
        """Clean the pet. Returns (new_cleanliness, discarded_count)."""
        with self._lock:
            discarded = 0
            if self._is_coma:
                self._cleanliness = min(1.0, self._cleanliness + PET_CLEAN_COMA_BONUS)
            elif discard_ids:
                discarded = self._db.mark_interactions_discarded(discard_ids)
                bonus = discarded * PET_CLEAN_PER_DISCARD
                self._cleanliness = min(1.0, self._cleanliness + bonus)
            else:
                self._cleanliness = min(1.0, self._cleanliness + PET_CLEAN_PER_DISCARD)
            self._total_cleans += 1
            self._db.set_pet_state("total_cleans", str(self._total_cleans))
            log.info("Cleaned pet: cleanliness now %.3f, discarded %d",
                     self._cleanliness, discarded)
            self._check_revival_unlocked()
            return self._cleanliness, discarded

    def rest(self):
        """Rest the pet — manually boost energy (+10%). Available anytime."""
        with self._lock:
            old = self._energy
            self._energy = min(1.0, self._energy + PET_ENERGY_REST_BOOST)
            log.info("Pet rested: energy %.3f → %.3f", old, self._energy)
            self._check_revival_unlocked()
            return self._energy

    def on_note_received(self):
        """Note saved — pet appreciates new data."""
        with self._lock:
            self._hunger = min(1.0, self._hunger + PET_HUNGER_PER_NOTE)

    def on_dataset_prep(self):
        """Dataset preparation step ran."""
        with self._lock:
            self._cleanliness = min(1.0, self._cleanliness + PET_CLEAN_DATASET_PREP_BONUS)
            log.info("Dataset prep bonus: cleanliness now %.3f", self._cleanliness)

    def on_training_complete(self):
        """Training feast hunger bonus (intelligence update handled separately)."""
        with self._lock:
            self._hunger = min(1.0, self._hunger + PET_HUNGER_FEED_FEAST)

    def set_intelligence(self, score, details):
        """Update intelligence score and details."""
        with self._lock:
            self._intelligence = score
            self._intelligence_details = details

    def get_intelligence_details(self):
        with self._lock:
            return self._intelligence_details.copy()

    # ── Tick (called from main loop) ─────────────────────────────

    def tick(self, now, mood_score, is_charging=False):
        """Periodic vitals decay/regen. Returns True if a DB persist is due."""
        self._is_charging = is_charging
        should_persist = False

        with self._lock:
            if now - self._last_vital_tick > PET_VITAL_TICK_INTERVAL_S:
                self._tick_vitals_unlocked(now, mood_score)
                self._last_vital_tick = now

            if now - self._last_vital_persist > PET_VITAL_PERSIST_INTERVAL_S:
                self._save_unlocked()
                self._last_vital_persist = now
                should_persist = True

        return should_persist

    def save(self):
        """Force-persist vitals to DB (e.g. on shutdown)."""
        with self._lock:
            self._save_unlocked()

    # ── Coma queries ─────────────────────────────────────────────

    def get_coma_status(self):
        with self._lock:
            return {
                "is_coma": self._is_coma,
                "coma_entered_at": self._db.get_pet_state("coma_entered_at", ""),
                "hunger": round(self._hunger, 3),
                "cleanliness": round(self._cleanliness, 3),
                "energy": round(self._energy, 3),
                "revival_threshold": PET_COMA_REVIVAL_THRESHOLD,
                "hunger_ready": self._hunger >= PET_COMA_REVIVAL_THRESHOLD,
                "cleanliness_ready": self._cleanliness >= PET_COMA_REVIVAL_THRESHOLD,
                "energy_ready": self._energy >= PET_COMA_REVIVAL_THRESHOLD,
            }

    # ── Internal (must be called with lock held) ─────────────────

    @staticmethod
    def _is_nighttime():
        """Check if current local time is within nighttime sleep hours."""
        if not PET_SLEEP_ENABLED:
            return False
        hour = datetime.now().hour
        if PET_SLEEP_HOUR_START > PET_SLEEP_HOUR_END:
            # Wraps midnight, e.g. 23–7
            return hour >= PET_SLEEP_HOUR_START or hour < PET_SLEEP_HOUR_END
        else:
            return PET_SLEEP_HOUR_START <= hour < PET_SLEEP_HOUR_END

    @property
    def is_sleeping(self):
        """True if pet is sleeping (nighttime OR event-based, not coma)."""
        return (self._is_nighttime() or self._force_sleeping) and not self._is_coma

    def set_force_sleeping(self, sleeping):
        """Called by Heartbeat to signal event-based sleep (user, tuck_in, etc.)."""
        with self._lock:
            self._force_sleeping = sleeping

    def _tick_vitals_unlocked(self, now, mood_score):
        elapsed_h = PET_VITAL_TICK_INTERVAL_S / 3600.0

        # Sleep: no decay, boosted energy regen (nighttime OR event-based)
        sleeping = (self._is_nighttime() or self._force_sleeping) and not self._is_coma
        if sleeping:
            pass  # No hunger/cleanliness decay while sleeping
        else:
            self._hunger = max(0.0, self._hunger - PET_HUNGER_DECAY_PER_HOUR * elapsed_h)
            self._cleanliness = max(0.0, self._cleanliness - PET_CLEAN_DECAY_PER_HOUR * elapsed_h)

        idle_s = now - self._last_inference_time
        regen = PET_ENERGY_REGEN_PER_HOUR
        if idle_s > PET_ENERGY_IDLE_THRESHOLD_S:
            regen = PET_ENERGY_REGEN_IDLE_PER_HOUR
        if self._is_coma:
            regen *= PET_ENERGY_COMA_REGEN_MULT
        if sleeping:
            regen *= 2.0  # double energy regen during sleep
        if getattr(self, '_is_charging', False):
            regen *= 2.0  # faster regen when plugged in
        self._energy = min(1.0, self._energy + regen * elapsed_h)

        # Happiness: blend mood (40%) with average vitals health (60%)
        mood_factor = (mood_score + 1.0) / 2.0  # 0.0-1.0
        vitals_avg = (self._hunger + self._cleanliness + self._energy) / 3.0
        base_happy = 0.4 * mood_factor + 0.6 * vitals_avg
        penalty = 0.0
        for vital in (self._hunger, self._cleanliness, self._energy):
            if vital < PET_VITAL_CRITICAL:
                penalty += 0.10
            elif vital < PET_VITAL_LOW:
                penalty += 0.05
        bonus = 0.10 if all(v > 0.7 for v in (self._hunger, self._cleanliness, self._energy)) else 0.0
        self._happiness = max(0.0, min(1.0, base_happy - penalty + bonus))

        # Coma check
        if self._is_coma:
            self._check_revival_unlocked()
        else:
            self._check_coma_unlocked(now)

    def _check_coma_unlocked(self, now):
        critical_count = sum(
            1 for v in (self._hunger, self._cleanliness, self._energy)
            if v < PET_COMA_THRESHOLD
        )
        if critical_count >= PET_COMA_VITALS_REQUIRED:
            if self._coma_warning_start is None:
                self._coma_warning_start = now
                log.warning("Coma warning: %d vitals critical", critical_count)
            elif (now - self._coma_warning_start) / 3600.0 >= PET_COMA_DURATION_HOURS:
                self._enter_coma_unlocked()
        else:
            if self._coma_warning_start is not None:
                log.info("Coma warning cleared — vitals recovered")
            self._coma_warning_start = None

    def _enter_coma_unlocked(self):
        if self._is_coma:
            return
        log.warning("Pet entering COMA — vitals critically low")
        self._is_coma = True
        self._coma_warning_start = None
        trigger = {
            "hunger": round(self._hunger, 3),
            "cleanliness": round(self._cleanliness, 3),
            "energy": round(self._energy, 3),
        }
        self._coma_id = self._db.log_pet_coma(trigger)
        self._db.set_pet_state("is_coma", "1")
        self._db.set_pet_state("coma_id", str(self._coma_id))
        self._db.set_pet_state("coma_entered_at",
                               datetime.now(timezone.utc).isoformat())

    def _check_revival_unlocked(self):
        if not self._is_coma:
            return
        if (self._hunger >= PET_COMA_REVIVAL_THRESHOLD and
                self._cleanliness >= PET_COMA_REVIVAL_THRESHOLD and
                self._energy >= PET_COMA_REVIVAL_THRESHOLD):
            self._exit_coma_unlocked()

    def _exit_coma_unlocked(self):
        log.info("Pet WAKING UP from coma!")
        self._is_coma = False
        if self._coma_id:
            self._db.end_pet_coma(self._coma_id, revival_method="manual")
        self._coma_id = None
        self._db.set_pet_state("is_coma", "0")
        self._db.set_pet_state("coma_id", "")
        log.info("Coma ended — pet is awake (model will reload on next ask)")

    def _save_unlocked(self):
        self._db.set_pet_state("vital_hunger", str(round(self._hunger, 4)))
        self._db.set_pet_state("vital_cleanliness", str(round(self._cleanliness, 4)))
        self._db.set_pet_state("vital_energy", str(round(self._energy, 4)))
        self._db.set_pet_state("vital_happiness", str(round(self._happiness, 4)))
        self._db.set_pet_state("intelligence_score", str(round(self._intelligence, 2)))
        self._db.set_pet_state("is_coma", "1" if self._is_coma else "0")
        self._db.set_pet_state("vitals_last_tick",
                               datetime.now(timezone.utc).isoformat())
        self._db.log_pet_vitals(
            self._hunger, self._cleanliness, self._energy,
            self._happiness, self._intelligence, self._is_coma,
        )

    def _load_from_db(self):
        self._hunger = float(self._db.get_pet_state("vital_hunger", "1.0"))
        self._cleanliness = float(self._db.get_pet_state("vital_cleanliness", "1.0"))
        self._energy = float(self._db.get_pet_state("vital_energy", "1.0"))
        self._happiness = float(self._db.get_pet_state("vital_happiness", "0.5"))
        self._intelligence = float(self._db.get_pet_state("intelligence_score", "0.0"))
        self._is_coma = self._db.get_pet_state("is_coma", "0") == "1"
        self._total_feeds = int(self._db.get_pet_state("total_feeds", "0"))
        self._total_cleans = int(self._db.get_pet_state("total_cleans", "0"))

        details_json = self._db.get_pet_state("intelligence_details", "{}")
        try:
            self._intelligence_details = json.loads(details_json)
        except (json.JSONDecodeError, TypeError):
            self._intelligence_details = {}

        if self._is_coma:
            coma_id_str = self._db.get_pet_state("coma_id", "")
            self._coma_id = int(coma_id_str) if coma_id_str else None

        # Retroactive decay for offline time
        last_tick_str = self._db.get_pet_state("vitals_last_tick", "")
        if last_tick_str:
            try:
                last_tick_dt = datetime.fromisoformat(last_tick_str)
                now_dt = datetime.now(timezone.utc)
                elapsed_hours = (now_dt - last_tick_dt).total_seconds() / 3600.0
                if elapsed_hours > 0.1:
                    self._apply_decay_unlocked(elapsed_hours)
                    log.info(
                        "Applied %.1fh offline decay: hunger=%.2f clean=%.2f energy=%.2f",
                        elapsed_hours, self._hunger, self._cleanliness, self._energy,
                    )
            except (ValueError, TypeError):
                pass

        log.info(
            "Loaded vitals: hunger=%.2f clean=%.2f energy=%.2f happy=%.2f "
            "iq=%.1f coma=%s feeds=%d cleans=%d",
            self._hunger, self._cleanliness, self._energy, self._happiness,
            self._intelligence, self._is_coma, self._total_feeds, self._total_cleans,
        )

    def _apply_decay_unlocked(self, elapsed_hours):
        self._hunger = max(0.0, self._hunger - PET_HUNGER_DECAY_PER_HOUR * elapsed_hours)
        self._cleanliness = max(0.0, self._cleanliness - PET_CLEAN_DECAY_PER_HOUR * elapsed_hours)
        regen_rate = PET_ENERGY_REGEN_PER_HOUR
        if self._is_coma:
            regen_rate *= PET_ENERGY_COMA_REGEN_MULT
        self._energy = min(1.0, self._energy + regen_rate * elapsed_hours)


class PetEngine:
    """Non-blocking LLM inference engine for the Cortex Pet.

    Parameters
    ----------
    db : CortexDB
        Database handle for reading/writing pet state and interactions.
    """

    def __init__(self, db, battery=None):
        self._db = db
        self._battery = battery
        self._model_loaded = False

        # Queues
        self._prompt_queue = Queue(maxsize=4)
        self._result_queue = Queue(maxsize=16)
        self._recent_responses = []  # last N responses (survives queue drain)

        # Mood/stage (protected by _lock)
        self._mood_score = 0.0
        self._mood_label = "neutral"
        self._stage = 0
        self._interaction_count = 0
        self._last_mood_update = 0.0
        self._lock = threading.Lock()

        # Thread-safe vitals
        self._vitals = PetVitals(db)

        # Bloom (training cycle counter) — loaded from DB
        bloom_str = db.get_pet_state("bloom_number")
        self._bloom = int(bloom_str) if bloom_str else 0

        # Intelligence tracking (main thread only)
        self._vocab_buffer = []
        self._interactions_since_iq_calc = 0

        # Background inference thread
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="pet-inference"
        )
        self._thread.start()

        # Initialize state from DB
        self._refresh_state()

        log.info("PetEngine initialized (model will lazy-load on first ask)")

    # ── Public interface ────────────────────────────────────────────────

    def ask(self, user_text, session_id=None):
        """Queue a prompt for background inference.

        Returns the interaction ID (from DB) or None if queue is full.
        During coma, the prompt is logged but the pet returns a canned "zzz"
        response. Talking during coma still restores some hunger.
        """
        if not PET_ENABLED:
            return None

        sentiment = simple_sentiment(user_text)

        # Interaction always restores hunger (pet appreciates being talked to)
        self._vitals.restore_hunger(PET_HUNGER_PER_INTERACTION)

        # Bad sentiment dirties the data
        if sentiment < -0.3:
            self._vitals.penalize_cleanliness(PET_CLEAN_PENALTY_BAD_SENTIMENT)

        # Insert interaction record (response filled in later)
        interaction_id = self._db.insert_pet_interaction(
            prompt=user_text,
            sentiment_score=sentiment,
            stage=self._stage,
            mood=self._mood_label,
            session_id=session_id,
        )

        # Coma: log the interaction but don't run inference
        if self._vitals.is_coma:
            coma_response = "*...zzz...* (pet is in a deep sleep)"
            self._db.update_pet_interaction(interaction_id, coma_response, 0, 0)
            self._result_queue.put({
                "id": interaction_id,
                "response": coma_response,
                "inference_time_ms": 0,
                "tokens": 0,
                "mood": "coma",
            })
            log.info("Pet in coma — returning zzz for prompt #%d", interaction_id)
            return interaction_id

        try:
            self._prompt_queue.put_nowait({
                "id": interaction_id,
                "text": user_text,
                "sentiment": sentiment,
                "session_id": session_id,
            })
            log.info("Queued pet prompt #%d (sentiment=%.2f)", interaction_id, sentiment)
            return interaction_id
        except Exception:
            log.warning("Pet prompt queue full — dropping request")
            return interaction_id  # still logged, just won't get a response

    def get_response(self):
        """Non-blocking poll for the oldest completed response.

        Returns dict with keys: id, response, inference_time_ms, tokens, mood
        or None if nothing ready.  Also saves to _recent_responses so the Hub
        can retrieve it even after the display consumes the queue.
        """
        try:
            resp = self._result_queue.get_nowait()
            self._recent_responses.append(resp)
            # Keep only last 20 responses
            if len(self._recent_responses) > 20:
                self._recent_responses = self._recent_responses[-20:]
            return resp
        except Empty:
            return None

    def get_all_responses(self):
        """Drain all completed responses (also saved to recent list)."""
        results = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except Empty:
                break
        if results:
            self._recent_responses.extend(results)
            if len(self._recent_responses) > 20:
                self._recent_responses = self._recent_responses[-20:]
        return results

    def get_recent_responses(self, since_id=None):
        """Get recent responses without draining the queue.

        Used by the Hub to retrieve responses that the display already consumed.
        If since_id is provided, returns only responses with id > since_id.
        """
        if since_id is not None:
            return [r for r in self._recent_responses if r.get("id", 0) > since_id]
        return list(self._recent_responses)

    def tick(self):
        """Called from main loop each frame. Non-blocking bookkeeping.

        Handles mood refresh, vitals decay/regen, coma checks, and
        periodic persistence of vitals to the database.
        """
        now = time.monotonic()

        # Mood + stage refresh
        if now - self._last_mood_update > PET_MOOD_UPDATE_INTERVAL_S:
            self._refresh_state()
            self._last_mood_update = now

        # Vitals tick (decay/regen/persist — all lock-protected inside PetVitals)
        is_charging = (self._battery and self._battery.available
                       and self._battery.is_charging)
        self._vitals.tick(now, self._mood_score, is_charging=bool(is_charging))

        # If vitals just entered coma, drain prompt queue
        if self._vitals.is_coma and self._model_loaded:
            self._model_loaded = False
            log.info("Coma entered — inference paused (llama-server still running)")
            while not self._prompt_queue.empty():
                try:
                    self._prompt_queue.get_nowait()
                except Empty:
                    break

    @property
    def is_thinking(self):
        """True if inference is currently running."""
        return not self._prompt_queue.empty()

    @property
    def mood(self):
        return self._mood_label

    @property
    def mood_score(self):
        return self._mood_score

    @property
    def stage(self):
        return self._stage

    @property
    def stage_name(self):
        if 0 <= self._stage < len(PET_STAGE_NAMES):
            return PET_STAGE_NAMES[self._stage]
        return "Unknown"

    @property
    def interaction_count(self):
        return self._interaction_count

    @property
    def model_loaded(self):
        return self._model_loaded

    @property
    def hunger(self):
        return self._vitals.hunger

    @property
    def cleanliness(self):
        return self._vitals.cleanliness

    @property
    def energy(self):
        return self._vitals.energy

    @property
    def happiness(self):
        return self._vitals.happiness

    @property
    def is_coma(self):
        return self._vitals.is_coma

    @property
    def intelligence(self):
        return self._vitals.intelligence

    @property
    def bloom(self):
        return self._bloom

    def get_stats(self):
        """Return full pet statistics dict including vitals."""
        stats = {
            "enabled": PET_ENABLED,
            "model_loaded": self._model_loaded,
            "model_path": PET_MODEL_PATH,
            "llama_server_url": PET_LLAMA_SERVER_URL,
            "lora_path": PET_LORA_PATH,
            "stage": self._stage,
            "stage_name": self.stage_name,
            "mood": self._mood_label,
            "mood_score": round(self._mood_score, 3),
            "interaction_count": self._interaction_count,
            "bloom": self._bloom,
            "pending_prompts": self._prompt_queue.qsize(),
            "pending_responses": self._result_queue.qsize(),
            "is_thinking": self.is_thinking,
        }
        stats.update(self._vitals.get_snapshot())
        return stats

    def get_vitals(self):
        """Return just the vitals subset for quick queries."""
        return self._vitals.get_snapshot()

    # ── Tamagotchi Actions ───────────────────────────────────────────

    def feed(self, feed_type="chat_snack"):
        """Feed the pet. Returns new hunger value."""
        return self._vitals.feed(feed_type)

    def clean(self, discard_ids=None):
        """Clean the pet. Returns (new_cleanliness, discarded_count)."""
        return self._vitals.clean(discard_ids)

    def rest(self):
        """Rest the pet — boost energy. Useful during coma revival."""
        return self._vitals.rest()

    def on_note_received(self):
        """Called when a note is saved — the pet appreciates new data."""
        self._vitals.on_note_received()

    def on_dataset_prep(self):
        """Called when the dataset preparation step runs."""
        self._vitals.on_dataset_prep()

    def on_training_complete(self, training_metrics=None):
        """Called when LoRA training completes and is deployed."""
        self._vitals.on_training_complete()
        if training_metrics:
            self._update_intelligence(training_metrics)
            # Update bloom number from lora_version (e.g. "bloom-18" or "18")
            lora_ver = str(training_metrics.get("lora_version", ""))
            bloom_num = None
            if lora_ver.startswith("bloom-"):
                try:
                    bloom_num = int(lora_ver.split("-", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif lora_ver.isdigit():
                bloom_num = int(lora_ver)
            if bloom_num is not None:
                self._bloom = bloom_num
                self._db.set_pet_state("bloom_number", str(bloom_num))
                log.info("Bloom updated to %d", bloom_num)
            log.info("Training complete: intelligence now %.1f, bloom %d",
                     self._vitals.intelligence, self._bloom)

    def cleanup(self):
        """Stop the inference thread."""
        log.info("PetEngine shutting down...")
        self._running = False
        # Persist vitals one final time
        self._vitals.save()
        # Unblock the thread if it's waiting on the queue
        try:
            self._prompt_queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=5)
        self._model_loaded = False
        log.info("PetEngine shutdown complete")

    def get_coma_status(self):
        """Get detailed coma status for display."""
        return self._vitals.get_coma_status()

    # ── Intelligence System ──────────────────────────────────────────

    def _track_vocab(self, response_text):
        """Track vocabulary diversity from a response."""
        if not response_text:
            return
        words = set(response_text.lower().split())
        self._vocab_buffer.append(words)
        if len(self._vocab_buffer) > PET_INTELLIGENCE_VOCAB_BUFFER:
            self._vocab_buffer.pop(0)
        self._interactions_since_iq_calc += 1

        # Auto-recalculate intelligence periodically
        if self._interactions_since_iq_calc >= PET_INTELLIGENCE_RECALC_INTERVAL:
            self._recalculate_intelligence()
            self._interactions_since_iq_calc = 0

    def _map_range(self, value, in_min, in_max, out_min=0.0, out_max=100.0):
        """Map a value from one range to another, clamped."""
        if in_max == in_min:
            return out_min
        ratio = (value - in_min) / (in_max - in_min)
        ratio = max(0.0, min(1.0, ratio))
        return out_min + ratio * (out_max - out_min)

    def _recalculate_intelligence(self):
        """Recalculate composite intelligence score from all components."""
        details = {}

        # 1. Training loss (from last training)
        loss_score = 0.0
        training_hist = self._db.get_pet_training_history(limit=1)
        if training_hist:
            latest = training_hist[0]
            if latest.get("final_loss") is not None:
                loss_min, loss_max = PET_IQ_LOSS_RANGE  # (2.0, 0.5)
                # Lower loss = higher score, so reverse the mapping
                loss_score = self._map_range(
                    latest["final_loss"], loss_min, loss_max
                )
            if latest.get("perplexity_base") and latest.get("perplexity_finetuned"):
                ratio = latest["perplexity_base"] / max(
                    0.01, latest["perplexity_finetuned"]
                )
                ppl_min, ppl_max = PET_IQ_PERPLEXITY_RANGE
                details["perplexity"] = round(
                    self._map_range(ratio, ppl_min, ppl_max), 1
                )
        details["loss"] = round(loss_score, 1)

        # 2. Perplexity improvement (default 0 if no training data)
        if "perplexity" not in details:
            details["perplexity"] = 0.0

        # 3. Vocabulary diversity
        vocab_score = 0.0
        if self._vocab_buffer:
            all_words = set()
            total_count = 0
            for word_set in self._vocab_buffer:
                all_words.update(word_set)
                total_count += len(word_set)
            diversity = len(all_words) / max(1, total_count)
            vmin, vmax = PET_IQ_VOCAB_RANGE
            vocab_score = self._map_range(diversity, vmin, vmax)
        details["vocab"] = round(vocab_score, 1)

        # 4. Response coherence (appropriate length relative to stage)
        coherence_score = 50.0  # default midpoint
        if self._vocab_buffer:
            avg_words = sum(len(ws) for ws in self._vocab_buffer) / len(self._vocab_buffer)
            # Stage expectations: [3, 10, 20, 30, 40] average words
            stage_targets = [3, 10, 20, 30, 40]
            target = stage_targets[min(self._stage, len(stage_targets) - 1)]
            # Closer to target = higher coherence
            deviation = abs(avg_words - target) / max(1, target)
            coherence_score = max(0, 100 - deviation * 100)
        details["coherence"] = round(coherence_score, 1)

        # 5. Training data volume
        data_score = 0.0
        try:
            # Count curated examples from DB
            count = self._db.get_pet_interaction_count()
            dmin, dmax = PET_IQ_DATA_RANGE
            data_score = self._map_range(count, dmin, dmax)
        except Exception:
            pass
        details["data_volume"] = round(data_score, 1)

        # Weighted composite
        score = (
            details["loss"] * PET_IQ_WEIGHT_LOSS +
            details["perplexity"] * PET_IQ_WEIGHT_PERPLEXITY +
            details["vocab"] * PET_IQ_WEIGHT_VOCAB +
            details["coherence"] * PET_IQ_WEIGHT_COHERENCE +
            details["data_volume"] * PET_IQ_WEIGHT_DATA_VOLUME
        )

        final_score = round(max(0.0, min(100.0, score)), 1)
        self._vitals.set_intelligence(final_score, details)
        self._db.set_pet_state("intelligence_score",
                               str(round(final_score, 2)))
        self._db.set_pet_state("intelligence_details", json.dumps(details))
        log.info("Intelligence recalculated: %.1f %s", final_score, details)

    def _update_intelligence(self, training_metrics):
        """Update intelligence from fresh training metrics.

        training_metrics dict expected keys:
        - final_loss, perplexity_base, perplexity_finetuned,
          lora_version, training_time_s, dataset_size
        """
        # Log training event
        self._db.log_pet_training(
            lora_version=training_metrics.get("lora_version", "unknown"),
            final_loss=training_metrics.get("final_loss"),
            perplexity_base=training_metrics.get("perplexity_base"),
            perplexity_finetuned=training_metrics.get("perplexity_finetuned"),
            intelligence_score=None,  # will be filled after recalc
            training_time_s=training_metrics.get("training_time_s"),
            dataset_size=training_metrics.get("dataset_size"),
        )
        # Recalculate with fresh data
        self._recalculate_intelligence()
        # Update the training history record with the new score
        hist = self._db.get_pet_training_history(limit=1)
        if hist:
            # Update the most recent record's intelligence_score
            self._db._conn.execute(
                "UPDATE pet_training_history SET intelligence_score=? WHERE id=?",
                (self._vitals.intelligence, hist[0]["id"]),
            )
            self._db._conn.commit()

    def get_intelligence(self):
        """Return intelligence breakdown for display."""
        return {
            "score": round(self._vitals.intelligence, 1),
            "details": self._vitals.get_intelligence_details(),
            "history": self._db.get_pet_training_history(limit=5),
        }

    # ── Internal ────────────────────────────────────────────────────────

    def _refresh_state(self):
        """Re-read mood and stage from DB."""
        with self._lock:
            self._mood_score = self._db.get_pet_mood_score(PET_MOOD_WINDOW)
            self._mood_label = self._score_to_mood(self._mood_score)
            self._interaction_count = self._db.get_pet_interaction_count()
            self._stage = self._count_to_stage(self._interaction_count)

    @staticmethod
    def _score_to_mood(score):
        """Map sentiment score [-1, 1] to mood label."""
        if score > 0.4:
            return "happy"
        elif score > 0.15:
            return "content"
        elif score > -0.15:
            return "neutral"
        elif score > -0.4:
            return "uneasy"
        else:
            return "sad"

    @staticmethod
    def _count_to_stage(count):
        """Map interaction count to evolution stage (0-4)."""
        stage = 0
        for i, threshold in enumerate(PET_STAGE_THRESHOLDS):
            if count >= threshold:
                stage = i
        return stage

    def _mood_temperature(self):
        """Adjust temperature based on mood (kind → more coherent)."""
        base = PET_TEMPERATURE
        if self._mood_label == "happy":
            return max(0.4, base - 0.3)
        elif self._mood_label == "content":
            return max(0.5, base - 0.15)
        elif self._mood_label in ("uneasy", "sad"):
            return min(1.5, base + 0.3)
        return base

    def _mood_max_tokens(self):
        """Adjust max tokens based on mood.

        Sad/uneasy → 64 (short, withdrawn)
        Neutral/content → 128 (soft default)
        Happy → 256 (full robust responses)
        """
        if self._mood_label in ("uneasy", "sad"):
            return 64
        elif self._mood_label == "happy":
            return 256
        return PET_MAX_TOKENS  # 128 (neutral/content)

    def _build_system_prompt(self):
        """Construct system prompt from stage + mood."""
        stage = min(self._stage, len(_STAGE_PROMPTS) - 1)
        base = _STAGE_PROMPTS[stage].format(name=PET_NAME)
        mood_mod = _MOOD_MODIFIERS.get(self._mood_label, "")
        return mood_mod + base

    def _check_server(self):
        """Check if llama-server is reachable. Returns True if healthy."""
        try:
            url = PET_LLAMA_SERVER_URL.rstrip("/") + "/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                ok = data.get("status") == "ok"
                if ok and not self._model_loaded:
                    self._model_loaded = True
                    log.info("llama-server is healthy at %s", PET_LLAMA_SERVER_URL)
                return ok
        except Exception as e:
            if self._model_loaded:
                log.warning("llama-server unreachable: %s", e)
                self._model_loaded = False
            return False

    def _http_chat(self, messages, temperature, max_tokens):
        """Send a chat completion request to llama-server. Returns parsed JSON."""
        url = PET_LLAMA_SERVER_URL.rstrip("/") + "/v1/chat/completions"
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": PET_TOP_P,
            "repeat_penalty": PET_REPEAT_PENALTY,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def _inference_loop(self):
        """Background thread: pull prompts, run inference via llama-server HTTP."""
        log.info("Inference thread started (HTTP backend: %s)", PET_LLAMA_SERVER_URL)

        while self._running:
            try:
                item = self._prompt_queue.get(timeout=1.0)
            except Empty:
                continue

            # Shutdown sentinel
            if item is None:
                break

            interaction_id = item["id"]
            user_text = item["text"]

            # Check server health
            if not self._check_server():
                self._result_queue.put({
                    "id": interaction_id,
                    "response": "*yawns* (model not available)",
                    "inference_time_ms": 0,
                    "tokens": 0,
                    "mood": self._mood_label,
                    "error": "server_unreachable",
                })
                continue

            # Build prompt
            system_prompt = self._build_system_prompt()
            temperature = self._mood_temperature()
            max_tokens = self._mood_max_tokens()

            try:
                log.info(
                    "Inference #%d: temp=%.2f, max_tok=%d, mood=%s, stage=%d",
                    interaction_id, temperature, max_tokens,
                    self._mood_label, self._stage,
                )
                start = time.monotonic()

                result = self._http_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                elapsed_ms = int((time.monotonic() - start) * 1000)
                response_text = ""
                tokens_generated = 0

                if (result and "choices" in result and len(result["choices"]) > 0):
                    choice = result["choices"][0]
                    msg = choice.get("message", {})
                    response_text = msg.get("content", "").strip()
                    # Strip stray thinking tags from Qwen3.5
                    for tag in ("</think>", "<think>"):
                        response_text = response_text.replace(tag, "").strip()
                    usage = result.get("usage", {})
                    tokens_generated = usage.get("completion_tokens", 0)

                # Update DB
                self._db.update_pet_interaction(
                    interaction_id, response_text, elapsed_ms, tokens_generated
                )

                # ── Vitals: energy cost + vocab tracking ─────────
                energy_cost = min(
                    PET_ENERGY_COST_CAP,
                    elapsed_ms * PET_ENERGY_COST_PER_INFERENCE_MS,
                )
                self._vitals.drain_energy(energy_cost)
                self._track_vocab(response_text)

                log.info(
                    "Inference #%d complete: %d tokens in %dms (energy -%.3f)",
                    interaction_id, tokens_generated, elapsed_ms, energy_cost,
                )

                self._result_queue.put({
                    "id": interaction_id,
                    "response": response_text,
                    "inference_time_ms": elapsed_ms,
                    "tokens": tokens_generated,
                    "mood": self._mood_label,
                })

            except Exception as e:
                log.error("Inference #%d failed: %s", interaction_id, e)
                # Failed inference dirties the data
                self._vitals.penalize_cleanliness(PET_CLEAN_PENALTY_FAILED_INFERENCE)
                self._result_queue.put({
                    "id": interaction_id,
                    "response": "*confused noises* (error)",
                    "inference_time_ms": 0,
                    "tokens": 0,
                    "mood": self._mood_label,
                    "error": str(e),
                })

        log.info("Inference thread stopped")
