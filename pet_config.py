"""Pet plugin configuration constants.

Moved from cortex-core/src/config.py during slice 2c1a. The pet plugin
files import from this module instead of core's config so cortex-core's
config.py can shed pet-specific knobs.

Slice 2c2 (or later) may consolidate further into plugin.toml's [config]
section, but the dataclass-style import surface here is the lowest-risk
intermediate — pet.py, heartbeat.py, etc. just swap `from config` for
`from pet_config` and otherwise read the constants exactly as before.
"""

import os

# ── Paths anchored to this file ─────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# plugins/pet -> plugins -> cortex-core -> /home/turfptax (Pi) or repo root (dev)
_HOME = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))


# ── Pet model / LLM ─────────────────────────────────────────────────
PET_ENABLED = True
PET_MODEL_DIR = os.path.join(_HOME, "models")
PET_MODEL_PATH = os.path.join(PET_MODEL_DIR, "qwen3.5-0.8b-q4_k_m.gguf")
PET_LORA_PATH = ""                  # set to adapter file path when available
PET_CONTEXT_SIZE = 2048             # context window tokens
PET_MAX_TOKENS = 128                # max generation length per response
PET_THREADS = 3                     # CPU threads for inference
PET_TEMPERATURE = 0.8               # base temperature (adjusted by mood)
PET_REPEAT_PENALTY = 1.1
PET_TOP_P = 0.9
PET_SEED = -1                       # -1 = random
PET_LLAMA_SERVER_URL = "http://127.0.0.1:8081"
PET_NAME = "Cortex Pet"
PET_MOOD_WINDOW = 20                # rolling interaction window for mood score
PET_MOOD_UPDATE_INTERVAL_S = 60     # seconds between mood recalculations

# ── Pet evolution stages ───────────────────────────────────────────
PET_STAGE_THRESHOLDS = [0, 50, 200, 1000, 5000]
PET_STAGE_NAMES = ["Primordial", "Babbling", "Echoing", "Responding", "Conversing"]

# ── Mood colors (neon RGB) ─────────────────────────────────────────
COLOR_PET_HAPPY = (0, 255, 140)         # Neon green
COLOR_PET_NEUTRAL = (0, 200, 255)       # Neon cyan-blue
COLOR_PET_SAD = (255, 60, 100)          # Neon pink-red

# ── Sprites ────────────────────────────────────────────────────────
SPRITE_DIR = os.path.join(_THIS_DIR, "assets", "sprites")
PET_SPRITE_SIZE = 80

# ── Tamagotchi UI colors ───────────────────────────────────────────
COLOR_CURSOR = (0, 255, 255)            # Neon cyan cursor
COLOR_MENU_BG = (12, 12, 28)            # Dark panel background
COLOR_HIGHLIGHT = (20, 30, 60)          # Selected item highlight
COLOR_SPEECH_BG = (14, 14, 30)          # Speech bubble interior
COLOR_XP_BAR = (0, 200, 255)            # Cyan XP fill
COLOR_XP_BAR_BG = (16, 16, 32)          # Dark XP trough

# Vital bar colors
COLOR_VITAL_HUNGER = (255, 160, 0)      # Neon orange
COLOR_VITAL_HUNGER_LOW = (255, 60, 0)   # Red-orange critical
COLOR_VITAL_CLEAN = (0, 160, 255)       # Electric blue
COLOR_VITAL_CLEAN_LOW = (160, 80, 0)    # Amber-brown when dirty
COLOR_VITAL_ENERGY = (255, 255, 0)      # Neon yellow
COLOR_VITAL_ENERGY_LOW = (255, 40, 40)  # Neon red critical
COLOR_VITAL_IQ = (200, 0, 255)          # Neon purple

# ── Pet auto-dismiss / sleep timers ────────────────────────────────
PET_RESPONSE_DISMISS_S = 15             # auto-dismiss pet response (seconds)
PET_SLEEP_IDLE_S = 20                   # seconds of no input before sleeping animation

# ── Pet vitals (Tamagotchi system) ─────────────────────────────────
PET_VITAL_TICK_INTERVAL_S = 300         # decay/regen check every 5 minutes
PET_VITAL_PERSIST_INTERVAL_S = 300      # save vitals to DB every 5 minutes

# Hunger
PET_HUNGER_DECAY_PER_HOUR = 0.04        # empty in ~25h of total neglect
PET_HUNGER_PER_INTERACTION = 0.08       # restored per pet.ask() call
PET_HUNGER_PER_NOTE = 0.03              # restored when a note is received
PET_HUNGER_FEED_SNACK = 0.15            # "chat snack" quick feed
PET_HUNGER_FEED_MEAL = 0.25             # "data meal" feed
PET_HUNGER_FEED_FEAST = 0.40            # "training feast"
PET_HUNGER_FEED_COMA = 0.15             # feeding during coma

# Cleanliness
PET_CLEAN_DECAY_PER_HOUR = 0.02         # dirty in ~50h
PET_CLEAN_PENALTY_BAD_SENTIMENT = 0.05
PET_CLEAN_PENALTY_FAILED_INFERENCE = 0.08
PET_CLEAN_PER_DISCARD = 0.05
PET_CLEAN_DATASET_PREP_BONUS = 0.30
PET_CLEAN_COMA_BONUS = 0.20

# Energy (inference budget)
PET_ENERGY_COST_PER_INFERENCE_MS = 0.000005
PET_ENERGY_COST_CAP = 0.05
PET_ENERGY_REGEN_PER_HOUR = 0.05
PET_ENERGY_REGEN_IDLE_PER_HOUR = 0.10
PET_ENERGY_IDLE_THRESHOLD_S = 1800
PET_ENERGY_COMA_REGEN_MULT = 2.0
PET_ENERGY_REST_BOOST = 0.10

# Nighttime / dynamic sleep
PET_SLEEP_HOUR_START = 23
PET_SLEEP_HOUR_END = 7
PET_SLEEP_ENABLED = True
PET_SLEEP_AUTO_ENERGY_THRESHOLD = 0.15
PET_SLEEP_AUTO_DELAY_S = 600
PET_USER_SLEEP_ENABLED = True

# Vital thresholds
PET_VITAL_LOW = 0.30
PET_VITAL_CRITICAL = 0.15
PET_VITAL_ALERT_DEBOUNCE_S = 600

# ── Coma system ────────────────────────────────────────────────────
PET_COMA_THRESHOLD = 0.10
PET_COMA_VITALS_REQUIRED = 2
PET_COMA_DURATION_HOURS = 2
PET_COMA_REVIVAL_THRESHOLD = 0.30

# ── Intelligence (IQ) ──────────────────────────────────────────────
PET_INTELLIGENCE_RECALC_INTERVAL = 100
PET_INTELLIGENCE_VOCAB_BUFFER = 100
PET_IQ_WEIGHT_LOSS = 0.30
PET_IQ_WEIGHT_PERPLEXITY = 0.25
PET_IQ_WEIGHT_VOCAB = 0.15
PET_IQ_WEIGHT_COHERENCE = 0.15
PET_IQ_WEIGHT_DATA_VOLUME = 0.15
PET_IQ_LOSS_RANGE = (2.0, 0.5)
PET_IQ_PERPLEXITY_RANGE = (1.0, 2.0)
PET_IQ_VOCAB_RANGE = (0.2, 0.6)
PET_IQ_DATA_RANGE = (0, 500)

# ── Heartbeat (autonomous pet life) ────────────────────────────────
HEARTBEAT_ENABLED = True
HEARTBEAT_INTERVAL_S = 1800
HEARTBEAT_INTERVAL_LOW_ENERGY_MULT = 2
HEARTBEAT_INTERVAL_CRITICAL_MULT = 4
HEARTBEAT_ENERGY_THRESHOLD_LOW = 0.6
HEARTBEAT_ENERGY_THRESHOLD_CRITICAL = 0.3
HEARTBEAT_MAX_TOKENS = 64
HEARTBEAT_TEMPERATURE = 0.9
HEARTBEAT_MAX_ROUNDS = 10
HEARTBEAT_THOUGHT_DISPLAY_S = 8

# ── Battery-energy blending (pet-side mixing only) ─────────────────
# The real battery thresholds (BATTERY_FORCE_SLEEP_PCT, BATTERY_CRITICAL_PCT,
# BATTERY_ENABLED, BATTERY_POLL_INTERVAL_S, BATTERY_I2C_BUS) stay in core
# config.py because they describe hardware behavior, not pet behavior.
BATTERY_ENERGY_WEIGHT = 0.6             # 60% real battery in blended energy
INFERENCE_ENERGY_WEIGHT = 0.4           # 40% inference budget
BATTERY_DREAM_MIN_PCT = 90              # min battery % to start dreaming

# ── Dream training (sleep-triggered LoRA) ──────────────────────────
DREAM_ENABLED = True
DREAM_MIN_INTERACTIONS = 20
DREAM_COOLDOWN_HOURS = 1
DREAM_HUB_TIMEOUT_S = 5
