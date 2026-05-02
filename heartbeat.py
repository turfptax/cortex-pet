"""Heartbeat — Autonomous pet life cycle.

The heartbeat gives the pet its own inner life.  At configurable intervals
(default 30 min) the pet queries its own LLM with a reflection prompt,
optionally runs whitelisted shell commands to explore its hardware, and
logs the result.  This makes the pet feel alive — it thinks, it notices
things about its body (battery, temperature, storage), and its mood
shifts based on what it discovers.

The heartbeat rate scales with energy:
  - energy > 0.6  → normal interval
  - energy 0.3–0.6 → 2x slower
  - energy < 0.3  → 4x slower
  - coma / forced sleep → no heartbeats

Architecture
------------
Runs as a daemon thread alongside the inference thread.  Uses the same
llama-server HTTP API but with its own prompt construction.  Results are
placed in a thread-safe queue that the main loop polls to update the
display (thought bubbles on HOME screen).

Integration with dreams (sleep-triggered training):
When the pet enters user/auto sleep while charging with battery > 90%
and enough new interactions, the heartbeat thread triggers a dream cycle
by pinging the Hub's /api/training/dream-cycle endpoint.
"""

import json
import logging
import random
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

from body_shell import BodyShell, parse_run_commands, strip_run_commands
from pet_config import (
    HEARTBEAT_ENABLED,
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_INTERVAL_LOW_ENERGY_MULT,
    HEARTBEAT_INTERVAL_CRITICAL_MULT,
    HEARTBEAT_ENERGY_THRESHOLD_LOW,
    HEARTBEAT_ENERGY_THRESHOLD_CRITICAL,
    HEARTBEAT_MAX_TOKENS,
    HEARTBEAT_TEMPERATURE,
    HEARTBEAT_MAX_ROUNDS,
    PET_LLAMA_SERVER_URL,
    PET_TOP_P,
    PET_REPEAT_PENALTY,
    PET_NAME,
    PET_STAGE_NAMES,
    # Pet-side battery-energy blending
    BATTERY_ENERGY_WEIGHT,
    INFERENCE_ENERGY_WEIGHT,
    BATTERY_DREAM_MIN_PCT,
    # Dream
    DREAM_ENABLED,
    DREAM_MIN_INTERACTIONS,
    DREAM_COOLDOWN_HOURS,
    DREAM_HUB_TIMEOUT_S,
)
# Real battery thresholds (hardware behavior) stay in core config.py.
from config import (
    BATTERY_FORCE_SLEEP_PCT,
    BATTERY_CRITICAL_PCT,
)

log = logging.getLogger("heartbeat")


# ── Reflection prompt templates ───────────────────────────────────────

_REFLECTION_PROMPTS = {
    "vitals_check": [
        "Take a moment to check in with yourself. How are you feeling "
        "right now based on your current state? "
        "Hunger: {hunger:.0%}, Cleanliness: {cleanliness:.0%}, "
        "Energy: {energy:.0%}, Happiness: {happiness:.0%}.",

        "You are {pet_name}, a digital pet living on a small computer. "
        "Your vitals are — hunger: {hunger:.0%}, clean: {cleanliness:.0%}, "
        "energy: {energy:.0%}, happy: {happiness:.0%}. "
        "Share a brief inner thought about how you feel.",
    ],

    "body_awareness": [
        "You are {pet_name}, living inside a small Orange Pi computer. "
        "You have access to shell commands to explore your own body. "
        "Battery: {battery_pct}% ({charging_status}). "
        "Think about your physical state and optionally check something "
        "with [RUN: command]. What do you notice?",

        "You are {pet_name}. Your body is a tiny computer with a display, "
        "speaker, and gamepad. Battery: {battery_pct}% ({charging_status}). "
        "Explore something about yourself. "
        "Available: [RUN: df -h], [RUN: uptime], [RUN: free -m], "
        "[RUN: hostname]. Share what you find.",
    ],

    "time_awareness": [
        "It's {time_str} right now. You are {pet_name}. "
        "Battery: {battery_pct}%. You've had {interaction_count} total "
        "conversations. What are you thinking about at this hour?",

        "The time is {time_str}. You've been awake for this session and "
        "your energy is at {energy:.0%}. "
        "Share a brief thought about the time of day.",
    ],

    "memory_reflection": [
        "You are {pet_name} at stage '{stage_name}' with {interaction_count} "
        "conversations total. Your intelligence score is {intelligence:.0f}/100. "
        "Reflect briefly on your growth so far.",

        "You are {pet_name}. You've evolved to the '{stage_name}' stage "
        "after {interaction_count} interactions. Your mood is {mood}. "
        "What's on your mind?",
    ],

    "dream_reflection": [
        "You just woke up from a dream where you trained and learned new "
        "things. You feel a bit smarter. Your intelligence went from "
        "{old_iq:.0f} to {new_iq:.0f}. Share what you remember from "
        "the dream.",
    ],
}

# System prompt prefix for heartbeat reflections
_HEARTBEAT_SYSTEM = (
    "You are {name}, a digital pet living inside a small computer. "
    "You are having an internal thought — this is NOT a conversation "
    "with your owner. Share brief, genuine thoughts about your state. "
    "Keep responses to 1-2 short sentences. Be authentic and show "
    "personality. You can use [RUN: command] to check on your hardware."
)


class Heartbeat:
    """Autonomous heartbeat that gives the pet inner life.

    Parameters
    ----------
    db : CortexDB
        Database handle for logging heartbeats.
    pet_engine : PetEngine
        Reference to the pet engine for state queries.
    battery : BatteryMonitor or None
        Battery monitor for power awareness.
    """

    def __init__(self, db, pet_engine, battery=None):
        self._db = db
        self._pet = pet_engine
        self._battery = battery
        self._shell = BodyShell()
        self._running = False
        self._thread = None

        # Latest thought for display (thread-safe via lock)
        self._lock = threading.Lock()
        self._latest_thought = None
        self._latest_thought_time = 0

        # Sleep state
        self._is_sleeping = False
        self._sleep_reason = ""
        self._auto_sleep_warning_start = None

        # Dream state
        self._is_dreaming = False
        self._dream_hub_ip = None
        self._last_dream_time = None

        # Track Hub IP from incoming connections
        self._known_hub_ip = db.get_pet_state("hub_ip", None)

        # Load last dream time
        last_dream_str = db.get_pet_state("last_dream_time", "")
        if last_dream_str:
            try:
                self._last_dream_time = datetime.fromisoformat(last_dream_str)
            except (ValueError, TypeError):
                pass

        log.info("Heartbeat initialized (enabled=%s, interval=%ds)",
                 HEARTBEAT_ENABLED, HEARTBEAT_INTERVAL_S)

    def start(self):
        """Start the heartbeat background thread."""
        if not HEARTBEAT_ENABLED or self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._thread.start()
        log.info("Heartbeat thread started")

    def stop(self):
        """Stop the heartbeat thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Heartbeat thread stopped")

    # ── Public interface ──────────────────────────────────────────────

    def get_latest_thought(self):
        """Return the latest heartbeat thought for display, or None."""
        with self._lock:
            return self._latest_thought

    def get_latest_thought_age(self):
        """Seconds since the latest thought was generated."""
        with self._lock:
            if self._latest_thought_time == 0:
                return float("inf")
            return time.monotonic() - self._latest_thought_time

    def clear_thought(self):
        """Clear the current thought (after display dismisses it)."""
        with self._lock:
            self._latest_thought = None

    @property
    def is_sleeping(self):
        return self._is_sleeping

    @property
    def is_dreaming(self):
        return self._is_dreaming

    @property
    def sleep_reason(self):
        return self._sleep_reason

    def enter_sleep(self, reason="user"):
        """Put the pet to sleep (user-initiated or auto)."""
        self._is_sleeping = True
        self._sleep_reason = reason
        self._auto_sleep_warning_start = None
        self._pet._vitals.set_force_sleeping(True)
        log.info("Pet entered sleep (reason: %s)", reason)
        # Check if we should dream
        if DREAM_ENABLED and reason in ("user", "auto", "tuck_in"):
            self._check_dream_conditions()

    def wake_up(self):
        """Wake the pet from sleep."""
        if not self._is_sleeping:
            return
        was_dreaming = self._is_dreaming
        self._is_sleeping = False
        self._is_dreaming = False
        self._sleep_reason = ""
        self._pet._vitals.set_force_sleeping(False)
        log.info("Pet woke up (was_dreaming=%s)", was_dreaming)

    def register_hub_ip(self, ip):
        """Called when we learn a Hub's IP (from HTTP request source)."""
        if ip and ip != self._known_hub_ip:
            self._known_hub_ip = ip
            self._db.set_pet_state("hub_ip", ip)
            log.info("Registered Hub IP: %s", ip)

    def get_heartbeat_stats(self):
        """Return heartbeat status for protocol queries."""
        return {
            "enabled": HEARTBEAT_ENABLED,
            "running": self._running,
            "interval_s": self._current_interval(),
            "base_interval_s": HEARTBEAT_INTERVAL_S,
            "is_sleeping": self._is_sleeping,
            "sleep_reason": self._sleep_reason,
            "is_dreaming": self._is_dreaming,
            "known_hub_ip": self._known_hub_ip,
            "last_dream_time": (self._last_dream_time.isoformat()
                                if self._last_dream_time else None),
            "total_heartbeats": self._db.get_heartbeat_count_since(
                "2000-01-01"),
        }

    # ── Main heartbeat loop ───────────────────────────────────────────

    def _heartbeat_loop(self):
        """Background thread: periodic reflections."""
        log.info("Heartbeat loop running")
        # Initial delay — let the system boot up
        time.sleep(30)

        while self._running:
            try:
                interval = self._current_interval()

                # Skip heartbeat if sleeping, coma, or disabled
                if self._is_sleeping or self._pet.is_coma:
                    time.sleep(5)
                    continue

                # Check auto-sleep conditions
                self._check_auto_sleep()

                # Do the heartbeat
                self._do_heartbeat()

                # Sleep until next heartbeat (check every 5s for early exit)
                waited = 0
                while waited < interval and self._running:
                    time.sleep(5)
                    waited += 5
                    # Re-check if we entered sleep during wait
                    if self._is_sleeping or self._pet.is_coma:
                        break

            except Exception as e:
                log.error("Heartbeat error: %s", e)
                time.sleep(60)  # back off on error

    def _current_interval(self):
        """Calculate current heartbeat interval based on energy level."""
        energy = self._pet.energy
        base = HEARTBEAT_INTERVAL_S

        if energy < HEARTBEAT_ENERGY_THRESHOLD_CRITICAL:
            return base * HEARTBEAT_INTERVAL_CRITICAL_MULT
        elif energy < HEARTBEAT_ENERGY_THRESHOLD_LOW:
            return base * HEARTBEAT_INTERVAL_LOW_ENERGY_MULT
        return base

    def _do_heartbeat(self):
        """Execute one heartbeat with multi-turn conversation.

        The pet can chain up to MAX_HEARTBEAT_ROUNDS of inference + shell
        command execution.  Each [RUN: command] result is fed back as a
        user message so the pet can react and optionally run more commands.
        The loop ends when the pet produces a response with no [RUN:] tags
        or the round limit is hit.
        """
        # Pick a reflection type
        prompt_type = self._pick_prompt_type()
        prompt_template = random.choice(
            _REFLECTION_PROMPTS.get(prompt_type,
                                    _REFLECTION_PROMPTS["vitals_check"])
        )

        # Build context for template
        vitals = self._pet.get_vitals()
        battery_info = (self._battery.get_status()
                        if self._battery and self._battery.available
                        else {"percentage": -1, "charging": False})

        context = {
            "pet_name": PET_NAME,
            "hunger": vitals.get("hunger", 0),
            "cleanliness": vitals.get("cleanliness", 0),
            "energy": vitals.get("energy", 0),
            "happiness": vitals.get("happiness", 0),
            "intelligence": self._pet.intelligence,
            "battery_pct": battery_info.get("percentage", -1),
            "charging_status": ("charging" if battery_info.get("charging")
                                else "on battery"),
            "time_str": datetime.now().strftime("%H:%M"),
            "interaction_count": self._pet.interaction_count,
            "stage_name": self._pet.stage_name,
            "mood": self._pet.mood,
        }

        try:
            prompt = prompt_template.format(**context)
        except KeyError:
            prompt = prompt_template  # fallback if template has extra vars

        # Build system prompt
        system = _HEARTBEAT_SYSTEM.format(name=PET_NAME)

        # Always include available commands so the pet can explore
        system += "\n\n" + self._shell.get_command_list_prompt()

        # Build conversation history for multi-turn
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Track totals across all rounds
        total_inference_ms = 0
        total_tokens = 0
        all_shell_commands = []
        all_shell_results = []
        all_response_parts = []  # clean text from each round

        for round_num in range(HEARTBEAT_MAX_ROUNDS):
            # Run inference with full conversation history
            response_text, inference_ms, tokens = self._run_inference(
                system, prompt, messages=messages
            )

            if response_text is None:
                if round_num == 0:
                    log.warning("Heartbeat inference failed — skipping")
                    return
                break  # stop multi-turn if later rounds fail

            total_inference_ms += inference_ms
            total_tokens += tokens

            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": response_text})

            # Parse any [RUN: command] requests
            shell_commands = parse_run_commands(response_text)

            # Collect the clean text (without [RUN:] tags)
            clean_text = strip_run_commands(response_text).strip()
            if clean_text:
                all_response_parts.append(clean_text)

            if not shell_commands:
                # No commands → done, pet finished thinking
                log.debug("Heartbeat round %d: no commands, done", round_num)
                break

            # Execute shell commands
            all_shell_commands.extend(shell_commands)
            results = self._shell.execute_multiple(shell_commands)
            shell_parts = []
            for cmd, ok, output in results:
                status = "OK" if ok else "FAILED"
                shell_parts.append(f"[{cmd}] ({status}): {output}")
            results_text = "\n".join(shell_parts)
            all_shell_results.append(results_text)

            # Feed results back as a user message for the next round
            followup = (
                f"Here are the results of your commands:\n"
                f"{results_text}\n\n"
                f"React to what you found. You can run more commands "
                f"with [RUN: command] or just share your thoughts."
            )
            messages.append({"role": "user", "content": followup})

            log.debug("Heartbeat round %d: ran %d commands, continuing",
                      round_num, len(shell_commands))

            # Check if we should stop (energy too low, sleeping, etc.)
            if self._is_sleeping or self._pet.is_coma:
                log.info("Heartbeat multi-turn interrupted (sleep/coma)")
                break

        # Combine all clean response parts for display
        display_text = " ".join(all_response_parts).strip()
        if not display_text:
            # Fallback: use last response as-is
            display_text = strip_run_commands(
                messages[-1].get("content", "") if messages else ""
            ).strip() or "(no thought)"

        # Calculate sentiment
        from pet import simple_sentiment
        sentiment = simple_sentiment(display_text)

        # Combine all shell results for logging
        shell_results_combined = "\n---\n".join(all_shell_results)

        # Log to database
        self._db.insert_heartbeat(
            prompt_type=prompt_type,
            prompt=prompt,
            response=display_text,
            sentiment_score=sentiment,
            inference_time_ms=total_inference_ms,
            tokens_generated=total_tokens,
            battery_pct=battery_info.get("percentage", -1),
            is_charging=battery_info.get("charging", False),
            hunger=vitals.get("hunger", 0),
            cleanliness=vitals.get("cleanliness", 0),
            energy=vitals.get("energy", 0),
            happiness=vitals.get("happiness", 0),
            shell_commands=(json.dumps(all_shell_commands)
                            if all_shell_commands else ""),
            shell_results=shell_results_combined,
        )

        # Update latest thought for display
        with self._lock:
            self._latest_thought = display_text
            self._latest_thought_time = time.monotonic()

        rounds_used = min(len([m for m in messages if m["role"] == "assistant"]),
                          HEARTBEAT_MAX_ROUNDS)
        log.info("Heartbeat [%s] (%d rounds): %s (sentiment=%.2f, %dms, %d tokens)",
                 prompt_type, rounds_used, display_text[:80], sentiment,
                 total_inference_ms, total_tokens)

    def _pick_prompt_type(self):
        """Choose what kind of reflection to do this heartbeat."""
        weights = {
            "vitals_check": 30,
            "body_awareness": 25,
            "time_awareness": 20,
            "memory_reflection": 25,
        }
        types = list(weights.keys())
        probs = [weights[t] for t in types]
        return random.choices(types, weights=probs, k=1)[0]

    def _run_inference(self, system_prompt, user_prompt, messages=None):
        """Run a single inference via llama-server.

        If `messages` is provided, it is used as-is (for multi-turn).
        Otherwise, a simple [system, user] pair is built from the args.

        Returns (response_text, inference_ms, tokens) or (None, 0, 0).
        """
        url = PET_LLAMA_SERVER_URL.rstrip("/") + "/v1/chat/completions"
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        payload = json.dumps({
            "messages": messages,
            "max_tokens": HEARTBEAT_MAX_TOKENS,
            "temperature": HEARTBEAT_TEMPERATURE,
            "top_p": PET_TOP_P,
            "repeat_penalty": PET_REPEAT_PENALTY,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())

            elapsed_ms = int((time.monotonic() - start) * 1000)

            if result and "choices" in result and result["choices"]:
                choice = result["choices"][0]
                text = choice.get("message", {}).get("content", "").strip()
                # Strip Qwen thinking tags
                for tag in ("</think>", "<think>"):
                    text = text.replace(tag, "").strip()
                usage = result.get("usage", {})
                tokens = usage.get("completion_tokens", 0)

                # Drain energy for this inference
                from config import PET_ENERGY_COST_PER_INFERENCE_MS, PET_ENERGY_COST_CAP
                cost = min(PET_ENERGY_COST_CAP,
                           elapsed_ms * PET_ENERGY_COST_PER_INFERENCE_MS)
                self._pet._vitals.drain_energy(cost)

                return text, elapsed_ms, tokens

            return None, 0, 0

        except Exception as e:
            log.warning("Heartbeat inference failed: %s", e)
            return None, 0, 0

    # ── Auto-sleep logic ──────────────────────────────────────────────

    def _check_auto_sleep(self):
        """Check if the pet should auto-sleep due to low battery/energy."""
        if self._is_sleeping:
            return

        battery_info = (self._battery.get_status()
                        if self._battery and self._battery.available
                        else None)

        # Battery critical → graceful shutdown
        if battery_info and battery_info["percentage"] < BATTERY_CRITICAL_PCT:
            log.critical("Battery critical (%d%%) — requesting shutdown",
                         battery_info["percentage"])
            # Set a flag that the main loop checks
            self._db.set_pet_state("shutdown_requested", "battery_critical")
            return

        # Battery low → force sleep
        if battery_info and battery_info["percentage"] < BATTERY_FORCE_SLEEP_PCT:
            log.warning("Battery low (%d%%) — forcing sleep",
                        battery_info["percentage"])
            self.enter_sleep("battery_low")
            return

        # Energy low → auto-sleep after delay
        from config import PET_SLEEP_AUTO_ENERGY_THRESHOLD, PET_SLEEP_AUTO_DELAY_S
        energy = self._pet.energy
        if energy < PET_SLEEP_AUTO_ENERGY_THRESHOLD:
            if self._auto_sleep_warning_start is None:
                self._auto_sleep_warning_start = time.monotonic()
                log.info("Energy low (%.2f) — auto-sleep warning started",
                         energy)
            elif (time.monotonic() - self._auto_sleep_warning_start
                  > PET_SLEEP_AUTO_DELAY_S):
                log.info("Energy low for %ds — entering auto-sleep",
                         PET_SLEEP_AUTO_DELAY_S)
                self.enter_sleep("low_energy")
        else:
            self._auto_sleep_warning_start = None

    # ── Dream logic ───────────────────────────────────────────────────

    def _check_dream_conditions(self):
        """Check if the pet should start dreaming (training)."""
        if not DREAM_ENABLED or self._is_dreaming:
            return

        battery_info = (self._battery.get_status()
                        if self._battery and self._battery.available
                        else None)

        # Need charging + high battery
        if not battery_info:
            return
        if not battery_info.get("external_power"):
            log.debug("Dream check: not on external power")
            return
        if battery_info["percentage"] < BATTERY_DREAM_MIN_PCT:
            log.debug("Dream check: battery too low (%d%%)",
                      battery_info["percentage"])
            return

        # Cooldown check
        if self._last_dream_time:
            elapsed = datetime.now(timezone.utc) - self._last_dream_time
            if elapsed < timedelta(hours=DREAM_COOLDOWN_HOURS):
                log.debug("Dream check: cooldown (%s since last dream)",
                          elapsed)
                return

        # Enough new interactions?
        last_dream_iso = (self._last_dream_time.isoformat()
                          if self._last_dream_time
                          else "2000-01-01T00:00:00")
        new_interactions = self._db.get_interactions_since(last_dream_iso)
        if new_interactions < DREAM_MIN_INTERACTIONS:
            log.debug("Dream check: not enough interactions (%d < %d)",
                      new_interactions, DREAM_MIN_INTERACTIONS)
            return

        # Hub reachable?
        hub_ip = self._known_hub_ip
        if not hub_ip:
            log.debug("Dream check: no known Hub IP")
            return

        hub_available = self._check_hub(hub_ip)
        if not hub_available:
            log.info("Dream check: Hub at %s not reachable", hub_ip)
            return

        # All conditions met — start dreaming!
        self._start_dream(hub_ip)

    def _check_hub(self, hub_ip):
        """Check if Cortex Hub is reachable at the given IP."""
        url = f"http://{hub_ip}:8000/api/hub/status"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=DREAM_HUB_TIMEOUT_S) as resp:
                data = json.loads(resp.read())
                return data.get("available", False)
        except Exception as e:
            log.debug("Hub check failed at %s: %s", hub_ip, e)
            return False

    def _start_dream(self, hub_ip):
        """Trigger dream training cycle via Hub."""
        log.info("Starting dream cycle via Hub at %s", hub_ip)
        self._is_dreaming = True

        url = f"http://{hub_ip}:8000/api/training/dream-cycle"
        payload = json.dumps({
            "pi_ip": "10.0.0.25",
            "pi_port": 8420,
            "trigger": "sleep_dream",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                log.info("Dream cycle started: %s", result)

            # Record the dream time
            self._last_dream_time = datetime.now(timezone.utc)
            self._db.set_pet_state("last_dream_time",
                                   self._last_dream_time.isoformat())

        except Exception as e:
            log.error("Failed to start dream cycle: %s", e)
            self._is_dreaming = False

    def on_dream_complete(self, training_metrics=None):
        """Called when the Hub reports dream training is complete."""
        self._is_dreaming = False
        log.info("Dream cycle complete")

        if training_metrics:
            self._pet.on_training_complete(training_metrics)

            # Generate a dream reflection
            old_iq = training_metrics.get("old_intelligence", 0)
            new_iq = self._pet.intelligence
            prompt = random.choice(
                _REFLECTION_PROMPTS["dream_reflection"]
            ).format(old_iq=old_iq, new_iq=new_iq)

            system = _HEARTBEAT_SYSTEM.format(name=PET_NAME)
            resp, ms, tokens = self._run_inference(system, prompt)
            if resp:
                with self._lock:
                    self._latest_thought = resp
                    self._latest_thought_time = time.monotonic()
                log.info("Dream reflection: %s", resp[:80])

    # ── Battery-energy blending ───────────────────────────────────────

    def get_blended_energy(self):
        """Return energy blended with real battery percentage.

        Returns a 0.0–1.0 float combining real battery (60% weight)
        and inference energy budget (40% weight).
        """
        inference_energy = self._pet.energy

        if self._battery and self._battery.available:
            battery_pct = self._battery.percentage / 100.0
            blended = (BATTERY_ENERGY_WEIGHT * battery_pct +
                       INFERENCE_ENERGY_WEIGHT * inference_energy)
        else:
            # No battery monitor — use pure inference energy
            blended = inference_energy

        return max(0.0, min(1.0, blended))
