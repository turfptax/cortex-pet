"""Cortex pet plugin — slice 2b1.

Instantiates PetEngine and Heartbeat against the plugin's own pet.db. The
old in-core path in cortex-core/src/main.py keeps creating its own
PetEngine + Heartbeat against cortex.db so we have a verifiable
parallel state for one slice.

Important constraint for slice 2b1
----------------------------------
The plugin's Heartbeat is INSTANTIATED but NOT STARTED. The in-core
heartbeat thread is still running; starting another would mean:
  - double autonomous reflections every ~30 min
  - double inference load on the Pi's llama-server
  - duplicated heartbeat_log rows in different DBs (ugly but harmless)

Slice 2b2 removes the in-core heartbeat and calls self.heartbeat.start()
here so the plugin owns the autonomous loop.

Likewise the plugin's PetEngine spawns its own pet-inference daemon
thread, but the queue stays empty unless something calls ask() on it,
so the thread just blocks waiting. No work, no llama-server contention.
"""

import logging

from plugin_api import Plugin

# Resolved via sys.path entry added by plugins_runtime when this module
# is imported. plugins/pet/ contains pet.py, heartbeat.py, body_shell.py.
from pet import PetEngine
from heartbeat import Heartbeat


log = logging.getLogger("plugin.pet")


class PetPlugin(Plugin):
    """Pet plugin owning its own PetEngine + Heartbeat instances."""

    def __init__(self, api):
        super().__init__(api)
        self.pet_engine = None
        self.heartbeat = None

    def on_load(self) -> None:
        # Plugin's PetEngine talks to plugins/pet/data/pet.db (fresh,
        # empty, no migrated history yet — that's slice 2b2).
        self.pet_engine = PetEngine(self.api.db, battery=None)
        self.api.log.info(
            "plugin PetEngine created (parallel to core path; using pet.db)"
        )

        self.heartbeat = Heartbeat(self.api.db, self.pet_engine, battery=None)
        # Deliberately NOT started in 2b1 — see module docstring.
        self.api.log.info(
            "plugin Heartbeat created (NOT started in 2b1 — slice 2b2 will start it)"
        )

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
