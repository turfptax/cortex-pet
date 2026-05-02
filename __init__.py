"""Cortex pet plugin — slice 2a stub.

This stub exists so the plugin runtime discovers the pet at boot and lists
it as `pet@0.1.0`. The runtime in v0 doesn't yet call register() — that
wiring lands in slice 2b along with the real pet engine, heartbeat, body
shell, tamagotchi display, and pet.db.

See ../../notes/plugin-interface-sketch.md for the v0 design and
~/.claude/projects/.../memory/pet_extraction_slice_plan.md for the slice
breakdown.
"""

from plugin_api import Plugin


class PetPlugin(Plugin):
    """Stub. Slice 2b fills in screens, MCP tools, HTTP routes, tasks."""

    def on_load(self) -> None:
        self.api.log.info("pet plugin stub loaded (slice 2a — no behavior yet)")

    def on_unload(self) -> None:
        self.api.log.info("pet plugin stub unloaded")


def register(api):
    """Entry point invoked by plugins_runtime once loading lands in slice 2b."""
    return PetPlugin(api)
