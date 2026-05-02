"""Sprite animation engine for Tamagotchi pet display.

Loads PNG sprite frames from a directory, groups them by animation name,
and provides a tick-based frame advancement system.

Naming convention: {anim_name}_{frame}.png
  e.g. idle_happy_0.png, idle_happy_1.png, thinking_0.png
"""

import os
import time

from PIL import Image

from config import SPRITE_DIR, PET_SPRITE_SIZE


class SpriteAnimator:
    """Frame-based sprite animation system.

    Usage:
        animator = SpriteAnimator()
        animator.play("idle_happy", fps=0.5, loop=True)

        # In main loop:
        animator.tick()
        frame = animator.get_frame()  # PIL Image (RGBA)
    """

    def __init__(self, sprite_dir=None):
        self._sprite_dir = sprite_dir or SPRITE_DIR
        self._animations = {}  # name -> list of PIL Images
        self._load_sprites()

        # Current animation state
        self._current_name = ""
        self._current_frames = []
        self._frame_index = 0
        self._fps = 1.0
        self._loop = True
        self._playing = False
        self._last_frame_time = 0
        self._on_complete = None

        # Fallback: single-pixel transparent image
        self._blank = Image.new("RGBA", (PET_SPRITE_SIZE, PET_SPRITE_SIZE), (0, 0, 0, 0))

    def _load_sprites(self):
        """Load all PNG sprites from the sprite directory."""
        if not os.path.isdir(self._sprite_dir):
            print(f"Sprite dir not found: {self._sprite_dir}")
            return

        # Gather all PNGs
        files = sorted(f for f in os.listdir(self._sprite_dir) if f.endswith(".png"))

        for fname in files:
            # Parse: name_frame.png -> animation name, frame index
            base = fname[:-4]  # strip .png
            parts = base.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                anim_name = parts[0]
                frame_idx = int(parts[1])
            else:
                # Single-frame animation or invalid naming
                anim_name = base
                frame_idx = 0

            try:
                img = Image.open(os.path.join(self._sprite_dir, fname)).convert("RGBA")
                # Resize to standard size if needed
                if img.size != (PET_SPRITE_SIZE, PET_SPRITE_SIZE):
                    img = img.resize((PET_SPRITE_SIZE, PET_SPRITE_SIZE), Image.NEAREST)
            except Exception as e:
                print(f"Failed to load sprite {fname}: {e}")
                continue

            if anim_name not in self._animations:
                self._animations[anim_name] = []

            # Ensure list is long enough
            while len(self._animations[anim_name]) <= frame_idx:
                self._animations[anim_name].append(None)
            self._animations[anim_name][frame_idx] = img

        # Remove None gaps (in case of missing frame numbers)
        for name in self._animations:
            self._animations[name] = [f for f in self._animations[name] if f is not None]

        loaded = sum(len(frames) for frames in self._animations.values())
        print(f"Sprites loaded: {len(self._animations)} animations, {loaded} frames")

    def has_animation(self, name):
        """Check if an animation exists."""
        return name in self._animations and len(self._animations[name]) > 0

    def play(self, name, fps=1.0, loop=True, on_complete=None):
        """Start playing an animation.

        Args:
            name: Animation name (e.g. "idle_happy", "thinking").
            fps: Frames per second.
            loop: Whether to loop the animation.
            on_complete: Callback when non-looping animation finishes.
        """
        if name == self._current_name and self._playing:
            return  # Already playing this animation

        if name in self._animations:
            self._current_name = name
            self._current_frames = self._animations[name]
            self._frame_index = 0
            self._fps = max(0.1, fps)
            self._loop = loop
            self._playing = True
            self._last_frame_time = time.monotonic()
            self._on_complete = on_complete
        else:
            # Unknown animation — show blank
            self._current_name = name
            self._current_frames = [self._blank]
            self._frame_index = 0
            self._playing = False

    def stop(self):
        """Stop the current animation (freeze on current frame)."""
        self._playing = False

    def tick(self):
        """Advance frame counter based on elapsed time. Call each main loop."""
        if not self._playing or not self._current_frames:
            return

        now = time.monotonic()
        frame_duration = 1.0 / self._fps

        if now - self._last_frame_time >= frame_duration:
            self._last_frame_time = now
            self._frame_index += 1

            if self._frame_index >= len(self._current_frames):
                if self._loop:
                    self._frame_index = 0
                else:
                    self._frame_index = len(self._current_frames) - 1
                    self._playing = False
                    if self._on_complete:
                        cb = self._on_complete
                        self._on_complete = None
                        cb()

    def get_frame(self):
        """Get the current frame as a PIL RGBA Image.

        Returns:
            PIL Image (RGBA, PET_SPRITE_SIZE x PET_SPRITE_SIZE).
        """
        if not self._current_frames:
            return self._blank
        idx = min(self._frame_index, len(self._current_frames) - 1)
        return self._current_frames[idx]

    def set_mood_idle(self, mood):
        """Convenience: switch to the idle animation for the given mood.

        Args:
            mood: "happy", "content", "neutral", "uneasy", "sad"
        """
        anim_name = f"idle_{mood}"
        if not self.has_animation(anim_name):
            anim_name = "idle_neutral"
        if not self.has_animation(anim_name):
            # Last resort: try any idle animation
            for name in self._animations:
                if name.startswith("idle_"):
                    anim_name = name
                    break
        self.play(anim_name, fps=0.5, loop=True)

    @property
    def current_animation(self):
        """Name of the currently active animation."""
        return self._current_name

    @property
    def is_playing(self):
        """Whether an animation is currently advancing."""
        return self._playing
