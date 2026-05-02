"""Voxel-based pet visualization — renders LoRA weight clusters as a rotating 3D point cloud.

Drop-in replacement for SpriteAnimator. Reads a pre-baked voxels.msgpack
file (350 voxels, ~10KB) and renders an 80x80 RGBA PIL Image each frame.

The visualization is unique to the pet's trained LoRA adapter — you can
literally see where the model learned. LoRA-affected clusters glow
orange→purple while base model clusters stay dim gray.

Performance target: <5ms per frame on Pi Zero 2W (Cortex-A53).
"""

import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from pet_config import PET_SPRITE_SIZE

VOXEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "voxels.msgpack"
)

# ── Mood-reactive parameters ──
# Each mood defines: rotation_speed (rad/tick), brightness_mult, color_tint (r,g,b), pulse_speed
MOOD_PARAMS = {
    "idle_happy":   {"rot": 0.035, "bright": 1.1, "tint": (0, 40, 0),   "pulse": 0.02},
    "idle_content": {"rot": 0.028, "bright": 1.0, "tint": (0, 20, 0),   "pulse": 0.015},
    "idle_neutral": {"rot": 0.020, "bright": 0.9, "tint": (0, 0, 0),    "pulse": 0.01},
    "idle_uneasy":  {"rot": 0.040, "bright": 0.85, "tint": (0, 20, 30), "pulse": 0.03},
    "idle_sad":     {"rot": 0.010, "bright": 0.6, "tint": (0, 0, 30),   "pulse": 0.008},
    "thinking":     {"rot": 0.100, "bright": 1.3, "tint": (20, 10, 0),  "pulse": 0.06},
    "talking":      {"rot": 0.065, "bright": 1.2, "tint": (10, 10, 10), "pulse": 0.05},
    "sleeping":     {"rot": 0.004, "bright": 0.4, "tint": (0, 0, 40),   "pulse": 0.005},
    "eating":       {"rot": 0.130, "bright": 1.4, "tint": (0, 30, 0),   "pulse": 0.08},
    "evolve":       {"rot": 0.200, "bright": 1.5, "tint": (20, 20, 20), "pulse": 0.10},
}
DEFAULT_MOOD = MOOD_PARAMS["idle_neutral"]


class VoxelAnimator:
    """3D voxel point cloud renderer — drop-in SpriteAnimator replacement.

    Usage:
        animator = VoxelAnimator()
        animator.play("idle_happy", fps=0.5, loop=True)

        # In main loop:
        animator.tick()
        frame = animator.get_frame()  # PIL Image (RGBA, 80x80)
    """

    def __init__(self, voxel_path=None):
        self._voxel_path = voxel_path or VOXEL_PATH
        self._size = PET_SPRITE_SIZE  # 80x80

        # Animation state (matches SpriteAnimator interface)
        self._current_name = ""
        self._fps = 1.0
        self._loop = True
        self._playing = False
        self._last_frame_time = 0.0
        self._on_complete = None

        # Rotation state
        self._angle = 0.0  # Y-axis rotation in radians
        self._tick_count = 0
        self._pulse_phase = 0.0

        # Mood params
        self._mood = DEFAULT_MOOD

        # Load voxel data
        self._loaded = False
        self._n = 0
        self._x = None
        self._y = None
        self._z = None
        self._r = None
        self._g = None
        self._b = None
        self._a = None
        self._sizes = None
        self._lora = None

        # Pre-computed center offset
        self._cx = 48.0
        self._cy = 48.0
        self._cz = 48.0

        # Cached blank frame
        self._blank = Image.new("RGBA", (self._size, self._size), (0, 0, 0, 0))

        self._load_voxels()

    def _load_voxels(self):
        """Load pre-baked voxels from msgpack file."""
        if not os.path.exists(self._voxel_path):
            print(f"Voxels not found: {self._voxel_path}")
            return

        try:
            import msgpack

            with open(self._voxel_path, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)

            meta = data["meta"]
            self._n = meta["n_voxels"]
            v = data["voxels"]

            # Zero-copy numpy views on flat byte arrays
            self._x = np.frombuffer(v["x"], dtype=np.float32).copy()
            self._y = np.frombuffer(v["y"], dtype=np.float32).copy()
            self._z = np.frombuffer(v["z"], dtype=np.float32).copy()
            self._r = np.frombuffer(v["r"], dtype=np.uint8).copy()
            self._g = np.frombuffer(v["g"], dtype=np.uint8).copy()
            self._b = np.frombuffer(v["b"], dtype=np.uint8).copy()
            self._a = np.frombuffer(v["a"], dtype=np.uint8).copy()
            self._sizes = np.frombuffer(v["size"], dtype=np.float32).copy()
            self._lora = np.frombuffer(v["lora"], dtype=np.uint8).copy()

            # Compute center of point cloud for rotation origin
            self._cx = float(np.mean(self._x))
            self._cy = float(np.mean(self._y))
            self._cz = float(np.mean(self._z))

            self._loaded = True
            lora_count = int(np.sum(self._lora > 0))
            print(f"Voxels loaded: {self._n} points, {lora_count} LoRA-affected")

        except Exception as e:
            print(f"Failed to load voxels: {e}")

    def has_animation(self, name):
        """Check if an animation exists — always True if voxels are loaded."""
        return self._loaded

    def play(self, name, fps=1.0, loop=True, on_complete=None):
        """Start playing an animation.

        Args:
            name: Animation/mood name (e.g. "idle_happy", "thinking").
            fps: Frames per second (controls rotation smoothness).
            loop: Whether to loop.
            on_complete: Callback when non-looping animation finishes.
        """
        if name == self._current_name and self._playing:
            return

        self._current_name = name
        self._fps = max(0.1, fps)
        self._loop = loop
        self._playing = True
        self._last_frame_time = time.monotonic()
        self._on_complete = on_complete

        # Set mood parameters
        self._mood = MOOD_PARAMS.get(name, DEFAULT_MOOD)

    def stop(self):
        """Stop animation (freeze rotation)."""
        self._playing = False

    def tick(self):
        """Advance rotation. Call each main loop iteration."""
        if not self._playing:
            return

        now = time.monotonic()
        frame_duration = 1.0 / self._fps

        if now - self._last_frame_time >= frame_duration:
            self._last_frame_time = now
            self._tick_count += 1
            self._angle += self._mood["rot"]
            self._pulse_phase += self._mood["pulse"]

            # For non-looping, complete after one full rotation
            if not self._loop and self._angle >= 2 * math.pi:
                self._playing = False
                if self._on_complete:
                    cb = self._on_complete
                    self._on_complete = None
                    cb()

    def get_frame(self):
        """Render the current frame as an 80x80 RGBA PIL Image.

        Pipeline (all numpy, no per-point Python loops):
          1. Rotate 350 points around Y-axis
          2. Orthographic project to 2D
          3. Depth-sort back-to-front
          4. Draw circles with pre-baked colors
        """
        if not self._loaded:
            return self._blank

        n = self._n
        size = self._size

        # ── 1. Rotate around Y-axis ──
        cos_a = math.cos(self._angle)
        sin_a = math.sin(self._angle)

        # Center, rotate, un-center
        dx = self._x - self._cx
        dz = self._z - self._cz

        rx = dx * cos_a - dz * sin_a + self._cx
        ry = self._y  # Y unchanged
        rz = dx * sin_a + dz * cos_a + self._cz

        # ── 2. Orthographic projection to 80x80 ──
        # Map from ~[0, 96] coordinate space to [0, 80] screen space
        # with some padding
        scale = size / 110.0  # slightly smaller to add margin
        offset = size / 2.0

        sx = (rx - self._cx) * scale + offset
        sy = (ry - self._cy) * scale + offset

        # ── 3. Depth sort (back-to-front) ──
        order = np.argsort(rz)  # ascending = back first

        # ── 4. Draw points ──
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Pulse brightness
        pulse = 0.5 + 0.5 * math.sin(self._pulse_phase)
        bright = self._mood["bright"]
        tint_r, tint_g, tint_b = self._mood["tint"]

        # Batch extract screen coordinates
        sx_arr = sx[order]
        sy_arr = sy[order]
        rz_arr = rz[order]

        # Depth dimming: front = bright, back = dim
        z_min = float(rz_arr[0]) if n > 0 else 0
        z_max = float(rz_arr[-1]) if n > 0 else 1
        z_range = max(z_max - z_min, 1.0)

        # Pre-fetch color arrays in sorted order
        r_arr = self._r[order]
        g_arr = self._g[order]
        b_arr = self._b[order]
        a_arr = self._a[order]
        s_arr = self._sizes[order]
        l_arr = self._lora[order]

        for i in range(n):
            px = float(sx_arr[i])
            py = float(sy_arr[i])

            # Skip off-screen points
            if px < -4 or px > size + 4 or py < -4 or py > size + 4:
                continue

            # Depth factor: 0.4 (back) to 1.0 (front)
            depth_t = (float(rz_arr[i]) - z_min) / z_range
            depth_bright = 0.4 + 0.6 * depth_t

            # LoRA pulse: LoRA-affected points pulse brighter
            lora_val = int(l_arr[i])
            lora_pulse = 1.0
            if lora_val > 0:
                lora_pulse = 1.0 + 0.3 * pulse * (lora_val / 255.0)

            # Final color
            mult = bright * depth_bright * lora_pulse
            cr = min(255, int(int(r_arr[i]) * mult + tint_r * pulse))
            cg = min(255, int(int(g_arr[i]) * mult + tint_g * pulse))
            cb = min(255, int(int(b_arr[i]) * mult + tint_b * pulse))
            ca = int(a_arr[i])

            # Point radius (scaled for 80x80)
            radius = max(1.0, float(s_arr[i]) * scale * 0.5)

            if radius <= 1.2:
                # Single pixel for tiny points
                ix, iy = int(px), int(py)
                if 0 <= ix < size and 0 <= iy < size:
                    draw.point((ix, iy), fill=(cr, cg, cb, ca))
            else:
                # Draw filled circle
                r_int = radius
                x0 = px - r_int
                y0 = py - r_int
                x1 = px + r_int
                y1 = py + r_int
                draw.ellipse([x0, y0, x1, y1], fill=(cr, cg, cb, ca))

        return img

    def set_mood_idle(self, mood):
        """Switch to idle animation for the given mood.

        Args:
            mood: "happy", "content", "neutral", "uneasy", "sad"
        """
        anim_name = f"idle_{mood}"
        if anim_name not in MOOD_PARAMS:
            anim_name = "idle_neutral"
        self.play(anim_name, fps=0.5, loop=True)

    @property
    def current_animation(self):
        """Name of the currently active animation."""
        return self._current_name

    @property
    def is_playing(self):
        """Whether animation is currently advancing."""
        return self._playing
