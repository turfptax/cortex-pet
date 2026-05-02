"""Cyberpunk Tamagotchi display renderer for the 240x280 ST7789P3.

Neon-accented UI with circuit trace backgrounds, pixel art sprites,
and glowing progress bars on a deep blue-black canvas.

Layout (240x280):
    [Status Bar  0-19]   Clock, BLE dot, mood bar, battery
    [Pet Zone   20-199]  Sprite + circuit traces + speech bubble / menu
    [Info Bar  200-279]  XP bar, vitals, button hints
"""

import time

from PIL import Image, ImageDraw, ImageFont

from config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT, SEGMENT_SECONDS,
    FONT_PATH, FONT_PATH_REGULAR, FONT_LARGE, FONT_MEDIUM, FONT_SMALL,
    COLOR_BG, COLOR_TEXT, COLOR_DIM, COLOR_RED, COLOR_GREEN,
    COLOR_YELLOW, COLOR_BLUE, COLOR_BAR_BG, COLOR_CYAN, COLOR_CYAN_DIM,
    COLOR_SEPARATOR, COLOR_SPEECH_BORDER, COLOR_MAGENTA, COLOR_MAGENTA_DIM,
    COLOR_CIRCUIT_PRIMARY, COLOR_CIRCUIT_NODE,
)
from pet_config import (
    COLOR_PET_HAPPY, COLOR_PET_NEUTRAL, COLOR_PET_SAD,
    COLOR_CURSOR, COLOR_MENU_BG, COLOR_HIGHLIGHT, COLOR_SPEECH_BG,
    COLOR_XP_BAR, COLOR_XP_BAR_BG,
    PET_SPRITE_SIZE, PET_STAGE_THRESHOLDS, PET_STAGE_NAMES,
    # Vitals colors
    COLOR_VITAL_HUNGER, COLOR_VITAL_HUNGER_LOW,
    COLOR_VITAL_CLEAN, COLOR_VITAL_CLEAN_LOW,
    COLOR_VITAL_ENERGY, COLOR_VITAL_ENERGY_LOW,
    COLOR_VITAL_IQ,
    PET_VITAL_LOW, PET_VITAL_CRITICAL,
    PET_COMA_REVIVAL_THRESHOLD,
)

from sprite import SpriteAnimator
from voxel_animator import VoxelAnimator, VOXEL_PATH


def _word_wrap(text, font, max_width):
    """Wrap text to fit within max_width pixels. Returns list of lines."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _format_size(bytes_val):
    """Format bytes as human-readable size."""
    if bytes_val >= 1_073_741_824:
        return f"{bytes_val / 1_073_741_824:.1f}GB"
    if bytes_val >= 1_048_576:
        return f"{bytes_val / 1_048_576:.0f}MB"
    return f"{bytes_val / 1024:.0f}KB"


class TamagotchiDisplay:
    """Cyberpunk Tamagotchi-style animated pet display.

    Drop-in replacement for Display class — same render(state) interface.
    """

    W = DISPLAY_WIDTH   # 240
    H = DISPLAY_HEIGHT  # 280

    # Layout zones
    STATUS_Y = 0
    STATUS_H = 20
    PET_Y = 20
    PET_H = 180
    INFO_Y = 200
    INFO_H = 80

    def __init__(self, board):
        self.board = board
        self.img = Image.new("RGB", (self.W, self.H), COLOR_BG)
        self.draw = ImageDraw.Draw(self.img)

        try:
            self.font_lg = ImageFont.truetype(FONT_PATH, FONT_LARGE)
            self.font_md = ImageFont.truetype(FONT_PATH_REGULAR, FONT_MEDIUM)
            self.font_sm = ImageFont.truetype(FONT_PATH_REGULAR, FONT_SMALL)
        except OSError:
            self.font_lg = ImageFont.load_default()
            self.font_md = ImageFont.load_default()
            self.font_sm = ImageFont.load_default()

        # Pre-allocate output buffer
        self._buf = bytearray(self.W * self.H * 2)

        # Pet animator — use voxel renderer if voxels.msgpack exists, else sprites
        import os
        if os.path.exists(VOXEL_PATH):
            self.sprites = VoxelAnimator()
        else:
            self.sprites = SpriteAnimator()

        # Track last mood to know when to change idle animation
        self._last_mood = None
        self._last_state = None

        # Pre-render circuit trace background (180px pet zone)
        self._circuit_bg = self._make_circuit_bg()

    # ---- Circuit trace background (pre-rendered) ----

    def _make_circuit_bg(self):
        """Pre-render subtle circuit trace pattern for the pet zone."""
        img = Image.new("RGB", (self.W, self.PET_H), COLOR_BG)
        draw = ImageDraw.Draw(img)

        # Horizontal traces
        for y in [25, 65, 105, 145]:
            draw.line([(0, y), (self.W, y)], fill=COLOR_CIRCUIT_PRIMARY, width=1)

        # Vertical traces
        for x in [40, 120, 200]:
            draw.line([(x, 0), (x, self.PET_H)], fill=COLOR_CIRCUIT_PRIMARY, width=1)

        # Junction nodes (small dots at intersections)
        for y in [25, 65, 105, 145]:
            for x in [40, 120, 200]:
                draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill=COLOR_CIRCUIT_NODE)

        # Right-angle trace turns (L-shapes for visual interest)
        traces = [
            # (start_x, start_y, turn_x, turn_y, end_x, end_y)
            (10, 25, 10, 50, 40, 50),
            (200, 105, 220, 105, 220, 130),
            (40, 145, 40, 165, 75, 165),
            (170, 25, 170, 45, 200, 45),
        ]
        for sx, sy, tx, ty, ex, ey in traces:
            draw.line([(sx, sy), (tx, ty), (ex, ey)], fill=COLOR_CIRCUIT_PRIMARY, width=1)
            # Node at turn
            draw.rectangle([tx - 1, ty - 1, tx + 1, ty + 1], fill=COLOR_CIRCUIT_NODE)

        return img

    def _draw_circuit_background(self):
        """Paste pre-rendered circuit traces into the pet zone."""
        self.img.paste(self._circuit_bg, (0, self.PET_Y))

    # ---- Neon UI helpers ----

    def _draw_neon_bar(self, x, y, w, h, fill_pct, fill_color,
                       outline_color=None):
        """Draw a neon-outlined progress bar with glow highlight."""
        if outline_color is None:
            outline_color = COLOR_SEPARATOR
        # Outline
        self.draw.rectangle([x, y, x + w, y + h], outline=outline_color)
        # Background fill
        self.draw.rectangle([x + 1, y + 1, x + w - 1, y + h - 1], fill=COLOR_BAR_BG)
        # Value fill
        fill_w = int((w - 2) * max(0.0, min(1.0, fill_pct)))
        if fill_w > 0:
            self.draw.rectangle([x + 1, y + 1, x + 1 + fill_w, y + h - 1],
                                fill=fill_color)
            # Glow highlight (bright line at top of fill)
            bright = tuple(min(255, int(c * 1.3)) for c in fill_color)
            self.draw.line([(x + 1, y + 1), (x + 1 + fill_w, y + 1)],
                           fill=bright, width=1)

    def _draw_cyber_bubble(self, x, y, w, h, pointer_dir="up"):
        """Draw angular speech bubble with neon border + chamfered corners.

        pointer_dir: 'up', 'left', or None
        """
        # Fill
        self.draw.rectangle([x + 3, y, x + w - 3, y + h], fill=COLOR_SPEECH_BG)
        self.draw.rectangle([x, y + 3, x + w, y + h - 3], fill=COLOR_SPEECH_BG)
        # Corner fills
        for cx, cy in [(x, y), (x + w - 3, y), (x, y + h - 3), (x + w - 3, y + h - 3)]:
            self.draw.rectangle([cx, cy, cx + 3, cy + 3], fill=COLOR_SPEECH_BG)

        # Neon border (top, bottom, left, right) with chamfered corners
        # Top edge (skipping corners)
        self.draw.line([(x + 3, y), (x + w - 3, y)], fill=COLOR_SPEECH_BORDER, width=1)
        # Bottom edge
        self.draw.line([(x + 3, y + h), (x + w - 3, y + h)], fill=COLOR_SPEECH_BORDER, width=1)
        # Left edge
        self.draw.line([(x, y + 3), (x, y + h - 3)], fill=COLOR_SPEECH_BORDER, width=1)
        # Right edge
        self.draw.line([(x + w, y + 3), (x + w, y + h - 3)], fill=COLOR_SPEECH_BORDER, width=1)
        # Chamfered corners (diagonal cuts)
        self.draw.line([(x, y + 3), (x + 3, y)], fill=COLOR_SPEECH_BORDER, width=1)
        self.draw.line([(x + w - 3, y), (x + w, y + 3)], fill=COLOR_SPEECH_BORDER, width=1)
        self.draw.line([(x, y + h - 3), (x + 3, y + h)], fill=COLOR_SPEECH_BORDER, width=1)
        self.draw.line([(x + w - 3, y + h), (x + w, y + h - 3)], fill=COLOR_SPEECH_BORDER, width=1)

        # Pointer arrow
        if pointer_dir == "up":
            px = x + w // 2
            self.draw.polygon([(px - 5, y), (px, y - 6), (px + 5, y)],
                              fill=COLOR_SPEECH_BG, outline=COLOR_SPEECH_BORDER)
        elif pointer_dir == "left":
            py_mid = y + h // 3
            self.draw.polygon([(x, py_mid), (x - 6, py_mid + 5), (x, py_mid + 10)],
                              fill=COLOR_SPEECH_BG, outline=COLOR_SPEECH_BORDER)

    def render(self, state):
        """Render full frame from application state dict."""
        # Clear canvas
        self.draw.rectangle([0, 0, self.W, self.H], fill=COLOR_BG)

        app = state.get("app_state", "HOME")

        # Tick sprite animation every frame
        self.sprites.tick()

        # Update idle animation based on mood when on HOME screen
        if app == "HOME":
            pet_info = state.get("pet_info")
            mood = pet_info.get("mood", "neutral") if pet_info else "neutral"
            is_nighttime = pet_info.get("is_sleeping", False) if pet_info else False
            idle_since = state.get("idle_since", 0)
            now = time.monotonic()

            if is_nighttime:
                # Nighttime sleep — always show sleeping animation
                if self.sprites.current_animation != "sleeping":
                    self.sprites.play("sleeping", fps=0.2, loop=True)
            elif idle_since > 0 and (now - idle_since) > 20:
                if self.sprites.current_animation != "sleeping":
                    self.sprites.play("sleeping", fps=0.3, loop=True)
            elif mood != self._last_mood or self._last_state != "HOME":
                self.sprites.set_mood_idle(mood)
                self._last_mood = mood
        elif app == "PET_ASKING":
            if self.sprites.current_animation != "thinking":
                self.sprites.play("thinking", fps=1.5, loop=True)
        elif app == "PET_RESPONSE":
            if self.sprites.current_animation != "talking":
                self.sprites.play("talking", fps=1.0, loop=True)
        elif app == "STT_LISTENING" or app == "NOTE_TAKING":
            if self.sprites.current_animation != "talking":
                self.sprites.play("talking", fps=0.8, loop=True)
        elif app == "PET_COMA":
            if self.sprites.current_animation != "sleeping":
                self.sprites.play("sleeping", fps=0.2, loop=True)
        elif app == "PET_SLEEPING":
            if self.sprites.current_animation != "sleeping":
                self.sprites.play("sleeping", fps=0.3, loop=True)
        elif app == "PET_DREAMING":
            if self.sprites.current_animation != "thinking":
                self.sprites.play("thinking", fps=0.5, loop=True)
        elif app == "PET_FEEDING":
            if self.sprites.current_animation != "talking":
                self.sprites.play("talking", fps=2.0, loop=True)

        self._last_state = app

        # Dispatch to renderer
        if app == "HOME":
            self._render_home(state)
        elif app == "MENU":
            self._render_menu(state)
        elif app == "PET_ASKING":
            self._render_pet_asking(state)
        elif app == "PET_RESPONSE":
            self._render_pet_response(state)
        elif app == "STT_LISTENING":
            self._render_stt_listening(state)
        elif app == "NOTE_TAKING":
            self._render_note_taking(state)
        elif app == "PET_STATUS":
            self._render_pet_status(state)
        elif app == "CONFIRM_SHUTDOWN":
            self._render_confirm_shutdown(state)
        elif app in ("RECORDING", "PAUSED"):
            self._render_recording(state)
        elif app == "PET_FEEDING":
            self._render_pet_feeding(state)
        elif app == "PET_CLEANING":
            self._render_pet_cleaning(state)
        elif app == "PET_COMA":
            self._render_pet_coma(state)
        elif app == "PET_SLEEPING":
            self._render_pet_sleeping(state)
        elif app == "PET_DREAMING":
            self._render_pet_dreaming(state)
        elif app == "SETTING_ADJUST":
            self._render_setting_adjust(state)
        elif app == "INFO_SCREEN":
            self._render_info_screen(state)
        else:
            self._render_home(state)

        self._flush()

    # ---- Status bar (shared) ----

    def _draw_status_bar(self, state):
        """Top 20px: clock + BLE indicator + mood bar + battery."""
        y = 2

        # Battery indicator (right-aligned, before clock)
        bat = state.get("battery_info")
        bat_str = ""
        if bat and bat.get("available"):
            pct = bat.get("percentage", -1)
            if pct >= 0:
                if pct > 60:
                    bat_color = COLOR_GREEN
                elif pct > 20:
                    bat_color = COLOR_YELLOW
                else:
                    bat_color = COLOR_RED
                bat_str = "{}%".format(pct)
                if bat.get("charging"):
                    bat_str += "+"
                bw = self.font_sm.getbbox(bat_str)[2]
                self.draw.text(
                    (self.W - bw - 8, y), bat_str,
                    fill=bat_color, font=self.font_sm,
                )

        # Clock (right-aligned, shifted left if battery shown)
        time_str = state.get("time_str", "--:--")
        tw = self.font_sm.getbbox(time_str)[2]
        clock_x = self.W - tw - 8
        if bat_str:
            bw = self.font_sm.getbbox(bat_str)[2]
            clock_x = self.W - tw - bw - 16
        self.draw.text((clock_x, y), time_str, fill=COLOR_DIM, font=self.font_sm)

        # BLE connection dot (left side) — neon cyan glow when connected
        ble_connected = state.get("ble_connected", False)
        if ble_connected:
            # Glow backdrop
            self.draw.ellipse([6, y, 18, y + 12], fill=COLOR_CYAN_DIM)
            self.draw.ellipse([8, y + 2, 16, y + 10], fill=COLOR_CYAN)
        else:
            self.draw.ellipse([8, y + 2, 16, y + 10], fill=COLOR_CIRCUIT_NODE)

        # Nighttime sleep indicator (moon icon next to BLE dot)
        pet_info = state.get("pet_info")
        if pet_info and pet_info.get("is_sleeping"):
            self.draw.text((20, y), "\u263D", fill=COLOR_YELLOW, font=self.font_sm)

        # Mood indicator bar (center)
        if pet_info:
            mood = pet_info.get("mood", "neutral")
            mood_color = self._mood_color(mood)
            bar_w = 50
            bar_x = (self.W - bar_w) // 2
            self.draw.rectangle([bar_x, y + 1, bar_x + bar_w, y + 5],
                                fill=COLOR_BAR_BG)
            score = pet_info.get("mood_score", 0.0)
            fill_w = int((score + 1) / 2 * bar_w)
            fill_w = max(2, min(fill_w, bar_w))
            self.draw.rectangle([bar_x, y + 1, bar_x + fill_w, y + 5],
                                fill=mood_color)

        # Separator line (neon tinted)
        self.draw.line([(0, 19), (self.W, 19)], fill=COLOR_SEPARATOR, width=1)

    # ---- HOME screen ----

    def _render_home(self, state):
        """Home: circuit background + animated sprite + XP bar + vitals."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Center the sprite in the pet zone
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + (self.PET_H - PET_SPRITE_SIZE) // 2 - 15
        self.img.paste(sprite_frame, (sx, sy), sprite_frame)

        # Pet name
        pet_info = state.get("pet_info")
        # Title line: "Cortex · B18" (name + bloom number)
        bloom = pet_info.get("bloom", 0) if pet_info else 0
        if bloom > 0:
            pet_title = f"Cortex \u00b7 B{bloom}"
        else:
            pet_title = pet_info.get("name", "Cortex") if pet_info else "Cortex"
        name_w = self.font_md.getbbox(pet_title)[2]
        self.draw.text(
            ((self.W - name_w) // 2, sy + PET_SPRITE_SIZE + 4),
            pet_title, fill=COLOR_TEXT, font=self.font_md,
        )

        # Stage + mood label (or sleeping indicator)
        if pet_info:
            if pet_info.get("is_sleeping"):
                label = "Sleeping..."
                lw = self.font_sm.getbbox(label)[2]
                self.draw.text(
                    ((self.W - lw) // 2, sy + PET_SPRITE_SIZE + 22),
                    label, fill=COLOR_DIM, font=self.font_sm,
                )
            else:
                mood = pet_info.get("mood", "neutral")
                stage_name = pet_info.get("stage_name", "")
                label = f"{stage_name} \u00b7 {mood}"
                lw = self.font_sm.getbbox(label)[2]
                self.draw.text(
                    ((self.W - lw) // 2, sy + PET_SPRITE_SIZE + 22),
                    label, fill=self._mood_color(mood), font=self.font_sm,
                )

        # ---- Info bar (y=200) ----
        self.draw.line([(0, self.INFO_Y), (self.W, self.INFO_Y)],
                       fill=COLOR_SEPARATOR, width=1)

        # Vitals bars
        if pet_info:
            self._draw_vitals_icons(pet_info, y=self.INFO_Y + 3)

        # XP / Evolution progress bar
        self._draw_xp_bar(state, y=self.INFO_Y + 30)

        # Intelligence score
        if pet_info:
            iq = pet_info.get("intelligence", 0)
            iq_str = f"IQ:{iq:.0f}"
            iqw = self.font_sm.getbbox(iq_str)[2]
            self.draw.text(
                (self.W - iqw - 10, self.INFO_Y + 30),
                iq_str, fill=COLOR_VITAL_IQ, font=self.font_sm,
            )

        # Thought bubble (from heartbeat)
        thought = state.get("thought_bubble", "")
        if thought:
            self._draw_thought_bubble(thought, sx, sy)

        # Button hints footer
        self._draw_footer_hints(state, [
            ("A", "Talk", COLOR_CYAN),
            ("X", "Feed", COLOR_VITAL_HUNGER),
            ("Y", "Clean", COLOR_VITAL_CLEAN),
            ("\u2606", "Menu", COLOR_DIM),
        ])

    def _draw_thought_bubble(self, text, sprite_x, sprite_y):
        """Draw a thought bubble near the pet sprite on HOME screen."""
        # Truncate long thoughts
        max_chars = 60
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        # Word wrap to fit in bubble
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if self.font_sm.getbbox(test)[2] > 160:
                if current:
                    lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        lines = lines[:3]  # max 3 lines

        if not lines:
            return

        # Calculate bubble dimensions
        line_h = 13
        pad = 5
        bw = max(self.font_sm.getbbox(ln)[2] for ln in lines) + pad * 2
        bh = len(lines) * line_h + pad * 2

        # Position: to the right of sprite, or above if no room
        bx = sprite_x + PET_SPRITE_SIZE + 8
        by = sprite_y + 10
        if bx + bw > self.W - 4:
            bx = sprite_x - bw - 8
        if bx < 4:
            bx = (self.W - bw) // 2
            by = sprite_y - bh - 10

        # Draw bubble with dotted border (thought style)
        self.draw.rounded_rectangle(
            [bx, by, bx + bw, by + bh],
            radius=6, fill=COLOR_SPEECH_BG,
            outline=COLOR_DIM, width=1,
        )

        # Small circles (thought tail) leading to sprite
        cx = bx - 4 if bx > sprite_x + PET_SPRITE_SIZE else bx + bw + 2
        cy = by + bh // 2
        self.draw.ellipse([cx, cy, cx + 4, cy + 4], fill=COLOR_DIM)
        self.draw.ellipse([cx - 3, cy + 5, cx + 1, cy + 8], fill=COLOR_DIM)

        # Draw text lines
        for i, line in enumerate(lines):
            self.draw.text(
                (bx + pad, by + pad + i * line_h),
                line, fill=COLOR_CYAN_DIM, font=self.font_sm,
            )

    # ---- PET_SLEEPING screen ----

    def _render_pet_sleeping(self, state):
        """Sleeping screen: dimmed sprite, moon, sleep reason, vitals."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Dimmed sleeping sprite
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + 20

        from PIL import Image
        dimmed = sprite_frame.copy()
        alpha = dimmed.split()[-1] if dimmed.mode == "RGBA" else None
        dimmed = dimmed.convert("RGB")
        dimmed = Image.blend(
            Image.new("RGB", dimmed.size, COLOR_BG), dimmed, 0.5
        )
        if alpha:
            dimmed.putalpha(alpha)
        self.img.paste(dimmed, (sx, sy), dimmed if dimmed.mode == "RGBA" else None)

        # Moon + "Zzz" animation
        phase = int(time.monotonic() * 1.5) % 3
        moon_x = sx + PET_SPRITE_SIZE - 5
        self.draw.text((moon_x, sy - 10), "\u263D",
                       fill=COLOR_YELLOW, font=self.font_md)
        for i in range(phase + 1):
            zx = moon_x + 15 + i * 8
            zy = sy - 15 - i * 8
            fade = 1.0 - i * 0.25
            c = tuple(int(v * fade) for v in COLOR_CYAN)
            self.draw.text((zx, zy), "z", fill=c, font=self.font_sm)

        # Label
        reason = state.get("sleep_reason", "user")
        if reason == "battery_low":
            label = "Sleeping... (low battery)"
        elif reason == "low_energy":
            label = "Sleeping... (tired)"
        else:
            label = "Sleeping peacefully..."
        lw = self.font_sm.getbbox(label)[2]
        self.draw.text(
            ((self.W - lw) // 2, sy + PET_SPRITE_SIZE + 6),
            label, fill=COLOR_DIM, font=self.font_sm,
        )

        # Battery / energy info
        bat = state.get("battery_info")
        if bat and bat.get("available"):
            pct = bat.get("percentage", 0)
            charging = bat.get("charging", False)
            info = f"Battery: {pct}%{'+' if charging else ''}"
            iw = self.font_sm.getbbox(info)[2]
            self.draw.text(
                ((self.W - iw) // 2, sy + PET_SPRITE_SIZE + 22),
                info, fill=COLOR_YELLOW if charging else COLOR_DIM,
                font=self.font_sm,
            )

        # Wake hint
        self._draw_footer_hints(state, [
            ("A/B", "Wake up", COLOR_CYAN),
        ])

    # ---- PET_DREAMING screen ----

    def _render_pet_dreaming(self, state):
        """Dream training screen: glowing sprite, dream particles, progress."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Glowing sprite (brighter than normal)
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + 20
        self.img.paste(sprite_frame, (sx, sy), sprite_frame)

        # Dream particles — floating dots in magenta/cyan
        now = time.monotonic()
        import math
        for i in range(6):
            angle = (now * 0.5 + i * 1.047) % (2 * math.pi)
            radius = 50 + math.sin(now * 0.3 + i) * 10
            px = int(sx + PET_SPRITE_SIZE // 2 + math.cos(angle) * radius)
            py = int(sy + PET_SPRITE_SIZE // 2 + math.sin(angle) * radius)
            color = COLOR_MAGENTA if i % 2 == 0 else COLOR_CYAN
            size = 2 + int(math.sin(now + i) * 1)
            self.draw.ellipse([px, py, px + size, py + size], fill=color)

        # "Dreaming..." label with pulsing glow
        pulse = int((math.sin(now * 2) + 1) * 60) + 100
        label = "Dreaming... (training)"
        lw = self.font_sm.getbbox(label)[2]
        glow = (pulse, 0, min(255, pulse + 50))
        self.draw.text(
            ((self.W - lw) // 2, sy + PET_SPRITE_SIZE + 6),
            label, fill=glow, font=self.font_sm,
        )

        # Sub-label
        sub = "Learning from today's conversations"
        sw = self.font_sm.getbbox(sub)[2]
        self.draw.text(
            ((self.W - sw) // 2, sy + PET_SPRITE_SIZE + 22),
            sub, fill=COLOR_DIM, font=self.font_sm,
        )

        # Battery info
        bat = state.get("battery_info")
        if bat and bat.get("available"):
            pct = bat.get("percentage", 0)
            info = f"Battery: {pct}%+"
            iw = self.font_sm.getbbox(info)[2]
            self.draw.text(
                ((self.W - iw) // 2, sy + PET_SPRITE_SIZE + 38),
                info, fill=COLOR_GREEN, font=self.font_sm,
            )

        # No wake hint — can't interrupt dreaming
        self._draw_footer_hints(state, [
            ("", "Please wait...", COLOR_DIM),
        ])

    # ---- MENU screen ----

    def _render_menu(self, state):
        """Menu overlay: mini sprite + breadcrumb + scrollable items."""
        self._draw_status_bar(state)

        menu_items = state.get("menu_items", [])
        menu_cursor = state.get("menu_cursor", 0)
        breadcrumb = state.get("menu_breadcrumb", "Menu")

        # Mini sprite top-left
        sprite_frame = self.sprites.get_frame()
        mini = sprite_frame.resize((28, 28), Image.NEAREST)
        self.img.paste(mini, (8, 24), mini)

        # Breadcrumb
        self.draw.text((42, 28), breadcrumb, fill=COLOR_CYAN, font=self.font_sm)

        # Separator
        y = 48
        self.draw.line([(8, y), (self.W - 8, y)], fill=COLOR_SEPARATOR, width=1)
        y += 4

        # Menu items (scrollable window)
        items_per_page = 7
        item_h = 28

        if len(menu_items) > items_per_page:
            scroll_start = max(0, min(menu_cursor - items_per_page // 2,
                                      len(menu_items) - items_per_page))
        else:
            scroll_start = 0

        visible_items = menu_items[scroll_start:scroll_start + items_per_page]

        for i, item in enumerate(visible_items):
            actual_idx = scroll_start + i
            iy = y + i * item_h
            is_selected = actual_idx == menu_cursor

            if is_selected:
                # Highlight background
                self.draw.rectangle(
                    [6, iy, self.W - 6, iy + item_h - 2],
                    fill=COLOR_HIGHLIGHT,
                )
                # Left accent bar
                self.draw.rectangle(
                    [4, iy + 2, 6, iy + item_h - 4],
                    fill=COLOR_CYAN,
                )
                # Cursor arrow
                self.draw.text((10, iy + 5), "\u25b6", fill=COLOR_CURSOR,
                               font=self.font_md)
                text_color = COLOR_TEXT
            else:
                text_color = COLOR_DIM

            # Item label
            label = item.label if hasattr(item, "label") else str(item)
            if hasattr(item, "is_branch") and item.is_branch:
                label += "  \u203a"
            self.draw.text((30, iy + 6), label, fill=text_color, font=self.font_md)

        # Scroll indicators
        if scroll_start > 0:
            self.draw.text((self.W // 2 - 4, y - 2), "\u25b2",
                           fill=COLOR_DIM, font=self.font_sm)
        if scroll_start + items_per_page < len(menu_items):
            self.draw.text(
                (self.W // 2 - 4, y + items_per_page * item_h),
                "\u25bc", fill=COLOR_DIM, font=self.font_sm,
            )

        # Footer
        self._draw_footer_hints(state, [
            ("A", "Select", COLOR_CYAN),
            ("B", "Back", COLOR_DIM),
        ])

    # ---- PET_ASKING screen ----

    def _render_pet_asking(self, state):
        """Thinking animation + prompt text."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Centered sprite
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + 10
        self.img.paste(sprite_frame, (sx, sy), sprite_frame)

        # "Thinking..." with animated dots
        dots = "." * (1 + (int(time.monotonic() * 2) % 3))
        think_text = "Thinking" + dots
        tw = self.font_md.getbbox(think_text)[2]
        self.draw.text(
            ((self.W - tw) // 2, sy + PET_SPRITE_SIZE + 6),
            think_text, fill=COLOR_PET_NEUTRAL, font=self.font_md,
        )

        # Show prompt (smaller, below)
        pet_prompt = state.get("pet_prompt", "")
        if pet_prompt:
            y = sy + PET_SPRITE_SIZE + 28
            self.draw.text((14, y), "You:", fill=COLOR_DIM, font=self.font_sm)
            y += 14
            lines = _word_wrap(pet_prompt, self.font_sm, self.W - 28)
            for line in lines[:3]:
                self.draw.text((14, y), line, fill=COLOR_TEXT, font=self.font_sm)
                y += 14

        # Footer
        self._draw_footer_hints(state, [
            ("B", "Cancel", COLOR_YELLOW),
        ])

    # ---- PET_RESPONSE screen ----

    def _render_pet_response(self, state):
        """Vertical layout: sprite at top, angular speech bubble below."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Sprite centered at top of pet zone
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + 4
        self.img.paste(sprite_frame, (sx, sy), sprite_frame)

        # Angular speech bubble below sprite
        bubble_x = 8
        bubble_y = sy + PET_SPRITE_SIZE + 4
        bubble_w = self.W - 16
        bubble_h = 72
        self._draw_cyber_bubble(bubble_x, bubble_y, bubble_w, bubble_h,
                                pointer_dir="up")

        # Response text inside bubble
        pet_response = state.get("pet_response_text", "")
        if pet_response:
            lines = _word_wrap(pet_response, self.font_sm, bubble_w - 16)
            ty = bubble_y + 8
            for line in lines[:4]:
                self.draw.text((bubble_x + 8, ty), line, fill=COLOR_TEXT,
                               font=self.font_sm)
                ty += 14

        # Stats below bubble
        y = bubble_y + bubble_h + 4
        pet_resp_data = state.get("pet_resp_data")
        if pet_resp_data:
            ms = pet_resp_data.get("inference_time_ms", 0)
            tokens = pet_resp_data.get("tokens", 0)
            stats_text = f"{ms}ms \u00b7 {tokens} tokens"
            self.draw.text((14, y), stats_text, fill=COLOR_DIM, font=self.font_sm)

        # Mood/stage
        pet_info = state.get("pet_info")
        if pet_info and pet_resp_data:
            mood = pet_info.get("mood", "neutral")
            stage_name = pet_info.get("stage_name", "")
            self.draw.text(
                (14, y + 14), f"{stage_name} \u00b7 {mood}",
                fill=self._mood_color(mood), font=self.font_sm,
            )

        # Footer
        self._draw_footer_hints(state, [
            ("B", "Dismiss", COLOR_PET_NEUTRAL),
        ])

    # ---- STT_LISTENING screen ----

    def _render_stt_listening(self, state):
        """Listening screen with sprite + mic indicator."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Sprite centered
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + 10
        self.img.paste(sprite_frame, (sx, sy), sprite_frame)

        # Mic indicator (pulsing cyan)
        mic_text = "\u25cf LISTENING"
        mw = self.font_md.getbbox(mic_text)[2]
        self.draw.text(
            ((self.W - mw) // 2, sy + PET_SPRITE_SIZE + 6),
            mic_text, fill=COLOR_CYAN, font=self.font_md,
        )

        # Partial transcript
        partial = state.get("stt_partial", "")
        if partial:
            y = sy + PET_SPRITE_SIZE + 28
            lines = _word_wrap(f'"{partial}"', self.font_sm, self.W - 28)
            for line in lines[:3]:
                self.draw.text((14, y), line, fill=COLOR_TEXT, font=self.font_sm)
                y += 14

        # Command hints
        y = self.INFO_Y + 8
        self.draw.text((14, y), '"note" \u00b7 "record" \u00b7 "pet"',
                       fill=COLOR_CYAN_DIM, font=self.font_sm)

        # Footer
        self._draw_footer_hints(state, [
            ("B", "Cancel", COLOR_YELLOW),
        ])

    # ---- NOTE_TAKING screen ----

    def _render_note_taking(self, state):
        """Note-taking with live transcript."""
        self._draw_status_bar(state)

        pet_mode = state.get("pet_mode", False)
        badge_text = "PET" if pet_mode else "NOTE"
        badge_color = COLOR_PET_NEUTRAL if pet_mode else COLOR_CYAN

        # Angular badge
        badge_w = self.font_md.getbbox(f" {badge_text} ")[2]
        self.draw.rectangle([10, 24, 10 + badge_w, 44], fill=badge_color)
        # Chamfered corner cut (top-right)
        self.draw.polygon([(10 + badge_w - 4, 24), (10 + badge_w, 24),
                           (10 + badge_w, 28)], fill=COLOR_BG)
        self.draw.text((14, 26), f" {badge_text} ", fill=COLOR_BG, font=self.font_md)

        # Transcript area
        y = 52
        self.draw.line([(10, y), (self.W - 10, y)], fill=COLOR_SEPARATOR, width=1)
        y += 6

        note_text = state.get("note_text", "")
        partial = state.get("stt_partial", "")
        display_text = note_text
        if partial:
            display_text = f"{display_text} {partial}".strip() if display_text else partial

        if display_text:
            lines = _word_wrap(display_text, self.font_sm, self.W - 28)
            max_lines = 9
            visible = lines[-max_lines:]
            for line in visible:
                self.draw.text((14, y), line, fill=COLOR_TEXT, font=self.font_sm)
                y += 15
        else:
            placeholder = "Ask your pet..." if pet_mode else "Speak your note..."
            self.draw.text((14, y + 10), placeholder, fill=COLOR_DIM, font=self.font_md)

        # Footer
        if pet_mode:
            self._draw_footer_hints(state, [
                ("Btn", "Send", COLOR_PET_NEUTRAL),
            ])
        else:
            self._draw_footer_hints(state, [
                ("Btn", "Save", COLOR_GREEN),
            ])

    # ---- PET_STATUS screen ----

    def _render_pet_status(self, state):
        """Detailed pet status display."""
        self._draw_status_bar(state)

        pet_info = state.get("pet_info")
        if not pet_info:
            self.draw.text((14, 40), "No pet data", fill=COLOR_DIM, font=self.font_md)
            self._draw_footer_hints(state, [("B", "Back", COLOR_DIM)])
            return

        # Sprite (resized to 48x48 for this detail screen)
        sprite_frame = self.sprites.get_frame()
        mini_sprite = sprite_frame.resize((48, 48), Image.NEAREST)
        self.img.paste(mini_sprite, (10, 28), mini_sprite)

        # Pet info next to sprite
        x = 10 + 48 + 12
        y = 30
        name = pet_info.get("name", "Pet")
        mood = pet_info.get("mood", "neutral")
        stage_name = pet_info.get("stage_name", "")
        mood_score = pet_info.get("mood_score", 0.0)
        interactions = pet_info.get("total_interactions", 0)

        self.draw.text((x, y), name, fill=COLOR_TEXT, font=self.font_md)
        y += 20
        self.draw.text((x, y), f"Stage: {stage_name}", fill=COLOR_DIM,
                       font=self.font_sm)
        y += 16
        self.draw.text((x, y), f"Mood: {mood} ({mood_score:+.2f})",
                       fill=self._mood_color(mood), font=self.font_sm)

        # Detailed stats below
        y = 90
        self.draw.line([(10, y), (self.W - 10, y)], fill=COLOR_SEPARATOR, width=1)
        y += 8

        stats = [
            f"Interactions: {interactions}",
            f"Model: {pet_info.get('model', 'unknown')}",
        ]
        if interactions < PET_STAGE_THRESHOLDS[-1]:
            for i, threshold in enumerate(PET_STAGE_THRESHOLDS):
                if interactions < threshold:
                    stats.append(f"Next stage: {threshold - interactions} more")
                    break

        for line in stats:
            self.draw.text((14, y), line, fill=COLOR_DIM, font=self.font_sm)
            y += 18

        # XP bar
        self._draw_xp_bar(state, y=y + 4)

        # Mood bar (detailed)
        y += 28
        self.draw.text((14, y), "Mood History:", fill=COLOR_DIM, font=self.font_sm)
        y += 16
        bar_x = 14
        bar_w = self.W - 28
        self._draw_neon_bar(bar_x, y, bar_w, 8,
                            (mood_score + 1) / 2, self._mood_color(mood))

        # Footer
        self._draw_footer_hints(state, [("B", "Back", COLOR_DIM)])

    # ---- CONFIRM_SHUTDOWN screen ----

    def _render_confirm_shutdown(self, state):
        """Shutdown confirmation dialog with cyberpunk panel."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Dark panel
        px, py, pw, ph = 20, 80, self.W - 40, 100
        self.draw.rectangle([px, py, px + pw, py + ph], fill=COLOR_MENU_BG)
        # Double neon border
        self.draw.rectangle([px, py, px + pw, py + ph], outline=COLOR_RED, width=2)
        self.draw.rectangle([px + 3, py + 3, px + pw - 3, py + ph - 3],
                            outline=COLOR_RED, width=1)

        # Warning text
        self.draw.text((40, 95), "Shutdown?", fill=COLOR_RED, font=self.font_lg)
        self.draw.text((40, 125), "Are you sure?", fill=COLOR_TEXT, font=self.font_md)

        # Options
        self.draw.text((40, 155), "A: Yes", fill=COLOR_RED, font=self.font_md)
        self.draw.text((130, 155), "B: No", fill=COLOR_GREEN, font=self.font_md)

    # ---- RECORDING screen ----

    def _render_recording(self, state):
        """Recording / Paused screen."""
        self._draw_status_bar(state)

        app = state.get("app_state", "RECORDING")
        is_paused = app == "PAUSED"

        # Large recording indicator
        y = 40
        if is_paused:
            badge_text = "PAUSED"
            badge_color = COLOR_YELLOW
        else:
            badge_text = "\u25cf REC"
            badge_color = COLOR_RED

        bw = self.font_lg.getbbox(badge_text)[2]
        self.draw.text(
            ((self.W - bw) // 2, y),
            badge_text, fill=badge_color, font=self.font_lg,
        )

        # Elapsed time
        y += 30
        elapsed = _format_duration(state.get("session_elapsed", 0))
        ew = self.font_lg.getbbox(elapsed)[2]
        self.draw.text(
            ((self.W - ew) // 2, y),
            elapsed, fill=COLOR_TEXT, font=self.font_lg,
        )

        # Segment info
        y += 35
        seg_count = state.get("segment_count", 0)
        self.draw.text((14, y), f"Segment #{seg_count}", fill=COLOR_DIM,
                       font=self.font_sm)

        # Segment progress bar (neon style)
        y += 18
        bar_color = COLOR_RED if not is_paused else COLOR_YELLOW
        seg_elapsed = state.get("segment_elapsed", 0)
        progress = min(seg_elapsed / SEGMENT_SECONDS, 1.0) if SEGMENT_SECONDS > 0 else 0
        self._draw_neon_bar(14, y, self.W - 28, 10, progress, bar_color)

        # Disk stats
        y += 20
        disk_free = state.get("disk_free", 0)
        remaining_h = state.get("remaining_hours", 0)
        self.draw.text(
            (14, y), f"Free: {_format_size(disk_free)}  (~{int(remaining_h)}h)",
            fill=COLOR_DIM, font=self.font_sm,
        )

        # Footer
        if is_paused:
            self._draw_footer_hints(state, [
                ("Btn", "Resume", COLOR_GREEN),
                ("Hold", "Stop", COLOR_DIM),
            ])
        else:
            self._draw_footer_hints(state, [
                ("Btn", "Pause", COLOR_YELLOW),
                ("Hold", "Stop", COLOR_DIM),
            ])

    # ---- Shared components ----

    def _draw_xp_bar(self, state, y=204):
        """Draw XP/evolution progress bar (neon style)."""
        pet_info = state.get("pet_info")
        if not pet_info:
            return

        interactions = pet_info.get("total_interactions", 0)

        # Find current and next stage thresholds
        current_threshold = 0
        next_threshold = PET_STAGE_THRESHOLDS[-1]
        for i, threshold in enumerate(PET_STAGE_THRESHOLDS):
            if interactions >= threshold:
                current_threshold = threshold
                if i + 1 < len(PET_STAGE_THRESHOLDS):
                    next_threshold = PET_STAGE_THRESHOLDS[i + 1]
                else:
                    next_threshold = threshold

        range_size = max(1, next_threshold - current_threshold)
        progress = min((interactions - current_threshold) / range_size, 1.0)

        bar_x = 10
        bar_w = self.W - 20
        self._draw_neon_bar(bar_x, y, bar_w, 8, progress, COLOR_XP_BAR)

        # Label
        xp_label = f"XP: {interactions}/{next_threshold}"
        self.draw.text((bar_x, y + 10), xp_label, fill=COLOR_DIM, font=self.font_sm)

    def _draw_footer_hints(self, state, hints):
        """Draw button hint footer at bottom of screen."""
        y = self.H - 28
        self.draw.line([(0, y - 4), (self.W, y - 4)], fill=COLOR_SEPARATOR, width=1)

        x = 10
        for key, action, color in hints:
            # Key badge
            key_text = f"[{key}]"
            self.draw.text((x, y), key_text, fill=color, font=self.font_sm)
            kw = self.font_sm.getbbox(key_text)[2]
            # Action label
            self.draw.text((x + kw + 4, y), action, fill=COLOR_DIM, font=self.font_sm)
            aw = self.font_sm.getbbox(action)[2]
            x += kw + aw + 16

    # ---- Vitals bars ----

    # ── Icon polygon definitions (relative coordinates, 16x20 bounding box) ──
    # Each is a list of (x, y) tuples forming the outline.

    _ICON_BOLT = [  # Lightning bolt (Energy)
        (9, 0), (3, 10), (7, 10), (5, 20), (13, 8), (9, 8), (11, 0),
    ]
    _ICON_FORK = [  # Fork/utensil (Hunger)
        (4, 0), (4, 8), (7, 8), (7, 0), (9, 0), (9, 8), (12, 8),
        (12, 0), (14, 0), (14, 10), (10, 12), (10, 20), (6, 20),
        (6, 12), (2, 10), (2, 0),
    ]
    _ICON_DROP = [  # Water drop (Cleanliness)
        (8, 0), (13, 8), (14, 12), (13, 15), (11, 18), (8, 20),
        (5, 18), (3, 15), (2, 12), (3, 8),
    ]
    _ICON_HEART = [  # Heart (Happiness)
        (8, 4), (5, 0), (2, 0), (0, 2), (0, 6), (2, 10),
        (8, 18), (14, 10), (16, 6), (16, 2), (14, 0), (11, 0),
    ]

    def _draw_vital_icon(self, cx, cy, icon_pts, fill_pct, color, low_color):
        """Draw a single vital icon with fill level.

        Args:
            cx, cy: Center position of the icon area.
            icon_pts: Polygon points (16x20 relative).
            fill_pct: 0.0 to 1.0 fill level.
            color: Normal color.
            low_color: Color when critical.
        """
        iw, ih = 16, 20
        ox = cx - iw // 2
        oy = cy - ih // 2

        flash = int(time.monotonic() * 3) % 2 == 0
        fill_pct = max(0.0, min(1.0, fill_pct))

        # Determine color
        if fill_pct < PET_VITAL_CRITICAL and flash:
            fill_color = COLOR_RED
            outline_color = COLOR_RED
        elif fill_pct < PET_VITAL_LOW:
            fill_color = low_color
            outline_color = low_color
        else:
            fill_color = color
            outline_color = color

        # Dim outline color (30% brightness)
        dim_outline = tuple(max(0, c // 3) for c in outline_color)

        # Build absolute polygon
        abs_pts = [(ox + px, oy + py) for px, py in icon_pts]

        # Draw filled portion using a mask image
        # Create a small mask for the icon, fill from bottom up
        mask = Image.new("L", (iw + 2, ih + 2), 0)
        mask_draw = ImageDraw.Draw(mask)
        shifted_pts = [(px + 1, py + 1) for px, py in icon_pts]
        mask_draw.polygon(shifted_pts, fill=255)

        # Fill level: clear pixels above the fill line
        fill_y = int(ih * (1.0 - fill_pct))
        if fill_y > 0:
            mask_draw.rectangle([0, 0, iw + 1, fill_y], fill=0)

        # Draw the filled portion pixel by pixel (fast enough for 16x20)
        for py in range(ih + 2):
            for px in range(iw + 2):
                if mask.getpixel((px, py)) > 0:
                    sx = ox + px - 1
                    sy = oy + py - 1
                    if 0 <= sx < self.W and 0 <= sy < self.H:
                        self.draw.point((sx, sy), fill=fill_color)

        # Draw outline on top
        self.draw.polygon(abs_pts, outline=dim_outline)

        # Glow highlight on the fill line
        if 0 < fill_pct < 1.0:
            line_y = oy + fill_y
            # Draw a subtle glow line across the icon at fill level
            bright = tuple(min(255, int(c * 1.5)) for c in fill_color)
            for px in range(iw):
                mx = px + 1
                if mask.getpixel((mx, fill_y + 1)) > 0 or (
                    fill_y + 2 <= ih + 1 and mask.getpixel((mx, fill_y + 2)) > 0
                ):
                    self.draw.point((ox + px, line_y), fill=bright)

    def _draw_vitals_icons(self, pet_info, y=203):
        """Draw 4 vital icons in a row with fill levels."""
        vitals = [
            (self._ICON_FORK, pet_info.get("hunger", 1.0),
             COLOR_VITAL_HUNGER, COLOR_VITAL_HUNGER_LOW),
            (self._ICON_DROP, pet_info.get("cleanliness", 1.0),
             COLOR_VITAL_CLEAN, COLOR_VITAL_CLEAN_LOW),
            (self._ICON_BOLT, pet_info.get("energy", 1.0),
             COLOR_VITAL_ENERGY, COLOR_VITAL_ENERGY_LOW),
            (self._ICON_HEART, pet_info.get("happiness", 1.0),
             COLOR_PET_HAPPY, COLOR_PET_SAD),
        ]

        # Space 4 icons evenly across the 240px width
        n = len(vitals)
        spacing = self.W // (n + 1)

        for i, (icon_pts, value, color, low_color) in enumerate(vitals):
            cx = spacing * (i + 1)
            cy = y + 12  # center vertically in the icon area
            self._draw_vital_icon(cx, cy, icon_pts, value, color, low_color)

    # ---- Pet Feeding screen ----

    def _render_pet_feeding(self, state):
        """Brief feeding animation — auto-dismisses after 2 seconds."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Sprite center
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + (self.PET_H - PET_SPRITE_SIZE) // 2 - 20
        self.img.paste(sprite_frame, (sx, sy), sprite_frame)

        # "Nom nom nom!" text
        nom = "Nom nom nom!"
        nw = self.font_md.getbbox(nom)[2]
        self.draw.text(
            ((self.W - nw) // 2, sy + PET_SPRITE_SIZE + 8),
            nom, fill=COLOR_PET_HAPPY, font=self.font_md,
        )

        # Hunger bar change indicator
        pet_info = state.get("pet_info")
        if pet_info:
            hunger = pet_info.get("hunger", 0)
            label = f"Hunger: {hunger:.0%}"
            lw = self.font_sm.getbbox(label)[2]
            self.draw.text(
                ((self.W - lw) // 2, sy + PET_SPRITE_SIZE + 28),
                label, fill=COLOR_VITAL_HUNGER, font=self.font_sm,
            )

    # ---- Pet Cleaning screen ----

    def _render_pet_cleaning(self, state):
        """Show bad interactions for review — A=discard, B=keep."""
        self._draw_status_bar(state)

        # Title
        title = "DATA CLEANUP"
        tw = self.font_md.getbbox(title)[2]
        self.draw.text(
            ((self.W - tw) // 2, self.PET_Y + 5),
            title, fill=COLOR_VITAL_CLEAN, font=self.font_md,
        )
        self.draw.line(
            [(10, self.PET_Y + 22), (self.W - 10, self.PET_Y + 22)],
            fill=COLOR_SEPARATOR, width=1,
        )

        interactions = state.get("cleaning_interactions", [])
        cursor = state.get("cleaning_cursor", 0)
        discarded = state.get("cleaning_discarded", [])

        if not interactions:
            msg = "No bad data to clean!"
            mw = self.font_sm.getbbox(msg)[2]
            self.draw.text(
                ((self.W - mw) // 2, self.PET_Y + 60),
                msg, fill=COLOR_PET_HAPPY, font=self.font_sm,
            )
        elif cursor < len(interactions):
            bad = interactions[cursor]
            # Progress indicator
            prog = f"{cursor + 1}/{len(interactions)}"
            self.draw.text(
                (10, self.PET_Y + 28), prog,
                fill=COLOR_DIM, font=self.font_sm,
            )

            # Show the bad prompt
            prompt = bad.get("prompt", "")[:60]
            score = bad.get("sentiment_score", 0)
            y = self.PET_Y + 45
            wrapped = _word_wrap(f'"{prompt}"', self.font_sm, self.W - 20)
            for line in wrapped[:3]:
                self.draw.text((10, y), line, fill=COLOR_TEXT, font=self.font_sm)
                y += 14

            # Sentiment score
            score_str = f"Sentiment: {score:.2f}"
            self.draw.text(
                (10, y + 4), score_str,
                fill=COLOR_PET_SAD if score < -0.5 else COLOR_YELLOW,
                font=self.font_sm,
            )

            # Cleanliness indicator
            pet_info = state.get("pet_info")
            if pet_info:
                clean = pet_info.get("cleanliness", 0)
                cstr = f"Clean: {clean:.0%}  Discarded: {len(discarded)}"
                self.draw.text(
                    (10, self.INFO_Y + 10), cstr,
                    fill=COLOR_VITAL_CLEAN, font=self.font_sm,
                )
        else:
            msg = f"Done! Discarded {len(discarded)} bad entries."
            mw = self.font_sm.getbbox(msg)[2]
            self.draw.text(
                ((self.W - mw) // 2, self.PET_Y + 60),
                msg, fill=COLOR_PET_HAPPY, font=self.font_sm,
            )

        self._draw_footer_hints(state, [
            ("A", "Discard", COLOR_RED),
            ("B", "Keep", COLOR_GREEN),
            ("\u2605", "Done", COLOR_DIM),
        ])

    # ---- Pet Coma screen ----

    def _render_pet_coma(self, state):
        """Coma screen: dimmed sleeping sprite, zzz particles, revival bars."""
        self._draw_status_bar(state)
        self._draw_circuit_background()

        # Dimmed sleeping sprite
        sprite_frame = self.sprites.get_frame()
        sx = (self.W - PET_SPRITE_SIZE) // 2
        sy = self.PET_Y + 20

        # Create dimmed version (40% opacity blend with BG)
        dimmed = sprite_frame.copy()
        alpha = dimmed.split()[-1] if dimmed.mode == "RGBA" else None
        dimmed = dimmed.convert("RGB")
        dimmed = Image.blend(
            Image.new("RGB", dimmed.size, COLOR_BG), dimmed, 0.4
        )
        if alpha:
            dimmed.putalpha(alpha)
        self.img.paste(dimmed, (sx, sy), dimmed if dimmed.mode == "RGBA" else None)

        # "Zzz" floating text with cyan glow animation
        phase = int(time.monotonic() * 2) % 3
        zzz_offsets = [(sx + PET_SPRITE_SIZE - 10, sy - 5),
                       (sx + PET_SPRITE_SIZE, sy - 15),
                       (sx + PET_SPRITE_SIZE - 5, sy - 25)]
        for i, (zx, zy) in enumerate(zzz_offsets):
            if i <= phase:
                fade = 1.0 - i * 0.3
                c = tuple(int(v * fade) for v in COLOR_CYAN)
                self.draw.text((zx, zy), "z", fill=c, font=self.font_md)

        # "Deep sleep..." label
        label = "Pet is in a deep sleep..."
        lw = self.font_sm.getbbox(label)[2]
        self.draw.text(
            ((self.W - lw) // 2, sy + PET_SPRITE_SIZE + 6),
            label, fill=COLOR_DIM, font=self.font_sm,
        )

        # Revival progress bars (neon style)
        pet_info = state.get("pet_info")
        if pet_info:
            y_start = sy + PET_SPRITE_SIZE + 24
            revival_header = "Revival Progress:"
            self.draw.text((10, y_start), revival_header,
                           fill=COLOR_TEXT, font=self.font_sm)
            y_start += 16

            threshold = PET_COMA_REVIVAL_THRESHOLD
            vitals = [
                ("Hunger", pet_info.get("hunger", 0),
                 COLOR_VITAL_HUNGER, threshold),
                ("Clean", pet_info.get("cleanliness", 0),
                 COLOR_VITAL_CLEAN, threshold),
                ("Energy", pet_info.get("energy", 0),
                 COLOR_VITAL_ENERGY, threshold),
            ]

            bar_w = 110
            for label_text, val, color, thresh in vitals:
                self.draw.text((10, y_start), label_text,
                               fill=COLOR_DIM, font=self.font_sm)
                bx = 70
                fill_pct = min(1.0, val / thresh) if thresh > 0 else 0
                self._draw_neon_bar(bx, y_start + 2, bar_w, 7, fill_pct, color)
                # Threshold marker
                tx = bx + bar_w + 4
                ok = val >= thresh
                self.draw.text(
                    (tx, y_start),
                    "\u2713" if ok else f"{val:.0%}",
                    fill=COLOR_GREEN if ok else COLOR_RED,
                    font=self.font_sm,
                )
                y_start += 16

        self._draw_footer_hints(state, [
            ("X", "Feed", COLOR_VITAL_HUNGER),
            ("Y", "Clean", COLOR_VITAL_CLEAN),
            ("\u2665", "Rest", COLOR_VITAL_ENERGY),
        ])

    # ---- SETTING_ADJUST screen ----

    def _render_setting_adjust(self, state):
        """Settings slider: name, bar, value, left/right hints."""
        self._draw_status_bar(state)

        name = state.get("setting_name", "Setting")
        value = state.get("setting_value", 0)
        s_min = state.get("setting_min", 0)
        s_max = state.get("setting_max", 100)

        # Title
        y = 40
        tw = self.font_lg.getbbox(name)[2]
        self.draw.text(
            ((self.W - tw) // 2, y), name,
            fill=COLOR_CYAN, font=self.font_lg,
        )

        # Current value
        y += 35
        val_str = str(value)
        if name in ("Brightness", "Volume"):
            val_str = f"{value}%"
        elif name == "Display Hz":
            val_str = f"{value} Hz"
        vw = self.font_lg.getbbox(val_str)[2]
        self.draw.text(
            ((self.W - vw) // 2, y), val_str,
            fill=COLOR_TEXT, font=self.font_lg,
        )

        # Slider bar
        y += 40
        bar_x = 30
        bar_w = self.W - 60
        s_range = max(1, s_max - s_min)
        progress = (value - s_min) / s_range
        self._draw_neon_bar(bar_x, y, bar_w, 12, progress, COLOR_CYAN)

        # Left / Right arrows
        self.draw.text((10, y - 2), "\u25c0", fill=COLOR_DIM, font=self.font_md)
        rw = self.font_md.getbbox("\u25b6")[2]
        self.draw.text((self.W - 10 - rw, y - 2), "\u25b6",
                       fill=COLOR_DIM, font=self.font_md)

        # Min / Max labels
        y += 22
        self.draw.text((bar_x, y), str(s_min), fill=COLOR_DIM, font=self.font_sm)
        max_str = str(s_max)
        mw = self.font_sm.getbbox(max_str)[2]
        self.draw.text((bar_x + bar_w - mw, y), max_str,
                       fill=COLOR_DIM, font=self.font_sm)

        # Footer
        self._draw_footer_hints(state, [
            ("\u2190\u2192", "Adjust", COLOR_CYAN),
            ("B", "Done", COLOR_DIM),
        ])

    # ---- INFO_SCREEN screen ----

    def _render_info_screen(self, state):
        """Generic info display: title + key-value lines."""
        self._draw_status_bar(state)

        title = state.get("info_title", "Info")
        lines = state.get("info_lines", [])

        # Title
        y = 30
        tw = self.font_lg.getbbox(title)[2]
        self.draw.text(
            ((self.W - tw) // 2, y), title,
            fill=COLOR_CYAN, font=self.font_lg,
        )

        # Separator
        y += 28
        self.draw.line([(10, y), (self.W - 10, y)],
                       fill=COLOR_SEPARATOR, width=1)
        y += 10

        # Key-value pairs
        for label, value in lines:
            # Label (dimmed)
            self.draw.text((14, y), f"{label}:", fill=COLOR_DIM, font=self.font_sm)
            # Value (bright, right-aligned to label)
            label_w = self.font_sm.getbbox(f"{label}: ")[2]
            self.draw.text((14 + label_w, y), str(value),
                           fill=COLOR_TEXT, font=self.font_sm)
            y += 20

            if y > self.H - 50:
                break  # prevent overflow

        # Footer
        self._draw_footer_hints(state, [
            ("B", "Back", COLOR_DIM),
        ])

    # ---- Helpers ----

    @staticmethod
    def _mood_color(mood):
        """Return color for a mood string."""
        if mood in ("happy", "content"):
            return COLOR_PET_HAPPY
        elif mood in ("uneasy", "sad"):
            return COLOR_PET_SAD
        return COLOR_PET_NEUTRAL

    def _flush(self):
        """Convert PIL RGB image to RGB565 bytes and send to display."""
        pixels = self.img.tobytes()
        buf = self._buf
        idx = 0
        for i in range(0, len(pixels), 3):
            r = pixels[i]
            g = pixels[i + 1]
            b = pixels[i + 2]
            rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            buf[idx] = (rgb565 >> 8) & 0xFF
            buf[idx + 1] = rgb565 & 0xFF
            idx += 2
        self.board.draw_image(0, 0, self.W, self.H, list(buf))
