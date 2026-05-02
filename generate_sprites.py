#!/usr/bin/env python3
"""Generate orange cat sprites for the Cortex AI pet.

Creates an 80x80 cute orange cat with warm gradients, big eyes,
pointed ears, and expressive mood-based animations.

Run once: python generate_sprites.py
"""

import os
import math
from PIL import Image, ImageDraw

# Sprite output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_DIR = os.path.join(SCRIPT_DIR, "assets", "sprites")

SIZE = 80
CX, CY = SIZE // 2, SIZE // 2  # 40, 40

# Orange cat color palette
ORANGE_LIGHT = (255, 190, 50)
ORANGE_MID = (240, 150, 30)
ORANGE_DARK = (200, 100, 20)
ORANGE_SHADOW = (160, 70, 10)
OUTLINE_YELLOW = (255, 230, 50)
EYE_DARK = (50, 30, 40)
NOSE_COLOR = (80, 50, 60)
WHITE = (255, 255, 255)
NEAR_WHITE = (255, 255, 240)

# Body geometry
HEAD_R = 18
HEAD_CY = 30
BODY_W = 30
BODY_H = 24
BODY_TOP = HEAD_CY + HEAD_R - 6
EAR_H = 14


def _dim(color, factor=0.5):
    return tuple(int(c * factor) for c in color)


def _lerp(c1, c2, t):
    """Lerp between two colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_body(draw, y_off=0, x_off=0):
    """Draw the cat's body (egg shape below head)."""
    ox = CX + x_off
    bt = y_off + BODY_TOP

    # Body ellipse
    draw.ellipse(
        [ox - BODY_W // 2, bt, ox + BODY_W // 2, bt + BODY_H],
        fill=ORANGE_DARK, outline=OUTLINE_YELLOW
    )

    # Front paws
    draw.ellipse([ox - 10, bt + BODY_H - 5, ox - 2, bt + BODY_H + 3], fill=ORANGE_DARK)
    draw.ellipse([ox + 2, bt + BODY_H - 5, ox + 10, bt + BODY_H + 3], fill=ORANGE_DARK)


def draw_tail(draw, y_off=0, x_off=0, wag=0):
    """Draw curled tail on right side."""
    ox = CX + x_off
    bt = y_off + BODY_TOP

    # Tail as thick curved line segments
    tail_pts = []
    for i in range(8):
        t = i / 7.0
        tx = ox + BODY_W // 2 - 2 + t * 14 + math.sin(t * 2.5 + wag) * 3
        ty = bt + BODY_H // 2 - t * 18
        tail_pts.append((tx, ty))

    for i in range(len(tail_pts) - 1):
        w = 4 - i * 0.3
        if w < 1:
            w = 1
        color = _lerp(ORANGE_DARK, ORANGE_LIGHT, i / len(tail_pts))
        draw.line([tail_pts[i], tail_pts[i + 1]], fill=color, width=max(1, int(w)))


def draw_head(draw, y_off=0, x_off=0):
    """Draw the cat head with gradient-like shading."""
    ox = CX + x_off
    hy = y_off + HEAD_CY

    # Head circle (main color)
    draw.ellipse(
        [ox - HEAD_R, hy - HEAD_R, ox + HEAD_R, hy + HEAD_R],
        fill=ORANGE_MID, outline=OUTLINE_YELLOW
    )

    # Inner highlight (lighter circle offset up-left)
    draw.ellipse(
        [ox - HEAD_R + 4, hy - HEAD_R + 2, ox + HEAD_R - 6, hy + 2],
        fill=ORANGE_LIGHT
    )


def draw_ears(draw, y_off=0, x_off=0):
    """Draw pointed ears with inner highlights."""
    ox = CX + x_off
    ear_base_y = y_off + HEAD_CY - HEAD_R + 4
    ear_tip_y = ear_base_y - EAR_H

    # Left ear
    l_ear = [
        (ox - HEAD_R + 3, ear_base_y + 2),
        (ox - HEAD_R + 11, ear_base_y + 2),
        (ox - HEAD_R + 1, ear_tip_y),
    ]
    draw.polygon(l_ear, fill=ORANGE_MID, outline=OUTLINE_YELLOW)
    # Inner ear
    l_inner = [
        (ox - HEAD_R + 5, ear_base_y + 1),
        (ox - HEAD_R + 9, ear_base_y + 1),
        (ox - HEAD_R + 3, ear_tip_y + 4),
    ]
    draw.polygon(l_inner, fill=ORANGE_LIGHT)

    # Right ear
    r_ear = [
        (ox + HEAD_R - 11, ear_base_y + 2),
        (ox + HEAD_R - 3, ear_base_y + 2),
        (ox + HEAD_R - 1, ear_tip_y),
    ]
    draw.polygon(r_ear, fill=ORANGE_MID, outline=OUTLINE_YELLOW)
    # Inner ear
    r_inner = [
        (ox + HEAD_R - 9, ear_base_y + 1),
        (ox + HEAD_R - 5, ear_base_y + 1),
        (ox + HEAD_R - 3, ear_tip_y + 4),
    ]
    draw.polygon(r_inner, fill=ORANGE_LIGHT)


def draw_eyes_normal(draw, y_off=0, x_off=0, look_x=0):
    """Draw big cute eyes with white glints."""
    ox = CX + x_off
    ey = y_off + HEAD_CY - 1
    spread = 8

    for ex in [ox - spread, ox + spread]:
        # Dark eye
        draw.ellipse([ex - 5, ey - 6, ex + 5, ey + 6], fill=EYE_DARK)
        # White glint (top-left)
        px = ex - 2 + look_x
        draw.ellipse([px - 2, ey - 4, px + 1, ey - 1], fill=WHITE)
        # Small glint (bottom-right)
        draw.ellipse([ex + 1, ey + 1, ex + 3, ey + 3], fill=NEAR_WHITE)


def draw_eyes_happy(draw, y_off=0, x_off=0):
    """Draw happy arc eyes (^_^)."""
    ox = CX + x_off
    ey = y_off + HEAD_CY - 1
    spread = 8

    for ex in [ox - spread, ox + spread]:
        # Happy arc
        draw.arc([ex - 5, ey - 4, ex + 5, ey + 6], 200, 340, fill=EYE_DARK, width=2)


def draw_eyes_closed(draw, y_off=0, x_off=0):
    """Draw closed eyes (sleeping/coma)."""
    ox = CX + x_off
    ey = y_off + HEAD_CY - 1
    spread = 8

    for ex in [ox - spread, ox + spread]:
        draw.line([(ex - 4, ey), (ex + 4, ey)], fill=EYE_DARK, width=2)


def draw_eyes_sad(draw, y_off=0, x_off=0):
    """Draw sad droopy eyes."""
    ox = CX + x_off
    ey = y_off + HEAD_CY - 1
    spread = 8

    for ex in [ox - spread, ox + spread]:
        draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=EYE_DARK)
        draw.ellipse([ex - 2, ey - 3, ex + 1, ey], fill=WHITE)


def draw_eyes_wide(draw, y_off=0, x_off=0):
    """Draw extra-wide curious/excited eyes."""
    ox = CX + x_off
    ey = y_off + HEAD_CY - 1
    spread = 8

    for ex in [ox - spread, ox + spread]:
        draw.ellipse([ex - 6, ey - 7, ex + 6, ey + 7], fill=EYE_DARK)
        draw.ellipse([ex - 3, ey - 5, ex, ey - 2], fill=WHITE)
        draw.ellipse([ex + 1, ey + 1, ex + 3, ey + 3], fill=NEAR_WHITE)


def draw_nose_mouth(draw, mood="neutral", y_off=0, x_off=0, open_mouth=False):
    """Draw small triangle nose and mouth."""
    ox = CX + x_off
    ny = y_off + HEAD_CY + 5

    # Nose triangle
    draw.polygon([(ox - 3, ny), (ox + 3, ny), (ox, ny + 3)], fill=NOSE_COLOR)

    # Mouth
    my = ny + 5
    if open_mouth:
        draw.ellipse([ox - 4, my - 2, ox + 4, my + 4], fill=(20, 5, 15))
    elif mood in ("happy", "content"):
        # W-shaped smile
        draw.line([(ox - 5, my + 1), (ox - 2, my - 1), (ox, my),
                   (ox + 2, my - 1), (ox + 5, my + 1)], fill=NOSE_COLOR, width=1)
    elif mood == "sad":
        # Frown
        draw.arc([ox - 4, my - 2, ox + 4, my + 4], 20, 160, fill=NOSE_COLOR, width=1)
    elif mood == "uneasy":
        # Wavy
        draw.line([(ox - 4, my), (ox - 2, my + 1), (ox, my - 1),
                   (ox + 2, my + 1), (ox + 4, my)], fill=NOSE_COLOR, width=1)
    elif mood == "thinking":
        # Small O
        draw.ellipse([ox - 2, my - 1, ox + 2, my + 2], outline=NOSE_COLOR)
    else:
        # Neutral W
        draw.line([(ox - 4, my), (ox - 1, my - 1), (ox, my),
                   (ox + 1, my - 1), (ox + 4, my)], fill=NOSE_COLOR, width=1)


def make_sprite(mood="neutral", y_off=0, x_off=0, eye_style="normal",
                look_x=0, open_mouth=False, tail_wag=0):
    """Create a complete sprite frame."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw in order: tail, body, head, ears, face
    draw_tail(draw, y_off=y_off, x_off=x_off, wag=tail_wag)
    draw_body(draw, y_off=y_off, x_off=x_off)
    draw_head(draw, y_off=y_off, x_off=x_off)
    draw_ears(draw, y_off=y_off, x_off=x_off)

    # Eyes
    if eye_style == "happy":
        draw_eyes_happy(draw, y_off=y_off, x_off=x_off)
    elif eye_style == "closed":
        draw_eyes_closed(draw, y_off=y_off, x_off=x_off)
    elif eye_style == "sad":
        draw_eyes_sad(draw, y_off=y_off, x_off=x_off)
    elif eye_style == "wide":
        draw_eyes_wide(draw, y_off=y_off, x_off=x_off)
    else:
        draw_eyes_normal(draw, y_off=y_off, x_off=x_off, look_x=look_x)

    draw_nose_mouth(draw, mood=mood, y_off=y_off, x_off=x_off,
                    open_mouth=open_mouth)
    return img


# ── Eye style mapping per mood ──

MOOD_EYES = {
    "happy": "happy",
    "content": "happy",
    "neutral": "normal",
    "uneasy": "sad",
    "sad": "sad",
}


# ── Animation generators ──────────────────────────────────────


def gen_idle(mood_name):
    """2-frame idle: subtle body bob."""
    eye = MOOD_EYES.get(mood_name, "normal")
    frames = []
    for i, y_off in enumerate([0, -2]):
        img = make_sprite(mood=mood_name, y_off=y_off, eye_style=eye, tail_wag=i * 0.5)
        frames.append((f"idle_{mood_name}_{i}.png", img))
    return frames


def gen_thinking():
    """3-frame thinking: eyes look around, thought dots."""
    frames = []
    look_dirs = [-1, 0, 1]
    for i, look in enumerate(look_dirs):
        img = make_sprite(mood="thinking", look_x=look, eye_style="wide")
        draw = ImageDraw.Draw(img)
        # Floating thought dots
        dot_x = CX + HEAD_R + 6
        dot_base_y = HEAD_CY - HEAD_R
        for d in range(i + 1):
            dy = dot_base_y - d * 6
            r = 3 - d
            if r < 1:
                r = 1
            brightness = max(0.3, 0.8 - d * 0.2)
            c = _dim(ORANGE_LIGHT, brightness)
            draw.ellipse([dot_x - r, dy - r, dot_x + r, dy + r], fill=c)
        frames.append((f"thinking_{i}.png", img))
    return frames


def gen_talking():
    """2-frame talking: mouth open/closed."""
    frames = []
    img0 = make_sprite(mood="neutral", open_mouth=True, eye_style="normal")
    frames.append(("talking_0.png", img0))
    img1 = make_sprite(mood="happy", eye_style="happy")
    frames.append(("talking_1.png", img1))
    return frames


def gen_sleeping():
    """2-frame sleeping: closed eyes, Zzz."""
    frames = []
    for i in range(2):
        img = make_sprite(mood="neutral", y_off=2, eye_style="closed")
        draw = ImageDraw.Draw(img)
        zx = CX + HEAD_R + 4
        zy = HEAD_CY - HEAD_R - 2 - i * 4
        draw.text((zx, zy + 6), "z", fill=_dim(ORANGE_LIGHT, 0.5))
        draw.text((zx + 6, zy), "Z", fill=_dim(ORANGE_LIGHT, 0.3))
        frames.append((f"sleeping_{i}.png", img))
    return frames


def gen_eating():
    """3-frame eating: approach -> bite -> happy."""
    frames = []
    img0 = make_sprite(mood="neutral", eye_style="wide")
    frames.append(("eating_0.png", img0))

    img1 = make_sprite(mood="neutral", open_mouth=True, eye_style="normal")
    draw1 = ImageDraw.Draw(img1)
    # Food bit
    draw1.rectangle([CX - 12, HEAD_CY + 12, CX - 8, HEAD_CY + 16], fill=(255, 180, 0))
    frames.append(("eating_1.png", img1))

    img2 = make_sprite(mood="happy", eye_style="happy")
    frames.append(("eating_2.png", img2))
    return frames


def gen_evolve():
    """4-frame evolve: glow effect around cat."""
    frames = []
    glow_levels = [0.0, 0.4, 1.0, 0.6]
    for i, glow in enumerate(glow_levels):
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if glow > 0:
            # Warm glow ring
            glow_a = int(100 * glow)
            glow_r = HEAD_R + int(glow * 12) + 4
            glow_color = (255, 200, 50, glow_a)
            draw.ellipse(
                [CX - glow_r, HEAD_CY - glow_r, CX + glow_r, HEAD_CY + glow_r],
                fill=glow_color
            )

        # Redraw full cat on top
        draw_tail(draw, wag=i * 0.3)
        draw_body(draw)
        draw_head(draw)
        draw_ears(draw)

        if i == 3:
            draw_eyes_happy(draw)
            draw_nose_mouth(draw, mood="happy")
        else:
            draw_eyes_closed(draw)
            draw_nose_mouth(draw, mood="neutral")

        frames.append((f"evolve_{i}.png", img))
    return frames


# ── Main ──────────────────────────────────────────────────────


def main():
    os.makedirs(SPRITE_DIR, exist_ok=True)

    all_frames = []

    # Idle animations — one per mood
    for mood in ["happy", "content", "neutral", "uneasy", "sad"]:
        all_frames.extend(gen_idle(mood))

    # Action animations
    all_frames.extend(gen_thinking())
    all_frames.extend(gen_talking())
    all_frames.extend(gen_sleeping())
    all_frames.extend(gen_eating())
    all_frames.extend(gen_evolve())

    # Save all frames
    for filename, img in all_frames:
        path = os.path.join(SPRITE_DIR, filename)
        img.save(path, "PNG")

    print(f"Generated {len(all_frames)} sprites ({SIZE}x{SIZE}) in {SPRITE_DIR}")
    for filename, _ in all_frames:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
