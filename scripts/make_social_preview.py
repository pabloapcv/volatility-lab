#!/usr/bin/env python3
"""Compose docs/social-preview.png (1280x640) from the light-theme screenshots."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
OUT = ROOT / "docs" / "social-preview.png"

W, H = 1280, 640
BG = (249, 250, 251)  # gray-50
WHITE = (255, 255, 255)
PRIMARY = (2, 132, 199)  # primary-600
PRIMARY_DARK = (3, 105, 161)
TEXT = (17, 24, 39)  # gray-900
MUTED = (107, 114, 128)  # gray-500
BORDER = (229, 231, 235)  # gray-200
GREEN = (34, 197, 94)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def round_corners(im: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *im.size), radius=radius, fill=255)
    out = im.convert("RGBA")
    out.putalpha(mask)
    return out


def drop_shadow(size: tuple[int, int], radius: int, offset: int = 8) -> Image.Image:
    shadow = Image.new("RGBA", (size[0] + offset * 4, size[1] + offset * 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.rounded_rectangle(
        (offset, offset, size[0] + offset, size[1] + offset),
        radius=radius,
        fill=(15, 23, 42, 45),
    )
    return shadow.filter(ImageFilter.GaussianBlur(12))


def crop_surface(src: Image.Image) -> Image.Image:
    """Crop the 3D plot card from the Surface Lab screenshot."""
    # Plot card sits under the toolbar; keep the surface + legend bar.
    w, h = src.size
    left = int(w * 0.04)
    right = int(w * 0.96)
    top = int(h * 0.28)
    bottom = int(h * 0.72)
    return src.crop((left, top, right, bottom))


def main() -> None:
    bold = font("/usr/share/fonts/truetype/macos/Inter-Bold.ttf", 54)
    semibold = font("/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf", 22)
    regular = font("/usr/share/fonts/truetype/macos/Inter-Regular.ttf", 18)
    small = font("/usr/share/fonts/truetype/macos/Inter-Medium.ttf", 15)
    mono = font("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf", 14)

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Top bar
    draw.rectangle((0, 0, W, 72), fill=WHITE)
    draw.line((0, 72, W, 72), fill=BORDER, width=1)

    # Logo tile
    rounded_rect(draw, (28, 16, 76, 56), 10, PRIMARY)
    draw.text((40, 22), "VL", font=semibold, fill=WHITE)

    draw.text((90, 14), "VolatilityLab", font=semibold, fill=TEXT)
    draw.text((90, 40), "Quantitative options research", font=small, fill=MUTED)

    # Live pill
    pill = "Live demo"
    tw = draw.textlength(pill, font=small)
    rounded_rect(draw, (W - 48 - int(tw) - 28, 22, W - 28, 50), 14, (240, 253, 244), (187, 247, 208))
    draw.ellipse((W - 48 - int(tw) - 16, 30, W - 48 - int(tw) - 6, 40), fill=GREEN)
    draw.text((W - 40 - int(tw), 28), pill, font=small, fill=(22, 101, 52))

    # Left copy
    draw.text((40, 110), "Options pricing.", font=bold, fill=TEXT)
    draw.text((40, 172), "Implied vol.", font=bold, fill=PRIMARY)
    draw.text((40, 234), "Vol forecasting.", font=bold, fill=TEXT)

    tagline = "European prices and Greeks, a market IV surface,\nand a walk-forward study that reports a negative H1."
    draw.multiline_text((40, 310), tagline, font=regular, fill=MUTED, spacing=6)

    badges = ["Black–Scholes", "CRR · MC", "HAR-RV · GARCH · XGB"]
    x = 40
    y = 400
    for label in badges:
        bw = int(draw.textlength(label, font=small)) + 24
        rounded_rect(draw, (x, y, x + bw, y + 32), 8, WHITE, BORDER)
        draw.text((x + 12, y + 7), label, font=small, fill=PRIMARY_DARK)
        x += bw + 10

    draw.text((40, 460), "H1: XGBoost beats HAR-RV  →  NOT SUPPORTED", font=small, fill=(185, 28, 28))

    draw.text((40, 582), "volatilitylab-hd1l.onrender.com", font=mono, fill=MUTED)
    draw.text((40, 604), "github.com/pabloapcv/volatility-lab", font=mono, fill=MUTED)

    # Right: IV surface crop in a card
    surface = Image.open(SHOTS / "volatility_surface_spy.png").convert("RGB")
    crop = crop_surface(surface)
    card_w, card_h = 620, 470
    fitted = crop.resize((card_w, card_h), Image.Resampling.LANCZOS)
    card = round_corners(fitted, 16)

    cx, cy = 620, 98
    shadow = drop_shadow((card_w, card_h), radius=16, offset=10)
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(shadow, (cx - 6, cy - 2))
    canvas_rgba.alpha_composite(card, (cx, cy))

    # White outline on the card
    overlay = ImageDraw.Draw(canvas_rgba)
    overlay.rounded_rectangle(
        (cx, cy, cx + card_w, cy + card_h),
        radius=16,
        outline=(*BORDER, 255),
        width=2,
    )

    canvas_rgba.convert("RGB").save(OUT, "PNG", optimize=True)
    saved = Image.open(OUT)
    if saved.size != (W, H):
        raise SystemExit(f"expected {W}x{H}, got {saved.size}")
    print(f"wrote {OUT} {saved.size}")


if __name__ == "__main__":
    main()
