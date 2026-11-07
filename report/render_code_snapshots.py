#!/usr/bin/env python3
"""Render high-DPI IDE-style syntax-highlighted PNG code captures from src/ files.

Strictly read-only on src/. Uses Pygments tokenization + Pillow rendering with
DejaVuSansMono (ASCII/code) and UnDotum (Korean comment fallback) so Hangul
comments render crisply without missing glyph boxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import (
    Comment,
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures" / "code"

MONO_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
KR_FONT = "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"

# Clean light-theme academic IDE palette (prints & reads sharply on white PDF pages)
BG_COLOR = (255, 255, 255)
GUTTER_BG = (246, 248, 250)
HEADER_BG = (30, 41, 59)
SUBHEADER_BG = (241, 245, 249)
BORDER_COLOR = (203, 213, 225)
DIVIDER_COLOR = (226, 232, 240)
ELLIPSIS_BG = (248, 250, 252)
LINE_NUM_COLOR = (100, 116, 139)
DEFAULT_TEXT = (30, 41, 59)


def token_style(ttype) -> Tuple[Tuple[int, int, int], bool]:
    """Return (RGB color, is_bold) for a Pygments token type."""
    if ttype in Comment or ttype in Comment.Single or ttype in Comment.Multiline:
        return (106, 115, 125), False
    if ttype in Keyword or ttype in Keyword.Namespace or ttype in Keyword.Declaration:
        return (207, 34, 46), True
    if ttype in Keyword.Constant or ttype in Keyword.Type:
        return (5, 80, 174), True
    if ttype in Name.Function or ttype in Name.Magic:
        return (130, 80, 223), True
    if ttype in Name.Class:
        return (149, 56, 0), True
    if ttype in Name.Decorator or ttype in Name.Builtin or ttype in Name.Builtin.Pseudo:
        return (5, 80, 174), False
    if ttype in String or ttype in Literal.String or str(ttype).startswith("Token.Literal.String"):
        return (10, 48, 105), False
    if ttype in Number or str(ttype).startswith("Token.Literal.Number"):
        return (5, 80, 174), False
    if ttype in Operator or ttype in Operator.Word:
        return (207, 34, 46), False
    if ttype in Punctuation:
        return (51, 65, 85), False
    return DEFAULT_TEXT, False


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xAC00 <= code <= 0xD7A3
        or 0x4E00 <= code <= 0x9FFF
    )


def split_runs_by_script(text: str) -> List[Tuple[str, bool]]:
    """Split text into contiguous (substring, is_cjk_flag) runs."""
    if not text:
        return []
    runs: List[Tuple[str, bool]] = []
    cur_flag = is_cjk(text[0])
    buf = [text[0]]
    for ch in text[1:]:
        f = is_cjk(ch)
        if f == cur_flag:
            buf.append(ch)
        else:
            runs.append(("".join(buf), cur_flag))
            buf = [ch]
            cur_flag = f
    runs.append(("".join(buf), cur_flag))
    return runs


def render_snippet_card(
    title: str,
    sections: Sequence[Tuple[str, Sequence[Tuple[int, int]]]],
    out_path: Path,
) -> None:
    """Render one multi-range or multi-file IDE snapshot card at high resolution."""
    font_size = 22
    header_font = ImageFont.truetype(MONO_BOLD, 22)
    subheader_font = ImageFont.truetype(MONO_BOLD, 19)
    mono_font = ImageFont.truetype(MONO_REG, font_size)
    mono_bold_font = ImageFont.truetype(MONO_BOLD, font_size)
    kr_font = ImageFont.truetype(KR_FONT, font_size)
    kr_header_font = ImageFont.truetype(KR_FONT, 21)

    line_h = 33
    gutter_w = 82
    pad_x = 20
    header_h = 54
    section_bar_h = 38

    lexer = PythonLexer(stripnl=False)

    # Prepare structured rows: ("section", label) | ("ellipsis", text) | ("code", lineno, tokens)
    rows = []
    max_px_width = 1280

    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)

    def measure_mixed(text: str, bold: bool = False, header: bool = False) -> float:
        total = 0.0
        for run, cjk in split_runs_by_script(text):
            if header:
                f = kr_header_font if cjk else header_font
            else:
                f = kr_font if cjk else (mono_bold_font if bold else mono_font)
            total += dummy_draw.textlength(run, font=f)
        return total

    for rel_path, line_ranges in sections:
        full_path = WORKSPACE_ROOT / rel_path
        src_lines = full_path.read_text(encoding="utf-8").splitlines()
        range_label = ", ".join(f"L{s}-L{e}" for s, e in line_ranges)
        rows.append(("section", f"{rel_path}  ({range_label})"))

        for idx, (start_line, end_line) in enumerate(line_ranges):
            if idx > 0:
                rows.append(("ellipsis", "⋮   (intermediate lines omitted for brevity)"))
            chunk = "\n".join(src_lines[start_line - 1 : end_line])
            # Tokenize chunk line by line while preserving Python multiline context
            cur_line_tokens: List[Tuple[Tuple[int, int, int], bool, str]] = []
            cur_lineno = start_line
            for ttype, val in lex(chunk, lexer):
                color, bold = token_style(ttype)
                parts = val.split("\n")
                for p_i, part in enumerate(parts):
                    if part:
                        cur_line_tokens.append((color, bold, part.expandtabs(4)))
                    if p_i < len(parts) - 1:
                        if cur_lineno <= end_line:
                            rows.append(("code", cur_lineno, cur_line_tokens))
                            cur_lineno += 1
                        cur_line_tokens = []
            if cur_lineno <= end_line:
                rows.append(("code", cur_lineno, cur_line_tokens))

    # Measure required canvas width & height
    total_h = header_h + 14
    for r in rows:
        if r[0] == "section":
            total_h += section_bar_h
        else:
            total_h += line_h
            if r[0] == "code":
                w = gutter_w + pad_x * 2 + sum(
                    measure_mixed(txt, bold=b) for _, b, txt in r[2]
                )
                if w + 24 > max_px_width:
                    max_px_width = int(w + 24)
    total_h += 14

    title_w = int(measure_mixed(title, bold=True, header=True)) + 180
    img_w = max(max_px_width, title_w)
    img = Image.new("RGB", (img_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw dark IDE title bar
    draw.rectangle([0, 0, img_w, header_h], fill=HEADER_BG)
    # macOS window traffic lights
    dot_colors = [(239, 68, 68), (245, 158, 11), (34, 197, 94)]
    for i, dc in enumerate(dot_colors):
        cx = 26 + i * 24
        cy = header_h // 2
        draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=dc)

    # Draw header title (mixed ASCII + Korean support)
    hx = 104.0
    hy = (header_h - 24) // 2
    for run, cjk in split_runs_by_script(title):
        f = kr_header_font if cjk else header_font
        draw.text((hx, hy), run, fill=(241, 245, 249), font=f)
        hx += draw.textlength(run, font=f)

    y = header_h
    for r in rows:
        kind = r[0]
        if kind == "section":
            draw.rectangle([0, y, img_w, y + section_bar_h], fill=SUBHEADER_BG)
            draw.line([0, y, img_w, y], fill=BORDER_COLOR, width=1)
            draw.line([0, y + section_bar_h - 1, img_w, y + section_bar_h - 1], fill=BORDER_COLOR, width=1)
            draw.text((18, y + 8), f"📄  {r[1]}", fill=(15, 23, 42), font=subheader_font)
            y += section_bar_h
        elif kind == "ellipsis":
            draw.rectangle([0, y, img_w, y + line_h], fill=ELLIPSIS_BG)
            draw.rectangle([0, y, gutter_w, y + line_h], fill=GUTTER_BG)
            draw.line([gutter_w, y, gutter_w, y + line_h], fill=DIVIDER_COLOR, width=1)
            draw.text((gutter_w + pad_x, y + 5), r[1], fill=(100, 116, 139), font=mono_font)
            y += line_h
        elif kind == "code":
            lineno, tokens = r[1], r[2]
            draw.rectangle([0, y, gutter_w, y + line_h], fill=GUTTER_BG)
            draw.line([gutter_w, y, gutter_w, y + line_h], fill=DIVIDER_COLOR, width=1)
            ln_str = f"{lineno:>4}"
            ln_w = draw.textlength(ln_str, font=mono_font)
            draw.text((gutter_w - ln_w - 14, y + 5), ln_str, fill=LINE_NUM_COLOR, font=mono_font)

            tx = float(gutter_w + pad_x)
            ty = y + 5
            for color, bold, txt in tokens:
                for run, cjk in split_runs_by_script(txt):
                    f = kr_font if cjk else (mono_bold_font if bold else mono_font)
                    draw.text((tx, ty), run, fill=color, font=f)
                    tx += draw.textlength(run, font=f)
            y += line_h

    # Outer border
    draw.rectangle([0, 0, img_w - 1, total_h - 1], outline=BORDER_COLOR, width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))
    print(f"[+] Rendered {out_path.relative_to(WORKSPACE_ROOT)} ({img_w}x{total_h}px)")


SNIPPETS = [
    (
        "1. Raw DICOM Loading, [-100, 300] HU Windowing & Subject-Disjoint Split",
        [
            ("src/data.py", [(19, 41), (50, 104)]),
        ],
        "code_01_data_pipeline.png",
    ),
    (
        "2. MONAI Paired Data Augmentation Registry (Single & Composite Transforms)",
        [
            ("src/utils.py", [(20, 36), (45, 61), (72, 103)]),
        ],
        "code_02_augment_registry.png",
    ),
    (
        "3. U-Net Without Global Residual (U_Net_NoRes) vs. Global +x (U_Net) & R2U-Net",
        [
            ("src/UNet/Models.py", [(5, 25), (47, 54), (115, 124), (194, 203), (205, 247)]),
        ],
        "code_03_unet_residual.png",
    ),
    (
        "4. Attention Gate (AttU-Net / R2AttU-Net) & Dense Nested Skips (UNet++)",
        [
            ("src/UNet/Models.py", [(333, 365), (626, 648)]),
        ],
        "code_04_att_nested.png",
    ),
    (
        "5. Custom PSNR / Gaussian-Window SSIM Metrics & Composite Loss Function",
        [
            ("src/loss.py", [(41, 51), (119, 161)]),
            ("src/main.py", [(78, 88)]),
        ],
        "code_05_loss_metrics.png",
    ),
    (
        "6. MONAI SupervisedTrainer (AMP, Grad Clipping, Early Stopping & Test Eval)",
        [
            ("src/trainer.py", [(96, 111), (166, 231), (383, 407)]),
        ],
        "code_06_trainer_engine.png",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-render even if PNGs exist")
    args = parser.parse_args()

    for title, sections, fname in SNIPPETS:
        out_path = OUT_DIR / fname
        if out_path.exists() and not args.force:
            print(f"[=] Skipping existing {out_path.relative_to(WORKSPACE_ROOT)}")
            continue
        render_snippet_card(title, sections, out_path)


if __name__ == "__main__":
    main()
