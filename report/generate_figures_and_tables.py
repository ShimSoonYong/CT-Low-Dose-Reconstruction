#!/usr/bin/env python3
"""Generate 300-DPI figures (in [-100, 300] HU) and LaTeX booktabs tables from eval_cache.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
REPORT_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPORT_DIR / "cache"
FIG_DIR = REPORT_DIR / "figures"
GEN_DIR = REPORT_DIR / "generated"

KR_FONT_PATH = "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"
if Path(KR_FONT_PATH).exists():
    fm.fontManager.addfont(KR_FONT_PATH)
    plt.rcParams["font.family"] = ["DejaVu Sans", "UnDotum"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300


def norm_to_hu(arr: np.ndarray) -> np.ndarray:
    """Inverse-map [0, 1] normalized CT intensity back to [-100, 300] HU."""
    return arr * 400.0 - 100.0


def find_high_detail_roi(full_2d: np.ndarray, q_2d: np.ndarray, box_size: int = 128) -> Tuple[int, int]:
    """Automatically locate an anatomical ROI inside soft-tissue parenchyma with rich vessel/organ edges."""
    h, w = full_2d.shape
    gy, gx = np.gradient(full_2d)
    grad = np.hypot(gx, gy)
    # Prefer interior soft-tissue + contrast vessel regions (avoid empty air background and table)
    soft_mask = ((full_2d > 0.25) & (full_2d < 0.92)).astype(np.float32)
    score_map = grad * soft_mask

    best_score = -1.0
    best_yx = (h // 2 - box_size // 2, w // 2 - box_size // 2)
    margin_y = int(h * 0.20)
    margin_x = int(w * 0.20)
    for y in range(margin_y, h - margin_y - box_size, 16):
        for x in range(margin_x, w - margin_x - box_size, 16):
            patch_soft = soft_mask[y : y + box_size, x : x + box_size].mean()
            if patch_soft < 0.55:
                continue
            sc = float(score_map[y : y + box_size, x : x + box_size].mean())
            if sc > best_score:
                best_score = sc
                best_yx = (y, x)
    return best_yx


def display_run_label(run_name: str, eval_entry: Dict[str, Any], meta: Dict[str, Any]) -> str:
    model_map = {
        "U_Net": "U-Net (+x)",
        "U_Net_NoRes": "U-Net (NoRes)",
        "R2U_Net": "R2U-Net",
        "AttU_Net": "AttU-Net",
        "R2AttU_Net": "R2AttU-Net",
        "NestedUNet": "UNet++",
    }
    aug_map = {
        "none": "No Aug",
        "Rot90_Flip_Combo": "Rot90+Flip",
        "RandRot90d": "RandRot90",
        "RandAffined": "RandAffined",
        "Affine_Noise_Combo": "Affine+Noise",
        "Rot90_Scale_Combo": "Rot90+Scale",
    }
    m = model_map.get(meta["model"], meta["model"])
    a = aug_map.get(meta["augment"], meta["augment"])
    mode = eval_entry.get("selection_mode", "")
    ep = eval_entry.get("selected_epoch", "?")
    if mode == "recovered_pre_nan":
        return f"{m} ({a}, Ep.{ep} Pre-NaN*)"
    if not meta.get("tb", {}).get("has_logged_test", False) and meta["augment"] == "none":
        return f"{m} ({a}, Ep.{ep} Snap.)"
    return f"{m} ({a}, Ep.{ep})"


def plot_3_visual_examples(
    cache: Dict[str, Any], npz: Any, best_run_name: str
) -> None:
    """Render 3 anatomically distinct test cases in [-100, 300] HU with ROI zooms & HU error maps."""
    picks = cache["representative_picks"][:3]
    q_all = npz["quarter"]
    f_all = npz["full"]
    p_all = npz[f"pred__{best_run_name}"]

    base_slices = {r["idx"]: r for r in cache["baseline"]["per_slice"]}
    best_slices = {
        r["idx"]: r for r in cache["evaluated_runs"][best_run_name]["per_slice"]
    }
    best_meta = cache["runs_metadata"][best_run_name]
    best_short = display_run_label(
        best_run_name, cache["evaluated_runs"][best_run_name], best_meta
    )

    fig = plt.figure(figsize=(16.5, 11.6))
    gs = fig.add_gridspec(
        3,
        5,
        width_ratios=[1.0, 1.0, 1.0, 0.98, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.09,
        hspace=0.25,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.11,
    )

    box_sz = 120
    im_hu = None
    im_err = None

    for row_i, pick in enumerate(picks):
        s_idx = pick["idx"]
        q_hu = norm_to_hu(q_all[row_i])
        p_hu = norm_to_hu(p_all[row_i])
        f_hu = norm_to_hu(f_all[row_i])

        ry, rx = find_high_detail_roi(f_all[row_i], q_all[row_i], box_size=box_sz)
        bm = base_slices[s_idx]
        pm = best_slices[s_idx]

        # Extract concise slice number from raw DICOM .IMA filename (e.g. .CT.0001.0061.)
        parts_fn = pick["filename"].split(".")
        slice_tag = f"Slice #{parts_fn[3]}" if len(parts_fn) > 4 else pick["filename"][:18]

        panels = [
            (q_hu, f"Input: Quarter Dose\nPSNR {bm['psnr']:.2f} dB | SSIM {bm['ssim']:.3f} | RMSE {bm['rmse_hu']:.1f} HU"),
            (p_hu, f"Output: {best_short}\nPSNR {pm['psnr']:.2f} dB | SSIM {pm['ssim']:.3f} | RMSE {pm['rmse_hu']:.1f} HU"),
            (f_hu, f"Label: Full Dose (Ground Truth)\nSubject {pick['subject']} ({slice_tag}, 3mm Sharp .IMA)"),
        ]

        for col_i, (img_hu, subtitle) in enumerate(panels):
            ax = fig.add_subplot(gs[row_i, col_i])
            im_hu = ax.imshow(img_hu, cmap="gray", vmin=-100.0, vmax=300.0, interpolation="nearest")
            rect = patches.Rectangle(
                (rx, ry), box_sz, box_sz, linewidth=1.6, edgecolor="#EF4444", facecolor="none"
            )
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                col_titles = [
                    "① Input (Quarter Dose)",
                    "② Reconstructed Output",
                    "③ Ground Truth (Full Dose)",
                ]
                ax.set_title(col_titles[col_i], fontsize=11, fontweight="bold", pad=18)
            ax.text(
                0.5,
                -0.035,
                subtitle,
                transform=ax.transAxes,
                fontsize=8.0,
                ha="center",
                va="top",
                color="#1E293B",
            )
            if col_i == 0:
                ax.set_ylabel(
                    f"{pick['case_id']} ({pick['subject']})\n{pick['region_ko'].split('(')[0].strip()}",
                    fontsize=9.5,
                    fontweight="bold",
                    labelpad=8,
                )

        # Column 4: 1x3 or 2x2 Zoomed ROI Comparison inside [-100, 300] HU
        ax_z = fig.add_subplot(gs[row_i, 3])
        z_q = q_hu[ry : ry + box_sz, rx : rx + box_sz]
        z_p = p_hu[ry : ry + box_sz, rx : rx + box_sz]
        z_f = f_hu[ry : ry + box_sz, rx : rx + box_sz]
        sep = np.full((box_sz, 3), 300.0, dtype=np.float32)
        strip = np.concatenate([z_q, sep, z_p, sep, z_f], axis=1)
        # Stack zoomed CT strip (top) and zoomed |Error| strip (bottom) for crystal clarity
        ez_q = np.abs(z_q - z_f)
        ez_p = np.abs(z_p - z_f)
        # Render ROI zoomed comparison cleanly as 2 rows inside a nested gridspec
        ax_z.axis("off")
        sub_gs = gs[row_i, 3].subgridspec(2, 3, wspace=0.04, hspace=0.14)
        for zi, (z_patch, z_lbl) in enumerate([(z_q, "In"), (z_p, "Out"), (z_f, "GT")]):
            s_ax = fig.add_subplot(sub_gs[0, zi])
            s_ax.imshow(z_patch, cmap="gray", vmin=-100.0, vmax=300.0, interpolation="nearest")
            s_ax.set_xticks([])
            s_ax.set_yticks([])
            for sp in s_ax.spines.values():
                sp.set_edgecolor("#EF4444")
                sp.set_linewidth(1.1)
            s_ax.set_title(z_lbl, fontsize=8, pad=2, fontweight="bold")
        for zi, (e_patch, e_lbl) in enumerate(
            [(ez_q, "|In-GT|"), (ez_p, "|Out-GT|"), (np.clip(ez_q - ez_p, 0, 80), "Noise ↓")]
        ):
            s_ax = fig.add_subplot(sub_gs[1, zi])
            s_ax.imshow(e_patch, cmap="inferno", vmin=0.0, vmax=70.0, interpolation="nearest")
            s_ax.set_xticks([])
            s_ax.set_yticks([])
            s_ax.set_title(e_lbl, fontsize=7.5, pad=2)
        if row_i == 0:
            ax_z.set_title("④ Red-Box ROI Zoom\n(Top: HU, Bot: Error)", fontsize=10.0, fontweight="bold", pad=8)

        # Column 5: Full-slice Absolute Error Map |Output - Full Dose| in HU (vs Quarter Dose inset)
        ax_e = fig.add_subplot(gs[row_i, 4])
        err_out = np.abs(p_hu - f_hu)
        err_in = np.abs(q_hu - f_hu)
        im_err = ax_e.imshow(err_out, cmap="inferno", vmin=0.0, vmax=70.0, interpolation="nearest")
        ax_e.set_xticks([])
        ax_e.set_yticks([])
        if row_i == 0:
            ax_e.set_title("⑤ Residual Error |Out - GT|", fontsize=11, fontweight="bold", pad=18)
        ax_e.text(
            0.5,
            -0.035,
            f"MAE: {bm['mae_hu']:.1f} → {pm['mae_hu']:.1f} HU ({(1 - pm['mae_hu']/bm['mae_hu'])*100:.1f}% ↓)\n"
            f"ROI Noise σ: {bm['roi_noise_hu']:.1f} → {pm['roi_noise_hu']:.1f} HU",
            transform=ax_e.transAxes,
            fontsize=8.0,
            ha="center",
            va="top",
            color="#1E293B",
        )

    fig.suptitle(
        "Held-Out Test Set (L192, L506) Reconstruction Examples Displayed in [-100, 300] HU (3mm Sharp Kernel)",
        fontsize=13.5,
        fontweight="bold",
        y=0.978,
    )

    # Bottom dual colorbars explicitly showing [-100, 300] HU and [0, 70] HU error scale
    cax1 = fig.add_axes([0.07, 0.048, 0.42, 0.016])
    cb1 = fig.colorbar(im_hu, cax=cax1, orientation="horizontal")
    cb1.set_label("CT Display Window Intensity [Hounsfield Units, -100 to 300 HU]", fontsize=9.5, fontweight="bold")
    cb1.ax.tick_params(labelsize=8.5)

    cax2 = fig.add_axes([0.57, 0.048, 0.36, 0.016])
    cb2 = fig.colorbar(im_err, cax=cax2, orientation="horizontal")
    cb2.set_label("Absolute Reconstruction Error |Pred - Full Dose| [0 to 70 HU]", fontsize=9.5, fontweight="bold")
    cb2.ax.tick_params(labelsize=8.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_p = FIG_DIR / "fig_visual_examples_3slices.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"[+] Saved {out_p.relative_to(WORKSPACE_ROOT)}")


def plot_residual_comparison(cache: Dict[str, Any], npz: Any) -> None:
    """Compare Quarter Dose, Non-Residual-Block U-Net, Intra-Block Residual R2U-Net (and others), and Full Dose."""
    pick_idx = 3 if len(cache["representative_picks"]) >= 4 else 0
    pick = cache["representative_picks"][pick_idx]
    s_idx = pick["idx"]

    q_hu = norm_to_hu(npz["quarter"][pick_idx])
    f_hu = norm_to_hu(npz["full"][pick_idx])
    ry, rx = find_high_detail_roi(npz["full"][pick_idx], npz["quarter"][pick_idx], box_size=128)

    # Gather available baseline ("none" aug) models prioritizing U_Net (+x) vs U_Net_NoRes vs AttU_Net vs NestedUNet / R2U_Net
    eval_runs = cache["evaluated_runs"]
    candidates = []
    for m in ["U_Net", "U_Net_NoRes", "AttU_Net", "NestedUNet", "R2U_Net", "R2AttU_Net"]:
        rn = f"{m}_none"
        if rn in eval_runs and f"pred__{rn}" in npz:
            candidates.append(rn)
    if len(candidates) < 4 and "U_Net_RandAffined" in eval_runs and "pred__U_Net_RandAffined" in npz:
        candidates.append("U_Net_RandAffined")

    cols = [("Quarter Dose\n(Noisy Input)", q_hu, cache["baseline"]["per_slice"])]
    for rn in candidates[:4]:
        meta = cache["runs_metadata"][rn]
        lbl = display_run_label(rn, eval_runs[rn], meta)
        cols.append((lbl, norm_to_hu(npz[f"pred__{rn}"][pick_idx]), eval_runs[rn]["per_slice"]))
    cols.append(("Full Dose\n(Ground Truth)", f_hu, None))

    ncols = len(cols)
    fig = plt.figure(figsize=(3.3 * ncols, 8.6))
    gs = fig.add_gridspec(
        3,
        ncols,
        height_ratios=[1.15, 0.95, 0.85],
        wspace=0.10,
        hspace=0.28,
        left=0.05,
        right=0.98,
        top=0.91,
        bottom=0.07,
    )

    # Row 1: Zoomed 128x128 ROI in [-100, 300] HU with horizontal profile scanline
    scan_row = 64
    for ci, (title, arr_hu, slice_list) in enumerate(cols):
        ax = fig.add_subplot(gs[0, ci])
        roi = arr_hu[ry : ry + 128, rx : rx + 128]
        ax.imshow(roi, cmap="gray", vmin=-100.0, vmax=300.0, interpolation="nearest")
        ax.axhline(scan_row, color="#38BDF8", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        if slice_list is not None:
            rec = next(r for r in slice_list if r["idx"] == s_idx)
            ax.set_xlabel(
                f"PSNR {rec['psnr']:.2f} dB | SSIM {rec['ssim']:.3f}\nEPI {rec['epi']:.3f} | RMSE {rec['rmse_hu']:.1f} HU",
                fontsize=8.0,
                labelpad=4,
            )
        else:
            ax.set_xlabel(f"Target ({pick['subject']}, [-100, 300] HU)", fontsize=8.0, labelpad=4)
        if ci == 0:
            ax.set_ylabel("Zoomed ROI\n[-100, 300] HU", fontsize=9.5, fontweight="bold")

        # Row 2: Zoomed Absolute Error Map |Pred - Full Dose| (or Full-slice overview on GT column)
        ax2 = fig.add_subplot(gs[1, ci])
        if slice_list is not None:
            err_roi = np.abs(roi - f_hu[ry : ry + 128, rx : rx + 128])
            im_e = ax2.imshow(err_roi, cmap="inferno", vmin=0.0, vmax=65.0, interpolation="nearest")
            ax2.set_xlabel(f"ROI Mean |Error|: {err_roi.mean():.1f} HU", fontsize=8.0, labelpad=4)
        else:
            ax2.imshow(f_hu, cmap="gray", vmin=-100.0, vmax=300.0, interpolation="nearest")
            rect = patches.Rectangle((rx, ry), 128, 128, linewidth=1.6, edgecolor="#EF4444", facecolor="none")
            ax2.add_patch(rect)
            ax2.set_xlabel("Full 512×512 Slice & ROI Box", fontsize=8.0, labelpad=4)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if ci == 0:
            ax2.set_ylabel("ROI |Pred - GT|\n[0, 65] HU", fontsize=9.5, fontweight="bold")

    # Row 3: 1D HU Intensity Line Profile across the cyan dashed line (demonstrates noise vs edge sharpness)
    ax_prof = fig.add_subplot(gs[2, :])
    x_pix = np.arange(128)
    gt_line = f_hu[ry + scan_row, rx : rx + 128]
    q_line = q_hu[ry + scan_row, rx : rx + 128]
    ax_prof.plot(x_pix, q_line, color="#94A3B8", lw=1.1, alpha=0.75, label="Quarter Dose (Input)")
    ax_prof.plot(x_pix, gt_line, color="#0F172A", lw=2.2, label="Full Dose (Ground Truth)")

    palette = ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]
    for pi, rn in enumerate(candidates[:4]):
        meta = cache["runs_metadata"][rn]
        lbl = display_run_label(rn, eval_runs[rn], meta)
        pr_line = norm_to_hu(npz[f"pred__{rn}"][pick_idx])[ry + scan_row, rx : rx + 128]
        ax_prof.plot(x_pix, pr_line, color=palette[pi % len(palette)], lw=1.7, label=lbl)

    ax_prof.set_xlim(0, 127)
    ax_prof.set_ylim(-105, 305)
    ax_prof.set_xlabel("Horizontal Pixel Coordinate Along Dashed Cyan Scanline in ROI [px]", fontsize=9.5)
    ax_prof.set_ylabel("Attenuation [HU]", fontsize=9.5, fontweight="bold")
    ax_prof.set_title(
        "1D Hounsfield Unit (HU) Profile Along Anatomical Boundary: Quantum Mottle Suppression vs. Edge Slope Preservation",
        fontsize=10.0,
        fontweight="bold",
    )
    ax_prof.grid(True, linestyle=":", alpha=0.5)
    ax_prof.legend(loc="upper right", ncol=min(5, ncols), fontsize=8.0, frameon=True)

    fig.suptitle(
        f"Comparative Analysis of Non-Residual-Block vs. Intra-Block Residual U-Net Variants ({pick['subject']} Test Slice)",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )
    out_p = FIG_DIR / "fig_residual_comparison.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"[+] Saved {out_p.relative_to(WORKSPACE_ROOT)}")


def plot_training_and_nan_diagnostics(cache: Dict[str, Any]) -> None:
    """Plot training loss curves (highlighting AMP NaN onset steps) and validation PSNR/SSIM progression."""
    runs_meta = cache["runs_metadata"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax_loss, ax_val = axes

    colors = {
        "U_Net_none": "#2563EB",
        "U_Net_NoRes_none": "#DC2626",
        "U_Net_Rot90_Flip_Combo": "#0284C7",
        "U_Net_RandAffined": "#F59E0B",
        "U_Net_Affine_Noise_Combo": "#EF4444",
        "U_Net_Rot90_Scale_Combo": "#EC4899",
        "R2U_Net_none": "#10B981",
        "AttU_Net_none": "#8B5CF6",
        "AttU_Net_Rot90_Flip_Combo": "#16A34A",
        "R2AttU_Net_none": "#06B6D4",
        "NestedUNet_none": "#F97316",
    }

    base_steps_per_epoch = max(1, cache["dataset_info"]["train_slices"] // 4)

    # Panel A: Early-to-mid training loss (first 15 epochs / ~6000 steps inset + full trajectory)
    for rn, meta in sorted(runs_meta.items()):
        tb = meta.get("tb", {})
        tl = tb.get("scalars", {}).get("train_loss", {})
        steps = tl.get("steps", [])
        vals = tl.get("values", [])
        steps_per_ep = (
            max(1, cache["dataset_info"]["train_slices"])
            if rn.startswith("NestedUNet")
            else base_steps_per_epoch
        )
        finite = [(s / steps_per_ep, v) for s, v in zip(steps, vals) if v is not None]
        if not finite:
            continue
        xs, ys = zip(*finite)
        # Smooth slightly if dense
        c = colors.get(rn, None)
        lbl = rn.replace("_", "-")
        ax_loss.plot(xs, ys, lw=1.5, alpha=0.88, color=c, label=lbl)

        if tb.get("first_nan_step") is not None:
            nan_ep = tb["first_nan_step"] / steps_per_ep
            ax_loss.scatter(
                [xs[-1]],
                [ys[-1]],
                color=c or "red",
                marker="X",
                s=65,
                zorder=6,
            )
            ax_loss.annotate(
                f"NaN @ Ep.{nan_ep:.1f}",
                xy=(xs[-1], ys[-1]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                color=c or "#B91C1C",
            )

    ax_loss.set_xlabel("Training Epoch (Iteration / 386)", fontsize=10)
    ax_loss.set_ylabel("Composite Training Loss (L1 + 0.1·SSIMLoss)", fontsize=10)
    ax_loss.set_title("(a) Training Loss Trajectories & FP16 AMP NaN Onset Points", fontsize=10.5, fontweight="bold")
    ax_loss.set_ylim(0.025, 0.22)
    ax_loss.grid(True, linestyle=":", alpha=0.5)
    ax_loss.legend(fontsize=8.0, loc="upper right")

    # Panel B: Validation PSNR progression across validated runs (+ baseline reference line)
    base_psnr = cache["baseline"]["summary"]["overall"]["psnr_mean"]
    ax_val.axhline(
        base_psnr,
        color="#64748B",
        linestyle="--",
        lw=1.3,
        label=f"Quarter-Dose Test Baseline ({base_psnr:.2f} dB)",
    )

    for rn, meta in sorted(runs_meta.items()):
        tb = meta.get("tb", {})
        vp = tb.get("scalars", {}).get("val_psnr", {})
        steps = vp.get("steps", [])
        vals = vp.get("values", [])
        pts = [
            ((i + 1) * 10, v)
            for i, v in enumerate(vals)
            if v is not None and v > 1.0
        ]
        if not pts:
            continue
        exs, vys = zip(*pts)
        c = colors.get(rn, None)
        ax_val.plot(
            exs,
            vys,
            marker="o",
            markersize=5,
            lw=1.8,
            color=c,
            label=f"{rn.replace('_', '-')} (Val PSNR)",
        )
        best_i = int(np.argmax(vys))
        ax_val.scatter([exs[best_i]], [vys[best_i]], s=95, facecolors="none", edgecolors=c or "black", lw=2.0, zorder=7)

    # Also plot recovered U_Net_RandAffined test PSNR marker if present
    if "U_Net_RandAffined" in cache["evaluated_runs"]:
        rec_e = cache["evaluated_runs"]["U_Net_RandAffined"]
        if rec_e["selection_mode"] == "recovered_pre_nan":
            ax_val.scatter(
                [rec_e["selected_epoch"]],
                [rec_e["summary"]["overall"]["psnr_mean"]],
                marker="D",
                s=70,
                color="#F59E0B",
                label=f"U-Net-RandAffined (Recovered Ep.{rec_e['selected_epoch']} Test PSNR)",
            )

    ax_val.set_xlabel("Epoch (Validation Executed Every 10 Epochs)", fontsize=10)
    ax_val.set_ylabel("PSNR [dB]", fontsize=10)
    ax_val.set_title("(b) Validation PSNR Progression & Selected Checkpoints", fontsize=10.5, fontweight="bold")
    ax_val.grid(True, linestyle=":", alpha=0.5)
    ax_val.legend(fontsize=8.0, loc="lower right")

    fig.tight_layout()
    out_p = FIG_DIR / "fig_training_and_nan_diagnostics.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"[+] Saved {out_p.relative_to(WORKSPACE_ROOT)}")


def plot_augmentation_pitfalls(npz: Any) -> None:
    """Demonstrate mathematically & visually why paired intensity/noise augmentations degraded CT restoration."""
    import torch
    from monai.transforms import RandAffined, RandGaussianNoised, RandRotate90d, RandScaleIntensityd

    q_orig = torch.tensor(npz["quarter"][0:1], dtype=torch.float32)  # [1, 512, 512]
    f_orig = torch.tensor(npz["full"][0:1], dtype=torch.float32)

    aug_configs = [
        ("none", "Baseline / Rot90_Flip_Combo\nExact [0, 1] Clean Target", []),
        (
            "RandAffined",
            "RandAffined\nBilinear Smoothing of Sharp Noise",
            [RandAffined(keys=["quarter", "full"], rotate_range=(0.15, 0.15), prob=1.0, mode="bilinear", padding_mode="reflection")],
        ),
        (
            "Affine_Noise_Combo",
            "Affine_Noise_Combo (Pre-Fix)\nTarget Corrupted + Out-of-[0,1]",
            [
                RandAffined(keys=["quarter", "full"], rotate_range=(0.12, 0.12), prob=1.0, mode="bilinear"),
                RandGaussianNoised(keys=["quarter", "full"], mean=0.0, std=0.05, prob=1.0),
            ],
        ),
        (
            "Rot90_Scale_Combo",
            "Rot90_Scale_Combo\nIntensity > 1.0 vs. clamp(0,1)",
            [
                RandRotate90d(keys=["quarter", "full"], prob=1.0, max_k=1),
                RandScaleIntensityd(keys=["quarter", "full"], factors=(0.10, 0.10), prob=1.0),
            ],
        ),
    ]

    samples = []
    for key, title, tfms in aug_configs:
        if not tfms:
            samples.append((key, title, q_orig[0].numpy(), f_orig[0].numpy()))
        else:
            d = {"quarter": q_orig.clone(), "full": f_orig.clone()}
            for t in tfms:
                t.set_random_state(seed=42)
                d = t(d)
            qn = np.asarray(d["quarter"][0], dtype=np.float32)
            fn = np.asarray(d["full"][0], dtype=np.float32)
            samples.append((key, title, qn, fn))

    fig = plt.figure(figsize=(14.5, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.9], wspace=0.22, hspace=0.36, left=0.05, right=0.98, top=0.89, bottom=0.10)

    for ci, (key, title, q_arr, f_arr) in enumerate(samples):
        ax_img = fig.add_subplot(gs[0, ci])
        ax_img.imshow(f_arr[140:340, 160:360], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        # Highlight out-of-range (<0 or >1) pixels in red/cyan overlay
        over = f_arr[140:340, 160:360] > 1.0
        under = f_arr[140:340, 160:360] < 0.0
        if over.any() or under.any():
            overlay = np.zeros((200, 200, 4), dtype=np.float32)
            overlay[over] = [1.0, 0.15, 0.15, 0.65]
            overlay[under] = [0.1, 0.75, 1.0, 0.55]
            ax_img.imshow(overlay, interpolation="nearest")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(title, fontsize=9.5, fontweight="bold")
        pct_out = ((f_arr < 0.0) | (f_arr > 1.0)).mean() * 100.0
        ax_img.set_xlabel(
            f"Target range: [{f_arr.min():.2f}, {f_arr.max():.2f}] (Out-of-[0,1]: {pct_out:.1f}%)",
            fontsize=8.0,
        )
        if ci == 0:
            ax_img.set_ylabel("Augmented Target y\n(Zoomed Patch)", fontsize=9.5, fontweight="bold")

        ax_h = fig.add_subplot(gs[1, ci])
        body_vals = f_arr[f_arr > -0.2].ravel()
        ax_h.hist(body_vals, bins=60, range=(-0.15, 1.18), color="#3B82F6", alpha=0.75, density=True)
        ax_h.axvline(0.0, color="#DC2626", linestyle="--", lw=1.2)
        ax_h.axvline(1.0, color="#DC2626", linestyle="--", lw=1.2, label="Network clamp [0, 1]")
        ax_h.axvspan(-0.15, 0.0, color="#FEE2E2", alpha=0.55)
        ax_h.axvspan(1.0, 1.18, color="#FEE2E2", alpha=0.55)
        ax_h.set_xlim(-0.15, 1.18)
        ax_h.set_xlabel("Normalized Intensity y", fontsize=8.5)
        if ci == 0:
            ax_h.set_ylabel("Pixel Density", fontsize=9.5, fontweight="bold")
        ax_h.tick_params(labelsize=8.0)
        if ci == 2:
            ax_h.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Source-of-Error Analysis: Why Paired Intensity/Noise Augmentations Trigger Target Corruption & FP16 SSIMLoss Instability",
        fontsize=11.8,
        fontweight="bold",
        y=0.97,
    )
    out_p = FIG_DIR / "fig_augmentation_pitfalls.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"[+] Saved {out_p.relative_to(WORKSPACE_ROOT)}")


def plot_quantitative_distributions(cache: Dict[str, Any]) -> None:
    """Plot per-subject (L192 vs L506) PSNR/SSIM bars and slice-wise RMSE(HU) vs ROI Noise/EPI distributions."""
    eval_runs = cache["evaluated_runs"]
    runs_meta = cache["runs_metadata"]

    entries: List[Tuple[str, Dict[str, Any]]] = [
        ("Quarter Dose\n(Input)", cache["baseline"])
    ]
    for rn in sorted(
        eval_runs.keys(),
        key=lambda k: eval_runs[k]["summary"]["overall"]["psnr_mean"],
    ):
        lbl = display_run_label(rn, eval_runs[rn], runs_meta[rn]).replace(" (", "\n(")
        entries.append((lbl, eval_runs[rn]))

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.4))
    ax1, ax2 = axes

    x = np.arange(len(entries))
    w = 0.26
    for gi, (gkey, glbl, col) in enumerate(
        [
            ("L192", "Subject L192", "#60A5FA"),
            ("L506", "Subject L506", "#34D399"),
            ("overall", "Overall Test Set", "#1E3A8A"),
        ]
    ):
        means = [e[1]["summary"][gkey]["psnr_mean"] for e in entries]
        stds = [e[1]["summary"][gkey]["psnr_std"] for e in entries]
        bars = ax1.bar(
            x + (gi - 1) * w,
            means,
            width=w,
            yerr=stds,
            capsize=2.5,
            color=col,
            alpha=0.9,
            label=glbl,
        )
        if gkey == "overall":
            for b, m in zip(bars, means):
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    m + 0.35,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold",
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([e[0].replace("\n", " ") for e in entries], fontsize=7.6, rotation=22, ha="right")
    ax1.set_ylabel("Test PSNR [dB] (Mean ± Std)", fontsize=10)
    ax1.set_ylim(18.0, max(32.5, max(e[1]["summary"]["overall"]["psnr_mean"] for e in entries) + 3.5))
    ax1.set_title("(a) Cross-Patient Test PSNR Comparison (L192 vs. L506)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8.2, loc="upper left")

    # Panel B: Violin/Box distribution of slice-wise RMSE (HU) & Edge Preservation annotation
    data_rmse = [[r["rmse_hu"] for r in e[1]["per_slice"]] for e in entries]
    parts = ax2.violinplot(data_rmse, positions=x, showmeans=True, showmedians=True, widths=0.58)
    for pc in parts["bodies"]:
        pc.set_facecolor("#3B82F6")
        pc.set_edgecolor("#1E3A8A")
        pc.set_alpha(0.55)

    for xi, (_, entry) in enumerate(entries):
        ov = entry["summary"]["overall"]
        y_off = 0.8 + (2.6 if (xi % 2 == 1 and xi >= 3) else 0.0)
        ax2.text(
            xi,
            max(data_rmse[xi]) + y_off,
            f"{ov['rmse_hu_mean']:.1f}HU\nS:{ov['ssim_mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([e[0].replace("\n", " ") for e in entries], fontsize=7.6, rotation=22, ha="right")
    ax2.set_ylabel("Slice-wise RMSE in [-100, 300] Window [HU] (Lower is Better)", fontsize=10)
    ax2.set_ylim(0.0, max(max(d) for d in data_rmse) + 10.5)
    ax2.set_title("(b) Slice-wise HU Error Distribution, SSIM & Edge Preservation (EPI)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_p = FIG_DIR / "fig_quantitative_distributions.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"[+] Saved {out_p.relative_to(WORKSPACE_ROOT)}")


def tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write_latex_tables_and_macros(cache: Dict[str, Any], best_run_name: str) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    ds = cache["dataset_info"]
    specs = cache["arch_specs"]
    eval_runs = cache["evaluated_runs"]
    runs_meta = cache["runs_metadata"]
    base_ov = cache["baseline"]["summary"]["overall"]
    best_ov = eval_runs[best_run_name]["summary"]["overall"]
    best_meta = runs_meta[best_run_name]

    # 1. Dataset Summary Table
    subj_rows = []
    for subj, cnt in sorted(ds["subjects"].items()):
        is_test = subj in ("L192", "L506")
        role = "\\textbf{Held-out Test (평가 전용)}" if is_test else "Train / Val Pool (80:20 분할)"
        subj_rows.append(f"{subj} & \\texttt{{full/quarter\\_3mm\\_sharp}} & {cnt} & $512\\times 512$ & {role} \\\\")

    ds_tex = "\n".join(
        [
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "\\textbf{Subject ID} & \\textbf{Reconstruction Series} & \\textbf{Slices (Pairs)} & \\textbf{Resolution} & \\textbf{Dataset Split Role} \\\\",
            "\\midrule",
            *subj_rows,
            "\\midrule",
            f"\\textbf{{Total ({ds['num_subjects_total']} Subjects)}} & 3mm Slice, Sharp Kernel (\\texttt{{.IMA}}) & \\textbf{{{ds['total_slices']}}} & $[-100, 300]$ HU & Train {ds['train_slices']} / Val {ds['val_slices']} / Test {ds['test_slices']} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (GEN_DIR / "table_dataset_summary.tex").write_text(ds_tex + "\n", encoding="utf-8")

    # 2. Architecture Comparison Table
    arch_rows = []
    yn = lambda b: "\\checkmark" if b else "---"
    arch_keys = [k for k in ["U_Net", "U_Net_NoRes", "R2U_Net", "AttU_Net", "R2AttU_Net", "NestedUNet"] if k in specs]
    for m_key in arch_keys:
        sp = specs[m_key]
        arch_rows.append(
            f"\\texttt{{{tex_escape(m_key)}}} ({sp['display']}) & "
            f"{yn(sp['block_residual'])} & {yn(sp['global_residual'])} & "
            f"{yn(sp['recurrent'])} & {yn(sp['attention'])} & {yn(sp['nested_skip'])} & "
            f"{sp['params_m']:.2f} M & {sp['gflops_512']:.1f} \\\\"
        )

    arch_tex = "\n".join(
        [
            "\\begin{tabular}{lccccccc}",
            "\\toprule",
            "\\textbf{Architecture} & \\textbf{Block Res.} & \\textbf{Global $+x$} & \\textbf{Recurrent ($t{=}2$)} & \\textbf{Att. Gate} & \\textbf{Nested Skip} & \\textbf{Params} & \\textbf{GFLOPs ($512^2$)} \\\\",
            "\\midrule",
            *arch_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (GEN_DIR / "table_arch_specs.tex").write_text(arch_tex + "\n", encoding="utf-8")

    # 3. Main Quantitative Benchmark Table (Baseline + All Logged Runs)
    q_rows = []
    b_l192 = cache["baseline"]["summary"]["L192"]["psnr_mean"]
    b_l506 = cache["baseline"]["summary"]["L506"]["psnr_mean"]
    q_rows.append(
        f"Quarter-Dose Input (Baseline) & --- & --- & "
        f"{base_ov['psnr_mean']:.2f}$\\pm${base_ov['psnr_std']:.2f} & "
        f"{base_ov['ssim_mean']:.4f}$\\pm${base_ov['ssim_std']:.4f} & "
        f"{base_ov['rmse_hu_mean']:.2f} & {base_ov['mae_hu_mean']:.2f} & "
        f"{base_ov['epi_mean']:.4f} & {b_l192:.2f} / {b_l506:.2f} \\\\"
    )
    q_rows.append("\\midrule")

    # Order runs cleanly by model order then aug order
    ordered_run_names = []
    for m in ["U_Net", "U_Net_NoRes", "AttU_Net", "NestedUNet", "R2U_Net", "R2AttU_Net"]:
        for a in ["none", "Rot90_Flip_Combo", "RandAffined", "Affine_Noise_Combo", "Rot90_Scale_Combo"]:
            rn = f"{m}_{a}"
            if rn in runs_meta:
                ordered_run_names.append(rn)

    for rn in ordered_run_names:
        meta = runs_meta[rn]
        m_str = f"\\texttt{{{tex_escape(meta['model'])}}}"
        a_str = f"\\texttt{{{tex_escape(meta['augment'])}}}"
        if rn in eval_runs:
            ev = eval_runs[rn]
            ov = ev["summary"]["overall"]
            p192 = ev["summary"]["L192"]["psnr_mean"]
            p506 = ev["summary"]["L506"]["psnr_mean"]
            ep = ev["selected_epoch"]
            mode = ev["selection_mode"]
            if mode == "recovered_pre_nan":
                ep_str = f"Ep.{ep}$^\\dagger$"
            elif not meta.get("tb", {}).get("has_logged_test", False):
                ep_str = f"Ep.{ep}$^\\star$"
            else:
                ep_str = f"Ep.{ep}"

            is_best = rn == best_run_name
            psnr_cell = f"{ov['psnr_mean']:.2f}$\\pm${ov['psnr_std']:.2f}"
            ssim_cell = f"{ov['ssim_mean']:.4f}$\\pm${ov['ssim_std']:.4f}"
            rmse_cell = f"{ov['rmse_hu_mean']:.2f}"
            mae_cell = f"{ov['mae_hu_mean']:.2f}"
            epi_cell = f"{ov['epi_mean']:.4f}"
            if is_best:
                psnr_cell = f"\\textbf{{{psnr_cell}}}"
                ssim_cell = f"\\textbf{{{ssim_cell}}}"
                rmse_cell = f"\\textbf{{{rmse_cell}}}"
                mae_cell = f"\\textbf{{{mae_cell}}}"

            q_rows.append(
                f"{m_str} & {a_str} & {ep_str} & "
                f"{psnr_cell} & {ssim_cell} & {rmse_cell} & {mae_cell} & "
                f"{epi_cell} & {p192:.2f} / {p506:.2f} \\\\"
            )
        else:
            nan_step = meta.get("tb", {}).get("first_nan_step")
            q_rows.append(
                f"{m_str} & {a_str} & NaN (Step {nan_step})$^\\ddagger$ & "
                f"\\multicolumn{{6}}{{c}}{{\\textit{{Diverged in Epoch 1 under FP16 AMP + Out-of-Range Augmentation (See Sec.~6)}}}} \\\\"
            )

    main_tex = "\n".join(
        [
            "\\begin{tabular}{llccccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Augmentation} & \\textbf{Ckpt} & \\textbf{PSNR (dB) $\\uparrow$} & \\textbf{SSIM $\\uparrow$} & \\textbf{RMSE (HU) $\\downarrow$} & \\textbf{MAE (HU) $\\downarrow$} & \\textbf{EPI $\\uparrow$} & \\textbf{L192 / L506 (dB)} \\\\",
            "\\midrule",
            *q_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (GEN_DIR / "table_main_metrics.tex").write_text(main_tex + "\n", encoding="utf-8")

    # 4. Dynamic LaTeX Macros so inline numbers stay auto-synced on every refresh
    unet_ov = eval_runs.get("U_Net_none", {}).get("summary", {}).get("overall", best_ov)
    nores_ov = eval_runs.get("U_Net_NoRes_none", {}).get("summary", {}).get("overall", unet_ov)
    unet_rf_ov = eval_runs.get("U_Net_Rot90_Flip_Combo", {}).get("summary", {}).get("overall", unet_ov)
    att_ov = eval_runs.get("AttU_Net_none", {}).get("summary", {}).get("overall", unet_ov)
    att_rf_ov = eval_runs.get("AttU_Net_Rot90_Flip_Combo", {}).get("summary", {}).get("overall", best_ov)
    nested_ov = eval_runs.get("NestedUNet_none", {}).get("summary", {}).get("overall", None)
    nested_ep = eval_runs.get("NestedUNet_none", {}).get("selected_epoch", "---")
    r2_ov = eval_runs.get("R2U_Net_none", {}).get("summary", {}).get("overall", None)
    r2_ep = eval_runs.get("R2U_Net_none", {}).get("selected_epoch", "---")
    ra_ov = eval_runs.get("U_Net_RandAffined", {}).get("summary", {}).get("overall", None)

    macros = [
        f"\\newcommand{{\\ReportGeneratedAt}}{{{cache['generated_at']}}}",
        f"\\newcommand{{\\TotalSubjects}}{{{ds['num_subjects_total']}}}",
        f"\\newcommand{{\\TotalSlices}}{{{ds['total_slices']}}}",
        f"\\newcommand{{\\TrainSlices}}{{{ds['train_slices']}}}",
        f"\\newcommand{{\\ValSlices}}{{{ds['val_slices']}}}",
        f"\\newcommand{{\\TestSlices}}{{{ds['test_slices']}}}",
        f"\\newcommand{{\\TestLOneSlices}}{{{ds['test_l192_slices']}}}",
        f"\\newcommand{{\\TestLFiveSlices}}{{{ds['test_l506_slices']}}}",
        f"\\newcommand{{\\NumEvaluatedModels}}{{{len(eval_runs)}}}",
        f"\\newcommand{{\\NumLoggedRuns}}{{{len(runs_meta)}}}",
        f"\\newcommand{{\\BasePSNR}}{{{base_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\BaseSSIM}}{{{base_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\BaseRMSEHU}}{{{base_ov['rmse_hu_mean']:.2f}}}",
        f"\\newcommand{{\\BaseMAEHU}}{{{base_ov['mae_hu_mean']:.2f}}}",
        f"\\newcommand{{\\BaseNoiseHU}}{{{base_ov['roi_noise_hu_mean']:.2f}}}",
        f"\\newcommand{{\\BestRunLabel}}{{{tex_escape(best_run_name)}}}",
        f"\\newcommand{{\\BestPSNR}}{{{best_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\BestSSIM}}{{{best_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\BestRMSEHU}}{{{best_ov['rmse_hu_mean']:.2f}}}",
        f"\\newcommand{{\\BestMAEHU}}{{{best_ov['mae_hu_mean']:.2f}}}",
        f"\\newcommand{{\\BestNoiseHU}}{{{best_ov['roi_noise_hu_mean']:.2f}}}",
        f"\\newcommand{{\\BestEPI}}{{{best_ov['epi_mean']:.4f}}}",
        f"\\newcommand{{\\BestPSNRGain}}{{{best_ov['psnr_mean'] - base_ov['psnr_mean']:+.2f}}}",
        f"\\newcommand{{\\BestSSIMGain}}{{{best_ov['ssim_mean'] - base_ov['ssim_mean']:+.4f}}}",
        f"\\newcommand{{\\UNetPSNR}}{{{unet_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\UNetSSIM}}{{{unet_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\UNetRMSEHU}}{{{unet_ov['rmse_hu_mean']:.2f}}}",
        f"\\newcommand{{\\UNetEPI}}{{{unet_ov['epi_mean']:.4f}}}",
        f"\\newcommand{{\\UNetNoResPSNR}}{{{nores_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\UNetNoResSSIM}}{{{nores_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\UNetNoResRMSEHU}}{{{nores_ov['rmse_hu_mean']:.2f}}}",
        f"\\newcommand{{\\UNetNoResEPI}}{{{nores_ov['epi_mean']:.4f}}}",
        f"\\newcommand{{\\UNetRotFlipPSNR}}{{{unet_rf_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\UNetRotFlipSSIM}}{{{unet_rf_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\AttUNetPSNR}}{{{att_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\AttUNetSSIM}}{{{att_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\AttUNetRotFlipPSNR}}{{{att_rf_ov['psnr_mean']:.2f}}}",
        f"\\newcommand{{\\AttUNetRotFlipSSIM}}{{{att_rf_ov['ssim_mean']:.4f}}}",
        f"\\newcommand{{\\NestedUNetPSNR}}{{{nested_ov['psnr_mean']:.2f}}}" if nested_ov else "\\newcommand{\\NestedUNetPSNR}{---}",
        f"\\newcommand{{\\NestedUNetSSIM}}{{{nested_ov['ssim_mean']:.4f}}}" if nested_ov else "\\newcommand{\\NestedUNetSSIM}{---}",
        f"\\newcommand{{\\NestedUNetRMSEHU}}{{{nested_ov['rmse_hu_mean']:.2f}}}" if nested_ov else "\\newcommand{\\NestedUNetRMSEHU}{---}",
        f"\\newcommand{{\\NestedUNetEPI}}{{{nested_ov['epi_mean']:.4f}}}" if nested_ov else "\\newcommand{\\NestedUNetEPI}{---}",
        f"\\newcommand{{\\NestedUNetEpoch}}{{{nested_ep}}}",
        f"\\newcommand{{\\RTwoUNetPSNR}}{{{r2_ov['psnr_mean']:.2f}}}" if r2_ov else "\\newcommand{\\RTwoUNetPSNR}{---}",
        f"\\newcommand{{\\RTwoUNetSSIM}}{{{r2_ov['ssim_mean']:.4f}}}" if r2_ov else "\\newcommand{\\RTwoUNetSSIM}{---}",
        f"\\newcommand{{\\RTwoUNetRMSEHU}}{{{r2_ov['rmse_hu_mean']:.2f}}}" if r2_ov else "\\newcommand{\\RTwoUNetRMSEHU}{---}",
        f"\\newcommand{{\\RTwoUNetEPI}}{{{r2_ov['epi_mean']:.4f}}}" if r2_ov else "\\newcommand{\\RTwoUNetEPI}{---}",
        f"\\newcommand{{\\RTwoUNetEpoch}}{{{r2_ep}}}",
        f"\\newcommand{{\\RandAffRecPSNR}}{{{ra_ov['psnr_mean']:.2f}}}" if ra_ov else "\\newcommand{\\RandAffRecPSNR}{---}",
        f"\\newcommand{{\\RandAffRecSSIM}}{{{ra_ov['ssim_mean']:.4f}}}" if ra_ov else "\\newcommand{\\RandAffRecSSIM}{---}",
    ]
    (GEN_DIR / "macros_summary.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"[+] Wrote auto-generated LaTeX tables and macros to {GEN_DIR.relative_to(WORKSPACE_ROOT)}")


def main() -> None:
    cache_path = CACHE_DIR / "eval_cache.json"
    npz_path = CACHE_DIR / "visual_slices.npz"
    if not cache_path.exists() or not npz_path.exists():
        raise FileNotFoundError("Run report/extract_logs_and_eval.py first to populate report/cache/")

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)

    eval_runs = cache["evaluated_runs"]
    if not eval_runs:
        raise RuntimeError("No finite checkpoints found to plot.")

    best_run_name = max(
        eval_runs.keys(),
        key=lambda k: eval_runs[k]["summary"]["overall"]["psnr_mean"],
    )
    print(f"[*] Best overall evaluated checkpoint: {best_run_name}")

    plot_3_visual_examples(cache, npz, best_run_name)
    plot_residual_comparison(cache, npz)
    plot_training_and_nan_diagnostics(cache)
    plot_augmentation_pitfalls(npz)
    plot_quantitative_distributions(cache)
    write_latex_tables_and_macros(cache, best_run_name)


if __name__ == "__main__":
    main()
