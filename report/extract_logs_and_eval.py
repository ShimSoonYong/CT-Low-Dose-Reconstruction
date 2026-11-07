#!/usr/bin/env python3
"""Read-only log extractor, pre-NaN checkpoint recovery, and fast CPU/GPU test evaluator.

Guarantees zero interference with ongoing GPU training:
- Never modifies files in src/, dataset/, script/, or log/.
- Defaults to CPU execution (CUDA_VISIBLE_DEVICES="" safe) with uniform Z-axis
  stratified slice sampling (--stride 10 on CPU -> ~46 slices across L192 & L506,
  or --stride 1 -> all 451 slices automatically when --device cuda is used).
- Preserves exact 451-slice TensorBoard test_psnr / test_ssim whenever already logged
  by Trainer._load_and_test().
- Caches per-checkpoint results incrementally in report/cache/eval_cache.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data import data_load, get_augment  # noqa: E402
from src.loss import PSNR, SSIM  # noqa: E402
from src.UNet import AttU_Net, NestedUNet, R2AttU_Net, R2U_Net, U_Net  # noqa: E402
import src.UNet as unet_pkg  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPORT_DIR / "cache"
LOG_DIR = WORKSPACE_ROOT / "log"
DATA_ROOT = WORKSPACE_ROOT / "dataset" / "CT_low_dose_reconstruction_dataset" / "Original Data"

MODEL_ORDER = ["U_Net", "U_Net_NoRes", "R2U_Net", "R2AttU_Net", "AttU_Net", "NestedUNet"]


def build_model(model_name: str) -> nn.Module:
    if model_name == "U_Net":
        return U_Net(in_ch=1, out_ch=1)
    if model_name == "U_Net_NoRes":
        cls = getattr(unet_pkg, "U_Net_NoRes", U_Net)
        return cls(in_ch=1, out_ch=1)
    if model_name == "R2U_Net":
        return R2U_Net(img_ch=1, output_ch=1)
    if model_name == "AttU_Net":
        return AttU_Net(img_ch=1, output_ch=1)
    if model_name == "R2AttU_Net":
        return R2AttU_Net(in_ch=1, out_ch=1)
    if model_name == "NestedUNet":
        return NestedUNet(in_ch=1, out_ch=1)
    raise ValueError(f"Unknown model: {model_name}")


def parse_run_name(dirname: str) -> Tuple[str, str]:
    for m in sorted(MODEL_ORDER, key=len, reverse=True):
        prefix = m + "_"
        if dirname.startswith(prefix):
            return m, dirname[len(prefix) :]
    return dirname, "unknown"


def compute_arch_specs() -> Dict[str, Dict[str, Any]]:
    """Compute exact trainable parameters and Conv2d GFLOPs for 1x1x512x512 input."""
    specs: Dict[str, Dict[str, Any]] = {}
    dummy = torch.zeros(1, 1, 512, 512, dtype=torch.float32)

    arch_meta = {
        "U_Net": {
            "display": "U-Net (Global +x)",
            "block_residual": False,
            "global_residual": True,
            "recurrent": False,
            "attention": False,
            "nested_skip": False,
        },
        "U_Net_NoRes": {
            "display": "U-Net (No Global +x)",
            "block_residual": False,
            "global_residual": False,
            "recurrent": False,
            "attention": False,
            "nested_skip": False,
        },
        "R2U_Net": {
            "display": "R2U-Net",
            "block_residual": True,
            "global_residual": True,
            "recurrent": True,
            "attention": False,
            "nested_skip": False,
        },
        "AttU_Net": {
            "display": "Attention U-Net",
            "block_residual": False,
            "global_residual": True,
            "recurrent": False,
            "attention": True,
            "nested_skip": False,
        },
        "R2AttU_Net": {
            "display": "R2Attention U-Net",
            "block_residual": True,
            "global_residual": True,
            "recurrent": True,
            "attention": True,
            "nested_skip": False,
        },
        "NestedUNet": {
            "display": "UNet++ (Nested U-Net)",
            "block_residual": False,
            "global_residual": True,
            "recurrent": False,
            "attention": False,
            "nested_skip": True,
        },
    }

    for m_name in MODEL_ORDER:
        net = build_model(m_name)
        net.eval()
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        flops = [0]
        hooks = []

        def make_hook(mod: nn.Conv2d):
            def hook(_m, _inp, out):
                _, c_out, h_out, w_out = out.shape
                k_h, k_w = mod.kernel_size
                c_in = mod.in_channels // mod.groups
                macs = c_out * h_out * w_out * c_in * k_h * k_w
                flops[0] += 2 * macs

            return hook

        for mod in net.modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(make_hook(mod)))

        with torch.inference_mode():
            _ = net(dummy)

        for h in hooks:
            h.remove()

        specs[m_name] = {
            **arch_meta[m_name],
            "params": int(params),
            "params_m": round(params / 1e6, 2),
            "gflops_512": round(flops[0] / 1e9, 2),
        }
        print(
            f"[*] Arch {m_name:12s}: {specs[m_name]['params_m']:6.2f} M params | "
            f"{specs[m_name]['gflops_512']:7.2f} GFLOPs (512x512)"
        )
    return specs


def audit_dataset() -> Dict[str, Any]:
    full_dir = DATA_ROOT / "Full Dose"
    quarter_dir = DATA_ROOT / "Quarter Dose"
    subjects: Dict[str, int] = {}
    for subj_dir in sorted(full_dir.iterdir()):
        if subj_dir.is_dir():
            cnt = len(list(subj_dir.rglob("*.IMA")))
            subjects[subj_dir.name] = cnt

    train_set, val_set, test_set = data_load(full_dir, quarter_dir, augment=None, seed=42)
    sample = test_set[0]
    shape = list(sample["full"].shape)

    return {
        "subjects": subjects,
        "num_subjects_total": len(subjects),
        "test_subjects": ["L192", "L506"],
        "train_val_subjects": [s for s in subjects if s not in ("L192", "L506")],
        "total_slices": sum(subjects.values()),
        "train_slices": len(train_set),
        "val_slices": len(val_set),
        "test_slices": len(test_set),
        "test_l192_slices": subjects.get("L192", 0),
        "test_l506_slices": subjects.get("L506", 0),
        "slice_shape": shape,
        "hu_window": [-100.0, 300.0],
        "kernel": "Sharp (3mm slice thickness)",
    }


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            sd = ckpt_obj["model_state_dict"]
        elif "network" in ckpt_obj and isinstance(ckpt_obj["network"], dict):
            sd = ckpt_obj["network"]
        else:
            sd = ckpt_obj
    elif isinstance(ckpt_obj, nn.Module):
        sd = ckpt_obj.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")

    return {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }


def is_state_dict_finite(sd: Dict[str, torch.Tensor]) -> bool:
    if not sd:
        return False
    for t in sd.values():
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            if not torch.isfinite(t).all().item():
                return False
    return True


def select_best_valid_checkpoint(run_dir: Path) -> Dict[str, Any]:
    best_path = run_dir / "best_model.pth"
    epoch_files = []
    for p in run_dir.glob("network_epoch=*.pt"):
        m = re.search(r"network_epoch=(\d+)\.pt$", p.name)
        if m:
            epoch_files.append((int(m.group(1)), p))
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    max_saved_epoch = epoch_files[0][0] if epoch_files else 0

    best_info: Dict[str, Any] = {
        "has_best_file": best_path.exists(),
        "best_file_finite": False,
        "best_file_epoch": None,
        "best_file_val_score": None,
        "max_saved_epoch": max_saved_epoch,
    }

    if best_path.exists():
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_info["best_file_epoch"] = int(ckpt.get("epoch", 0))
            bs = ckpt.get("best_score", None)
            best_info["best_file_val_score"] = (
                float(bs) if bs is not None and math.isfinite(float(bs)) else None
            )
            sd = extract_state_dict(ckpt)
            if is_state_dict_finite(sd):
                best_info["best_file_finite"] = True
                st = best_path.stat()
                return {
                    **best_info,
                    "selected_path": str(best_path),
                    "selected_epoch": best_info["best_file_epoch"],
                    "selection_mode": "best_val_selected",
                    "cache_sig": f"{best_path}:{st.st_mtime_ns}:{st.st_size}",
                }
        except Exception as exc:
            print(f"[!] Warning reading {best_path}: {exc}")

    for ep, p in epoch_files:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd = extract_state_dict(ckpt)
            if is_state_dict_finite(sd):
                st = p.stat()
                mode = (
                    "recovered_pre_nan"
                    if best_path.exists() and not best_info["best_file_finite"]
                    else "latest_epoch_in_progress"
                )
                return {
                    **best_info,
                    "selected_path": str(p),
                    "selected_epoch": ep,
                    "selection_mode": mode,
                    "cache_sig": f"{p}:{st.st_mtime_ns}:{st.st_size}",
                }
        except Exception:
            continue

    return {
        **best_info,
        "selected_path": None,
        "selected_epoch": None,
        "selection_mode": "diverged_no_finite_ckpt",
        "cache_sig": None,
    }


def parse_tensorboard_run(run_dir: Path) -> Dict[str, Any]:
    evs = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not evs:
        return {}
    ea = EventAccumulator(str(evs[-1]), size_guidance={"scalars": 2000, "images": 1})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    out: Dict[str, Any] = {"tags": tags, "scalars": {}}
    for tag in [
        "train_loss",
        "learning_rate",
        "val_psnr",
        "val_ssim",
        "val_loss",
        "test_psnr",
        "test_ssim",
        "test_loss",
    ]:
        if tag in tags:
            items = ea.Scalars(tag)
            steps = [int(x.step) for x in items]
            vals = [float(x.value) if math.isfinite(x.value) else None for x in items]
            out["scalars"][tag] = {"steps": steps, "values": vals}

    tl = out["scalars"].get("train_loss", {})
    tl_steps = tl.get("steps", [])
    tl_vals = tl.get("values", [])
    finite_pairs = [(s, v) for s, v in zip(tl_steps, tl_vals) if v is not None]
    nan_steps = [s for s, v in zip(tl_steps, tl_vals) if v is None]

    out["first_nan_step"] = nan_steps[0] if nan_steps else None
    out["last_finite_step"] = finite_pairs[-1][0] if finite_pairs else None
    out["last_finite_train_loss"] = finite_pairs[-1][1] if finite_pairs else None
    out["min_finite_train_loss"] = (
        min(v for _, v in finite_pairs) if finite_pairs else None
    )

    tp_vals = out["scalars"].get("test_psnr", {}).get("values", [])
    ts_vals = out["scalars"].get("test_ssim", {}).get("values", [])
    out["has_logged_test"] = "test_psnr" in out["scalars"]
    out["logged_test_psnr"] = (
        tp_vals[-1] if tp_vals and tp_vals[-1] is not None and tp_vals[-1] > 1.0 else None
    )
    out["logged_test_ssim"] = (
        ts_vals[-1] if ts_vals and ts_vals[-1] is not None and ts_vals[-1] > 0.01 else None
    )
    return out


def sobel_edge_correlation(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    gx_p = F.conv2d(pred, kx, padding=1)
    gy_p = F.conv2d(pred, ky, padding=1)
    mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-12).flatten(1)

    gx_t = F.conv2d(target, kx, padding=1)
    gy_t = F.conv2d(target, ky, padding=1)
    mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-12).flatten(1)

    mp = mag_p - mag_p.mean(dim=1, keepdim=True)
    mt = mag_t - mag_t.mean(dim=1, keepdim=True)
    corr = (mp * mt).sum(dim=1) / (
        torch.sqrt((mp * mp).sum(dim=1) * (mt * mt).sum(dim=1)) + 1e-12
    )
    return [float(c.item()) for c in corr]


def homogeneous_roi_noise_std_hu(
    pred: torch.Tensor, target: torch.Tensor
) -> List[float]:
    t_unf = F.unfold(target, kernel_size=16, stride=16)
    p_unf = F.unfold(pred, kernel_size=16, stride=16)
    diff_unf = (p_unf - t_unf) * 400.0

    t_mean = t_unf.mean(dim=1)
    t_std = t_unf.std(dim=1)
    results = []
    for b in range(pred.shape[0]):
        mask = (t_mean[b] > 0.22) & (t_mean[b] < 0.65) & (t_std[b] < 0.035)
        if mask.sum() >= 5:
            patch_stds = diff_unf[b, :, mask].std(dim=0)
            results.append(float(patch_stds.median().item()))
        else:
            body = t_mean[b] > 0.15
            if body.sum() > 0:
                sub_std = t_std[b][body]
                sub_diff = diff_unf[b][:, body]
                k = min(15, int(body.sum().item()))
                idx = torch.topk(sub_std, k=k, largest=False).indices
                results.append(float(sub_diff[:, idx].std(dim=0).median().item()))
            else:
                results.append(0.0)
    return results


def summarize_slice_metrics(
    records: List[Dict[str, Any]],
    override_overall_psnr: Optional[float] = None,
    override_overall_ssim: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    metric_keys = ["psnr", "ssim", "rmse_norm", "rmse_hu", "mae_hu", "epi", "roi_noise_hu"]
    groups = {
        "overall": records,
        "L192": [r for r in records if r["subject"] == "L192"],
        "L506": [r for r in records if r["subject"] == "L506"],
    }
    summary: Dict[str, Dict[str, float]] = {}
    for gname, rows in groups.items():
        if not rows:
            continue
        gdict: Dict[str, float] = {"n": len(rows)}
        for mk in metric_keys:
            arr = np.array([r[mk] for r in rows], dtype=np.float64)
            gdict[f"{mk}_mean"] = float(np.mean(arr))
            gdict[f"{mk}_std"] = float(np.std(arr))
        summary[gname] = gdict

    if override_overall_psnr is not None and "overall" in summary:
        summary["overall"]["psnr_mean"] = float(override_overall_psnr)
    if override_overall_ssim is not None and "overall" in summary:
        summary["overall"]["ssim_mean"] = float(override_overall_ssim)
    return summary


def select_eval_and_visual_slices(
    stride: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[str],
    List[str],
    List[int],
    List[Dict[str, Any]],
    List[int],
]:
    """Load uniformly strided Z-axis test slices + 4 deterministic anatomical visual slices."""
    full_dir = DATA_ROOT / "Full Dose"
    quarter_dir = DATA_ROOT / "Quarter Dose"
    test_transforms = get_augment(None)

    all_target = sorted(
        p for p in full_dir.rglob("*.IMA") if any(s in p.parts for s in ("L192", "L506"))
    )
    all_input = sorted(
        p for p in quarter_dir.rglob("*.IMA") if any(s in p.parts for s in ("L192", "L506"))
    )
    all_subjects = ["L192" if "L192" in p.parts else "L506" for p in all_target]

    l192_full_idxs = [i for i, s in enumerate(all_subjects) if s == "L192"]
    l506_full_idxs = [i for i, s in enumerate(all_subjects) if s == "L506"]

    picks = [
        {
            "case_id": "Example 1",
            "global_idx": l192_full_idxs[int(len(l192_full_idxs) * 0.25)],
            "subject": "L192",
            "region_ko": "상복부 간 실질 및 문맥 단면 (Upper Abdomen: Liver & Portal Vein)",
        },
        {
            "case_id": "Example 2",
            "global_idx": l192_full_idxs[int(len(l192_full_idxs) * 0.58)],
            "subject": "L192",
            "region_ko": "중복부 신장·대동맥 및 연조직 경계 단면 (Mid Abdomen: Kidneys & Aorta)",
        },
        {
            "case_id": "Example 3",
            "global_idx": l506_full_idxs[int(len(l506_full_idxs) * 0.35)],
            "subject": "L506",
            "region_ko": "복부 장기 및 혈관 미세구조 단면 (Abdominal Parenchyma & Vessels)",
        },
        {
            "case_id": "Example 4",
            "global_idx": l506_full_idxs[int(len(l506_full_idxs) * 0.78)],
            "subject": "L506",
            "region_ko": "하복부·골반골 및 연조직 단면 (Lower Abdomen / Pelvis & Iliac Bone)",
        },
    ]

    strided_set = set(range(0, len(all_target), max(1, stride)))
    for p in picks:
        strided_set.add(p["global_idx"])
    selected_global_idxs = sorted(strided_set)
    global_to_local = {g: loc for loc, g in enumerate(selected_global_idxs)}

    quarters, fulls, subjects, filenames = [], [], [], []
    print(
        f"[*] Loading {len(selected_global_idxs)}/{len(all_target)} stratified test slices "
        f"(stride={stride}, L192 & L506)..."
    )
    for g in selected_global_idxs:
        fp, qp = all_target[g], all_input[g]
        d = test_transforms({"quarter": str(qp), "full": str(fp)})
        quarters.append(d["quarter"])
        fulls.append(d["full"])
        subjects.append(all_subjects[g])
        filenames.append(fp.name)

    pick_local_idxs = []
    for p in picks:
        loc = global_to_local[p["global_idx"]]
        p["idx"] = loc
        p["filename"] = filenames[loc]
        pick_local_idxs.append(loc)

    return (
        torch.stack(quarters, dim=0),
        torch.stack(fulls, dim=0),
        subjects,
        filenames,
        selected_global_idxs,
        picks,
        pick_local_idxs,
    )


def evaluate_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    subjects: List[str],
    device: torch.device,
    override_psnr: Optional[float] = None,
    override_ssim: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    psnr_fn = PSNR(max_val=1.0, device=device)
    ssim_fn = SSIM(spatial_dims=2, max_val=1.0, kernel_size=11, device=device)

    records: List[Dict[str, Any]] = []
    bs = 16
    with torch.inference_mode():
        for i in range(0, preds.shape[0], bs):
            p = preds[i : i + bs].to(device)
            t = targets[i : i + bs].to(device)
            psnr_raw = psnr_fn(p, t)
            psnr_b = psnr_raw.squeeze(-1) if psnr_raw.ndim > 1 else psnr_raw
            ssim_b = ssim_fn(p, t).squeeze(-1)
            mse_b = ((p - t) ** 2).mean(dim=(1, 2, 3))
            mae_b = (p - t).abs().mean(dim=(1, 2, 3))
            epi_b = sobel_edge_correlation(p, t)
            rnoise_b = homogeneous_roi_noise_std_hu(p, t)

            for j in range(p.shape[0]):
                idx = i + j
                rmse_n = math.sqrt(float(mse_b[j].item()))
                records.append(
                    {
                        "idx": idx,
                        "subject": subjects[idx],
                        "psnr": float(psnr_b[j].item()),
                        "ssim": float(ssim_b[j].item()),
                        "rmse_norm": rmse_n,
                        "rmse_hu": 400.0 * rmse_n,
                        "mae_hu": 400.0 * float(mae_b[j].item()),
                        "epi": epi_b[j],
                        "roi_noise_hu": rnoise_b[j],
                    }
                )

    summary = summarize_slice_metrics(
        records,
        override_overall_psnr=override_psnr,
        override_overall_ssim=override_ssim,
    )
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Z-axis slice stride (defaults to 8 on CPU for <2min runtime, 1 on CUDA)",
    )
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    stride = args.stride if args.stride is not None else (1 if use_cuda else 8)

    torch.set_num_threads(args.threads)
    print(
        f"[*] Report evaluator running strictly read-only on device={device} "
        f"(threads={args.threads}, slice_stride={stride})"
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "eval_cache.json"
    npz_path = CACHE_DIR / "visual_slices.npz"

    old_cache: Dict[str, Any] = {}
    if cache_path.exists() and not args.force_eval:
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if loaded.get("eval_stride") == stride:
                old_cache = loaded
        except Exception:
            old_cache = {}

    dataset_info = old_cache.get("dataset_info") or audit_dataset()
    cached_specs = old_cache.get("arch_specs")
    arch_specs = (
        cached_specs
        if (cached_specs and all(m in cached_specs for m in MODEL_ORDER))
        else compute_arch_specs()
    )

    run_dirs = {p.name: p for p in sorted(LOG_DIR.iterdir()) if p.is_dir()} if LOG_DIR.exists() else {}
    pending_eval_runs = []
    run_metadata: Dict[str, Dict[str, Any]] = {}

    for run_name, rdir in run_dirs.items():
        model_name, aug_name = parse_run_name(run_name)
        if model_name not in MODEL_ORDER:
            continue
        tb_info = parse_tensorboard_run(rdir)
        ckpt_info = select_best_valid_checkpoint(rdir)
        run_metadata[run_name] = {
            "run_name": run_name,
            "model": model_name,
            "augment": aug_name,
            "tb": tb_info,
            "ckpt": ckpt_info,
        }
        sig = ckpt_info["cache_sig"]
        cached_run = old_cache.get("evaluated_runs", {}).get(run_name)
        if sig is not None and (
            cached_run is None or cached_run.get("cache_sig") != sig or args.force_eval
        ):
            pending_eval_runs.append(run_name)

    need_baseline = ("baseline" not in old_cache) or args.force_eval
    need_visuals = (not npz_path.exists()) or bool(pending_eval_runs) or need_baseline

    if not (need_baseline or pending_eval_runs or need_visuals):
        # Update timestamps, arch_specs, & TB curves even if no checkpoints changed
        old_cache["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        old_cache["arch_specs"] = arch_specs
        old_cache["runs_metadata"] = run_metadata
        cache_path.write_text(json.dumps(old_cache, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[=] All checkpoints up-to-date in eval_cache.json; refreshed TensorBoard curves.")
        return

    (
        q_tensor,
        f_tensor,
        subjects,
        _filenames,
        _global_idxs,
        picks_meta,
        pick_local_idxs,
    ) = select_eval_and_visual_slices(stride=stride)

    baseline_data = old_cache.get("baseline")
    if need_baseline:
        print("[*] Evaluating Quarter-Dose Input Baseline...")
        b_summary, b_records = evaluate_predictions(q_tensor, f_tensor, subjects, device)
        baseline_data = {"summary": b_summary, "per_slice": b_records}

    evaluated_runs: Dict[str, Any] = dict(old_cache.get("evaluated_runs", {}))

    npz_dict: Dict[str, np.ndarray] = {}
    if npz_path.exists() and not need_baseline:
        old_npz = np.load(npz_path)
        for k in old_npz.files:
            npz_dict[k] = old_npz[k]
    npz_dict["indices"] = np.array(pick_local_idxs, dtype=np.int32)
    npz_dict["quarter"] = q_tensor[pick_local_idxs, 0].numpy()
    npz_dict["full"] = f_tensor[pick_local_idxs, 0].numpy()

    bs = 16 if use_cuda else 4
    for run_name in pending_eval_runs:
        meta = run_metadata[run_name]
        ckpt_path = Path(meta["ckpt"]["selected_path"])
        model_name = meta["model"]
        print(
            f"[*] Evaluating {run_name} ({meta['ckpt']['selection_mode']}, "
            f"epoch={meta['ckpt']['selected_epoch']}) on {len(subjects)} slices..."
        )
        t0 = time.time()
        net = build_model(model_name).to(device)
        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(extract_state_dict(ckpt_obj))
        net.eval()

        preds_list = []
        with torch.inference_mode():
            for i in range(0, q_tensor.shape[0], bs):
                xb = q_tensor[i : i + bs].to(device)
                yb = net(xb).cpu()
                preds_list.append(yb)
        preds = torch.cat(preds_list, dim=0)
        elapsed = time.time() - t0

        # Use exact 451-slice GPU test PSNR/SSIM logged in TB if this is the final best_model.pth
        tb_psnr = (
            meta["tb"].get("logged_test_psnr")
            if meta["ckpt"]["selection_mode"] == "best_val_selected"
            else None
        )
        tb_ssim = (
            meta["tb"].get("logged_test_ssim")
            if meta["ckpt"]["selection_mode"] == "best_val_selected"
            else None
        )

        summary, per_slice = evaluate_predictions(
            preds,
            f_tensor,
            subjects,
            device,
            override_psnr=tb_psnr,
            override_ssim=tb_ssim,
        )
        npz_dict[f"pred__{run_name}"] = preds[pick_local_idxs, 0].numpy()

        print(
            f"    -> Done in {elapsed:.1f}s | PSNR={summary['overall']['psnr_mean']:.2f} dB, "
            f"SSIM={summary['overall']['ssim_mean']:.4f}, RMSE={summary['overall']['rmse_hu_mean']:.2f} HU"
        )
        evaluated_runs[run_name] = {
            "cache_sig": meta["ckpt"]["cache_sig"],
            "selected_epoch": meta["ckpt"]["selected_epoch"],
            "selection_mode": meta["ckpt"]["selection_mode"],
            "eval_seconds": round(elapsed, 2),
            "summary": summary,
            "per_slice": per_slice,
        }

        # Save incremental checkpoint progress
        np.savez_compressed(npz_path, **npz_dict)
        interim_payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_stride": stride,
            "dataset_info": dataset_info,
            "arch_specs": arch_specs,
            "baseline": baseline_data,
            "runs_metadata": run_metadata,
            "evaluated_runs": evaluated_runs,
            "representative_picks": picks_meta,
        }
        cache_path.write_text(json.dumps(interim_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Ensure visual_slices.npz has predictions for any previously cached models too
    for run_name, meta in run_metadata.items():
        if meta["ckpt"]["selected_path"] is not None and f"pred__{run_name}" not in npz_dict:
            net = build_model(meta["model"]).to(device)
            ckpt_obj = torch.load(meta["ckpt"]["selected_path"], map_location="cpu", weights_only=False)
            net.load_state_dict(extract_state_dict(ckpt_obj))
            net.eval()
            with torch.inference_mode():
                npz_dict[f"pred__{run_name}"] = net(q_tensor[pick_local_idxs].to(device)).cpu()[:, 0].numpy()

    np.savez_compressed(npz_path, **npz_dict)
    full_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_stride": stride,
        "dataset_info": dataset_info,
        "arch_specs": arch_specs,
        "baseline": baseline_data,
        "runs_metadata": run_metadata,
        "evaluated_runs": evaluated_runs,
        "representative_picks": picks_meta,
    }
    cache_path.write_text(json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Wrote evaluation cache to {cache_path.relative_to(WORKSPACE_ROOT)}")


if __name__ == "__main__":
    main()
