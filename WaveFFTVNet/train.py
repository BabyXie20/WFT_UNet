import os
import re
import json
import time
import math
import csv
import argparse
import tempfile
import random
import shutil
import sys
import platform
import hashlib
import inspect
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from monai.config import print_config
from monai.utils import set_determinism
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAffined,
    RandShiftIntensityd,
)
import numpy as np
from networks.model import VNet


CLASS_LABELS = {
    "0": "background",
    "1": "spleen",
    "2": "rkid",
    "3": "lkid",
    "4": "gall",
    "5": "eso",
    "6": "liver",
    "7": "sto",
    "8": "aorta",
    "9": "IVC",
    "10": "veins",
    "11": "pancreas",
    "12": "rad",
    "13": "lad",
}


FIXED_EVAL_CASES = [
    {"image": "../data/btcv/imagesTr/img0027.nii.gz", "label": "../data/btcv/labelsTr/label0027.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0010.nii.gz", "label": "../data/btcv/labelsTr/label0010.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0022.nii.gz", "label": "../data/btcv/labelsTr/label0022.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0030.nii.gz", "label": "../data/btcv/labelsTr/label0030.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0005.nii.gz", "label": "../data/btcv/labelsTr/label0005.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0032.nii.gz", "label": "../data/btcv/labelsTr/label0032.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0008.nii.gz", "label": "../data/btcv/labelsTr/label0008.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0007.nii.gz", "label": "../data/btcv/labelsTr/label0007.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0001.nii.gz", "label": "../data/btcv/labelsTr/label0001.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0025.nii.gz", "label": "../data/btcv/labelsTr/label0025.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0034.nii.gz", "label": "../data/btcv/labelsTr/label0034.nii.gz"},
    {"image": "../data/btcv/imagesTr/img0036.nii.gz", "label": "../data/btcv/labelsTr/label0036.nii.gz"},
]

rot = np.deg2rad(30.0)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../data/btcv")
    parser.add_argument("--split_json", type=str, default="dataset_0.json")

    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default="", help="optional, if empty use timestamp")

    parser.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--nsd_tol_mm", type=float, default=1.0, help="NSD tolerance in mm")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2, help="RandCropByPosNegLabeld num_samples")
    parser.add_argument("--cache_num_train", type=int, default=18, help="<=0 means cache all")
    parser.add_argument("--cache_num_val", type=int, default=12, help="<=0 means cache all")
    parser.add_argument("--cache_num_test", type=int, default=12, help="<=0 means cache all")
    parser.add_argument("--cache_rate", type=float, default=1.0)

    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=6)
    parser.add_argument("--num_workers_test", type=int, default=6)

    parser.add_argument("--max_iterations", type=int, default=32000)
    parser.add_argument("--eval_num", type=int, default=400)
    parser.add_argument("--val_start_iter", type=int, default=8400, help="start val eval at this iter (inclusive)")
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=12,
        help="stop after N validations without improvement (monitor: val foreground Dice)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-3,
        help="minimum absolute improvement to reset patience",
    )
    parser.add_argument(
        "--early_stop_warmup",
        type=int,
        default=0,
        help="ignore early-stop counting for first N validations",
    )
    parser.add_argument("--train_num", type=int, default=18, help="number of train cases (default 18)")
    parser.add_argument(
        "--snapshot_extra",
        type=str,
        nargs="*",
        default=["networks"],
        help="extra relative paths (dirs/files) to snapshot into outputs (e.g. networks configs).",
    )

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="torch.backends.cudnn.benchmark=True")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def tensor_to_float_list(t: torch.Tensor) -> List[Optional[float]]:
    t = t.detach().float().cpu()
    out: List[Optional[float]] = []
    for v in t.tolist():
        out.append(safe_float(v))
    return out


def get_class_names(num_classes: int) -> List[str]:
    names = []
    for i in range(num_classes):
        key = str(i)
        names.append(CLASS_LABELS.get(key, f"class_{i}"))
    return names


def _safe_run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 5) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return int(p.returncode), p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 999, "", f"{type(e).__name__}: {e}"


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ignore_snapshot_files(dirpath: str, names: List[str]) -> set:
    ignore = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".idea",
        ".vscode",
        "wandb",
        ".DS_Store",
    }
    out = set()
    for n in names:
        if n in ignore:
            out.add(n)
        elif n.endswith((".pyc", ".pyo", ".so")):
            out.add(n)
    return out


def save_code_snapshot(output_dir: str, extra_rel_paths: Optional[List[str]] = None) -> None:
    snapshot_dir = os.path.join(output_dir, "snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

    script_path = None
    try:
        script_path = os.path.abspath(__file__)
    except Exception:
        script_path = None

    base_dir = None
    if script_path and os.path.isfile(script_path):
        base_dir = os.path.dirname(script_path)
    else:
        base_dir = os.getcwd()

    copied: List[Dict[str, Any]] = []

    if script_path and os.path.isfile(script_path):
        dst = os.path.join(snapshot_dir, os.path.basename(script_path))
        shutil.copy2(script_path, dst)
        copied.append({"type": "file", "src": script_path, "dst": dst, "sha256": _sha256_file(dst)})
    else:
        try:
            src_text = inspect.getsource(sys.modules[__name__])
            dst = os.path.join(snapshot_dir, "train_script_snapshot.py")
            with open(dst, "w", encoding="utf-8") as f:
                f.write(src_text)
            copied.append(
                {"type": "file_generated", "src": "<inspect.getsource>", "dst": dst, "sha256": _sha256_file(dst)}
            )
        except Exception as e:
            print(f"[SNAPSHOT][WARN] cannot save script source: {type(e).__name__}: {e}")

    extra_rel_paths = extra_rel_paths or []
    for rel in extra_rel_paths:
        if not rel:
            continue
        src = rel
        if not os.path.isabs(src):
            src = os.path.join(base_dir, rel)
        if not os.path.exists(src):
            print(f"[SNAPSHOT][WARN] extra path not found: {src}")
            continue

        dst = os.path.join(snapshot_dir, os.path.basename(src))
        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst, ignore=_ignore_snapshot_files)
                copied.append({"type": "dir", "src": src, "dst": dst})
            else:
                shutil.copy2(src, dst)
                copied.append({"type": "file", "src": src, "dst": dst, "sha256": _sha256_file(dst)})
        except Exception as e:
            print(f"[SNAPSHOT][WARN] failed copying {src} -> {dst}: {type(e).__name__}: {e}")

    git_commit = ""
    git_dirty = None
    git_root = None
    if base_dir:
        rc, out, _ = _safe_run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=base_dir)
        if rc == 0 and out:
            git_root = out
            rc2, out2, _ = _safe_run_cmd(["git", "rev-parse", "HEAD"], cwd=git_root)
            if rc2 == 0:
                git_commit = out2.strip()
            rc3, out3, _ = _safe_run_cmd(["git", "status", "--porcelain"], cwd=git_root)
            if rc3 == 0:
                git_dirty = (len(out3.strip()) > 0)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": os.path.abspath(output_dir),
        "snapshot_dir": os.path.abspath(snapshot_dir),
        "script_path": os.path.abspath(script_path) if script_path else "",
        "base_dir": os.path.abspath(base_dir) if base_dir else "",
        "cmdline": sys.argv,
        "python": {"version": sys.version.replace("\n", " "), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": {
            "torch": getattr(torch, "__version__", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", ""),
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "monai": "",
        },
        "git": {"root": git_root or "", "commit": git_commit, "dirty": git_dirty},
        "copied": copied,
    }

    try:
        import monai

        meta["packages"]["monai"] = getattr(monai, "__version__", "")
    except Exception:
        meta["packages"]["monai"] = ""

    with open(os.path.join(snapshot_dir, "snapshot_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SNAPSHOT] saved to: {snapshot_dir}")
    print(f"[SNAPSHOT] items: {len(copied)} (meta written: snapshot_meta.json)")


def build_fixed_splits_from_json(
    json_path: str,
    train_num: int,
    eval_cases: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pool: List[Dict[str, Any]] = []
    for k in ["training", "validation"]:
        try:
            part = load_decathlon_datalist(json_path, True, k)
            pool.extend(part)
        except Exception:
            continue

    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in pool:
        key = (d.get("image", ""), d.get("label", ""))
        if not key[0] or not key[1]:
            continue
        uniq[key] = d

    all_cases = list(uniq.values())

    def norm_case_key(image_path: str, label_path: str) -> Tuple[str, str]:
        return (os.path.normpath(image_path).replace("\\", "/"), os.path.normpath(label_path).replace("\\", "/"))

    eval_key_to_case: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in all_cases:
        eval_key_to_case[norm_case_key(d.get("image", ""), d.get("label", ""))] = d

    val_files: List[Dict[str, Any]] = []
    missing_eval: List[Dict[str, str]] = []
    for ec in eval_cases:
        key = norm_case_key(ec["image"], ec["label"])
        matched = eval_key_to_case.get(key)
        if matched is None:
            missing_eval.append(ec)
        else:
            val_files.append(matched)

    if missing_eval:
        miss = json.dumps(missing_eval, ensure_ascii=False)
        raise RuntimeError(f"Fixed eval cases not found in dataset json: {miss}")

    eval_keys = {norm_case_key(d.get("image", ""), d.get("label", "")) for d in val_files}
    train_pool = [d for d in all_cases if norm_case_key(d.get("image", ""), d.get("label", "")) not in eval_keys]

    train_num = int(train_num)
    if train_num < 0:
        train_files = train_pool
    else:
        train_files = train_pool[: min(train_num, len(train_pool))]

    test_files = list(val_files)
    return train_files, val_files, test_files


def _get_case_id(batch: Dict[str, Any]) -> str:
    try:
        md = batch.get("image_meta_dict", None)
        if md and "filename_or_obj" in md:
            fn = md["filename_or_obj"][0]
            return str(fn)
    except Exception:
        pass
    return "unknown_case"


def dice_per_class_onehot(
    y_pred_1hot: torch.Tensor,
    y_true_1hot: torch.Tensor,
    eps: float = 1e-8,
    ignore_empty: bool = True,
) -> torch.Tensor:
    if y_pred_1hot.dim() == 5:
        y_pred_1hot = y_pred_1hot[0]
    if y_true_1hot.dim() == 5:
        y_true_1hot = y_true_1hot[0]

    y_pred = y_pred_1hot.float()
    y_true = y_true_1hot.float()

    dims = tuple(range(1, y_pred.dim()))
    inter = (y_pred * y_true).sum(dim=dims)
    pred_sum = y_pred.sum(dim=dims)
    true_sum = y_true.sum(dim=dims)
    denom = pred_sum + true_sum

    dice = (2.0 * inter) / (denom + eps)

    if ignore_empty:
        nan = torch.tensor(float("nan"), device=dice.device, dtype=dice.dtype)
        dice = torch.where(true_sum > 0, dice, nan)
    else:
        dice = torch.where(denom > 0, dice, torch.ones_like(dice))
    return dice


@torch.no_grad()
def compute_hd95_nsd_per_case(
    pred_case: torch.Tensor,
    true_case: torch.Tensor,
    pixdim: Tuple[float, float, float],
    fg_classes: int,
    nsd_tol_mm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nsd_thresholds = [float(nsd_tol_mm)] * fg_classes
    tmp_nsd = SurfaceDiceMetric(
        class_thresholds=nsd_thresholds,
        include_background=False,
        reduction="none",
        get_not_nans=False,
        use_subvoxels=True,
    )
    tmp_nsd(y_pred=[pred_case], y=[true_case], spacing=pixdim)
    nsd_case = tmp_nsd.aggregate()
    tmp_nsd.reset()

    tmp_hd95 = HausdorffDistanceMetric(
        include_background=False,
        reduction="none",
        percentile=95,
        get_not_nans=False,
    )
    tmp_hd95(y_pred=[pred_case], y=[true_case], spacing=pixdim)
    hd95_case = tmp_hd95.aggregate()
    tmp_hd95.reset()

    if isinstance(nsd_case, (tuple, list)):
        nsd_case = nsd_case[0]
    if isinstance(hd95_case, (tuple, list)):
        hd95_case = hd95_case[0]

    nsd_case = nsd_case.squeeze(0).detach().cpu().float()
    hd95_case = hd95_case.squeeze(0).detach().cpu().float()
    return hd95_case, nsd_case


@torch.no_grad()
def run_evaluation_single_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    num_classes: int,
    class_names: List[str],
    post_label: AsDiscrete,
    post_pred: AsDiscrete,
    global_step: int,
    writer: Optional[SummaryWriter] = None,
    prefix: str = "val",
    compute_hd95: bool = False,
    compute_nsd: bool = False,
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    nsd_tol_mm: float = 1.0,
) -> Dict[str, Any]:
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)

    fg_classes = max(num_classes - 1, 0)
    sum_hd95 = torch.zeros((fg_classes,), dtype=torch.float64)
    cnt_hd95 = torch.zeros((fg_classes,), dtype=torch.float64)
    sum_nsd = torch.zeros((fg_classes,), dtype=torch.float64)
    cnt_nsd = torch.zeros((fg_classes,), dtype=torch.float64)

    pbar = tqdm(loader, desc=f"{prefix.upper()}@{global_step}", dynamic_ncols=True)
    for batch in pbar:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian",
        )

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(x) for x in labels_list]

        outputs_list = decollate_batch(outputs)
        outputs_convert = [post_pred(x) for x in outputs_list]

        dice_metric(y_pred=outputs_convert, y=labels_convert)

        if (compute_hd95 or compute_nsd) and fg_classes > 0:
            for pred_case, true_case in zip(outputs_convert, labels_convert):
                hd95_case, nsd_case = compute_hd95_nsd_per_case(
                    pred_case=pred_case,
                    true_case=true_case,
                    pixdim=pixdim,
                    fg_classes=fg_classes,
                    nsd_tol_mm=nsd_tol_mm,
                )
                if compute_hd95:
                    hd95_case = hd95_case.to(torch.float64)
                    valid_hd = ~torch.isnan(hd95_case)
                    sum_hd95 += torch.nan_to_num(hd95_case, nan=0.0)
                    cnt_hd95 += valid_hd.to(torch.float64)
                if compute_nsd:
                    nsd_case = nsd_case.to(torch.float64)
                    valid_nsd = ~torch.isnan(nsd_case)
                    sum_nsd += torch.nan_to_num(nsd_case, nan=0.0)
                    cnt_nsd += valid_nsd.to(torch.float64)

    dice_agg = dice_metric.aggregate()
    dice_metric.reset()

    dice_per_class = dice_agg[0] if isinstance(dice_agg, (tuple, list)) else dice_agg
    dice_per_class = dice_per_class.detach().float().cpu()

    hd95_per_class = torch.full((fg_classes,), float("nan"), dtype=torch.float32)
    nsd_per_class = torch.full((fg_classes,), float("nan"), dtype=torch.float32)

    if compute_hd95 and fg_classes > 0:
        valid_hd = cnt_hd95 > 0
        hd95_per_class[valid_hd] = (sum_hd95[valid_hd] / cnt_hd95[valid_hd]).float()
    if compute_nsd and fg_classes > 0:
        valid_nsd = cnt_nsd > 0
        nsd_per_class[valid_nsd] = (sum_nsd[valid_nsd] / cnt_nsd[valid_nsd]).float()

    dice_mean_incl_bg = float(torch.nanmean(dice_per_class).item())
    dice_mean_fg = float(torch.nanmean(dice_per_class[1:]).item()) if num_classes > 1 else dice_mean_incl_bg
    hd95_mean_excl_bg = float(torch.nanmean(hd95_per_class).item()) if hd95_per_class.numel() > 0 else float("nan")
    nsd_mean_excl_bg = float(torch.nanmean(nsd_per_class).item()) if nsd_per_class.numel() > 0 else float("nan")

    if writer is not None:
        writer.add_scalar(f"{prefix}/dice_mean_fg", dice_mean_fg, global_step)
        writer.add_scalar(f"{prefix}/dice_mean_incl_bg", dice_mean_incl_bg, global_step)
        if compute_hd95:
            writer.add_scalar(f"{prefix}/hd95_mean_excl_bg", hd95_mean_excl_bg, global_step)
        if compute_nsd:
            writer.add_scalar(f"{prefix}/nsd_mean_excl_bg", nsd_mean_excl_bg, global_step)

        for c in range(num_classes):
            writer.add_scalar(
                f"{prefix}/per_class_dice/{class_names[c]}",
                float(dice_per_class[c].item()),
                global_step,
            )

        if compute_hd95:
            for idx in range(hd95_per_class.numel()):
                writer.add_scalar(
                    f"{prefix}/per_class_hd95/{class_names[idx + 1]}",
                    float(hd95_per_class[idx].item()),
                    global_step,
                )
        if compute_nsd:
            for idx in range(nsd_per_class.numel()):
                writer.add_scalar(
                    f"{prefix}/per_class_nsd/{class_names[idx + 1]}",
                    float(nsd_per_class[idx].item()),
                    global_step,
                )

    return {
        "dice_mean_incl_bg": dice_mean_incl_bg,
        "dice_mean_fg": dice_mean_fg,
        "hd95_mean_excl_bg": hd95_mean_excl_bg,
        "nsd_mean_excl_bg": nsd_mean_excl_bg,
        "dice_per_class_incl_bg": dice_per_class,
        "hd95_per_class_excl_bg": hd95_per_class,
        "nsd_per_class_excl_bg": nsd_per_class,
    }


@torch.no_grad()
def run_test_single_model_and_print_cases(
    model: torch.nn.Module,
    ckpt_path: str,
    loader: DataLoader,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    num_classes: int,
    class_names: List[str],
    post_label: AsDiscrete,
    post_pred: AsDiscrete,
    global_step: int,
    writer: Optional[SummaryWriter] = None,
    prefix: str = "test",
    compute_hd95: bool = True,
    compute_nsd: bool = True,
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    nsd_tol_mm: float = 1.0,
) -> Dict[str, Any]:
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    sum_dice = torch.zeros((num_classes,), dtype=torch.float64)
    cnt_dice = torch.zeros((num_classes,), dtype=torch.float64)

    fg_classes = max(num_classes - 1, 0)
    sum_hd95 = torch.zeros((fg_classes,), dtype=torch.float64)
    cnt_hd95 = torch.zeros((fg_classes,), dtype=torch.float64)
    sum_nsd = torch.zeros((fg_classes,), dtype=torch.float64)
    cnt_nsd = torch.zeros((fg_classes,), dtype=torch.float64)

    case_rows: List[Dict[str, Any]] = []

    pbar = tqdm(loader, desc=f"{prefix.upper()}(best.pt)@{global_step}", dynamic_ncols=True)
    for batch in pbar:
        case_id = _get_case_id(batch)
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian",
        )

        labels_list = decollate_batch(labels)
        labels_1hot_list = [post_label(x) for x in labels_list]

        logits_list = decollate_batch(logits)
        preds_1hot_list = [post_pred(x) for x in logits_list]

        dice_c = dice_per_class_onehot(preds_1hot_list[0], labels_1hot_list[0], ignore_empty=True)
        dice_fg = float(torch.nanmean(dice_c[1:]).item()) if num_classes > 1 else float(torch.nanmean(dice_c).item())

        dice_c_cpu = dice_c.detach().cpu().double()
        mask = ~torch.isnan(dice_c_cpu)
        sum_dice += torch.nan_to_num(dice_c_cpu, nan=0.0)
        cnt_dice += mask.double()

        hd95_case = torch.full((fg_classes,), float("nan"), dtype=torch.float32)
        nsd_case = torch.full((fg_classes,), float("nan"), dtype=torch.float32)
        if (compute_hd95 or compute_nsd) and fg_classes > 0:
            hd95_case, nsd_case = compute_hd95_nsd_per_case(
                pred_case=preds_1hot_list[0],
                true_case=labels_1hot_list[0],
                pixdim=pixdim,
                fg_classes=fg_classes,
                nsd_tol_mm=nsd_tol_mm,
            )
            if compute_hd95:
                hd95_case_f64 = hd95_case.to(torch.float64)
                valid_hd = ~torch.isnan(hd95_case_f64)
                sum_hd95 += torch.nan_to_num(hd95_case_f64, nan=0.0)
                cnt_hd95 += valid_hd.to(torch.float64)
            if compute_nsd:
                nsd_case_f64 = nsd_case.to(torch.float64)
                valid_nsd = ~torch.isnan(nsd_case_f64)
                sum_nsd += torch.nan_to_num(nsd_case_f64, nan=0.0)
                cnt_nsd += valid_nsd.to(torch.float64)

        case_rows.append(
            {
                "case_id": case_id,
                "dice_mean_fg": dice_fg,
                "dice_per_class_incl_bg": tensor_to_float_list(dice_c.detach().cpu().float()),
                "hd95_per_class_excl_bg": tensor_to_float_list(hd95_case),
                "nsd_per_class_excl_bg": tensor_to_float_list(nsd_case),
            }
        )

        pbar.set_postfix({"case_fg_dice": f"{dice_fg:.4f}"})

    dice_per_class_mean = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid = cnt_dice > 0
    dice_per_class_mean[valid] = sum_dice[valid] / cnt_dice[valid]
    dice_per_class_mean_f32 = dice_per_class_mean.float()

    dice_mean_incl_bg = float(torch.nanmean(dice_per_class_mean_f32).item())
    dice_mean_fg = float(torch.nanmean(dice_per_class_mean_f32[1:]).item()) if num_classes > 1 else dice_mean_incl_bg

    hd95_per_class = torch.full((fg_classes,), float("nan"), dtype=torch.float32)
    nsd_per_class = torch.full((fg_classes,), float("nan"), dtype=torch.float32)
    if compute_hd95 and fg_classes > 0:
        valid_hd = cnt_hd95 > 0
        hd95_per_class[valid_hd] = (sum_hd95[valid_hd] / cnt_hd95[valid_hd]).float()
    if compute_nsd and fg_classes > 0:
        valid_nsd = cnt_nsd > 0
        nsd_per_class[valid_nsd] = (sum_nsd[valid_nsd] / cnt_nsd[valid_nsd]).float()

    hd95_mean_excl_bg = float(torch.nanmean(hd95_per_class).item()) if hd95_per_class.numel() > 0 else float("nan")
    nsd_mean_excl_bg = float(torch.nanmean(nsd_per_class).item()) if nsd_per_class.numel() > 0 else float("nan")

    case_rows_sorted = sorted(case_rows, key=lambda r: r["dice_mean_fg"])
    dices = np.array([r["dice_mean_fg"] for r in case_rows_sorted], dtype=np.float32)
    mean_fg = float(np.mean(dices)) if dices.size > 0 else float("nan")
    std_fg = float(np.std(dices)) if dices.size > 0 else float("nan")
    thresh = mean_fg - 2.0 * std_fg if (not math.isnan(mean_fg) and not math.isnan(std_fg)) else float("-inf")

    for i, row in enumerate(case_rows_sorted):
        row["rank_ascending_fg_dice"] = i + 1
        row["is_outlier"] = bool(row["dice_mean_fg"] < thresh)

    print("\n================ TEST CASES (sorted by fg Dice) ================")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Case fgDice mean={mean_fg:.6f} std={std_fg:.6f} | outlier<thr={thresh:.6f}\n")
    for i, r in enumerate(case_rows_sorted):
        flag = "  <-- OUTLIER" if r["is_outlier"] else ""
        print(f"[{i:02d}] fgDice={r['dice_mean_fg']:.6f} | {r['case_id']}{flag}")
    print("===============================================================\n")

    if writer is not None:
        writer.add_scalar(f"{prefix}/dice_mean_fg", dice_mean_fg, global_step)
        writer.add_scalar(f"{prefix}/dice_mean_incl_bg", dice_mean_incl_bg, global_step)
        if compute_hd95:
            writer.add_scalar(f"{prefix}/hd95_mean_excl_bg", hd95_mean_excl_bg, global_step)
        if compute_nsd:
            writer.add_scalar(f"{prefix}/nsd_mean_excl_bg", nsd_mean_excl_bg, global_step)

        for c in range(num_classes):
            writer.add_scalar(
                f"{prefix}/per_class_dice/{class_names[c]}",
                float(dice_per_class_mean_f32[c].item()) if not torch.isnan(dice_per_class_mean_f32[c]) else float("nan"),
                global_step,
            )

        if compute_hd95:
            for idx in range(hd95_per_class.numel()):
                writer.add_scalar(
                    f"{prefix}/per_class_hd95/{class_names[idx + 1]}",
                    float(hd95_per_class[idx].item()),
                    global_step,
                )
        if compute_nsd:
            for idx in range(nsd_per_class.numel()):
                writer.add_scalar(
                    f"{prefix}/per_class_nsd/{class_names[idx + 1]}",
                    float(nsd_per_class[idx].item()),
                    global_step,
                )

    return {
        "ckpt_path": ckpt_path,
        "dice_mean_incl_bg": dice_mean_incl_bg,
        "dice_mean_fg": dice_mean_fg,
        "hd95_mean_excl_bg": hd95_mean_excl_bg,
        "nsd_mean_excl_bg": nsd_mean_excl_bg,
        "dice_per_class_incl_bg": dice_per_class_mean_f32,
        "hd95_per_class_excl_bg": hd95_per_class,
        "nsd_per_class_excl_bg": nsd_per_class,
        "case_rows": case_rows_sorted,
        "case_fg_dice_mean": mean_fg,
        "case_fg_dice_std": std_fg,
        "case_fg_dice_outlier_threshold": thresh,
    }


def save_test_results_csv(
    output_dir: str,
    test_res: Dict[str, Any],
    best_path: str,
    best_step: int,
    best_val_dice_fg: float,
    class_names: List[str],
) -> Dict[str, str]:
    summary_csv = os.path.join(output_dir, "test_summary.csv")
    per_class_csv = os.path.join(output_dir, "test_per_class_metrics.csv")
    case_ranking_csv = os.path.join(output_dir, "test_case_ranking.csv")

    dice_pc = test_res["dice_per_class_incl_bg"]
    hd_pc = test_res["hd95_per_class_excl_bg"]
    nsd_pc = test_res["nsd_per_class_excl_bg"]
    case_rows = test_res["case_rows"]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "best_iter",
                "best_path",
                "best_val_dice_mean_fg",
                "test_dice_mean_fg",
                "test_dice_mean_incl_bg",
                "test_hd95_mean_excl_bg",
                "test_nsd_mean_excl_bg",
                "case_fg_dice_mean",
                "case_fg_dice_std",
                "case_fg_dice_outlier_threshold",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "best_iter": int(best_step),
                "best_path": best_path,
                "best_val_dice_mean_fg": safe_float(best_val_dice_fg),
                "test_dice_mean_fg": safe_float(test_res["dice_mean_fg"]),
                "test_dice_mean_incl_bg": safe_float(test_res["dice_mean_incl_bg"]),
                "test_hd95_mean_excl_bg": safe_float(test_res["hd95_mean_excl_bg"]),
                "test_nsd_mean_excl_bg": safe_float(test_res["nsd_mean_excl_bg"]),
                "case_fg_dice_mean": safe_float(test_res["case_fg_dice_mean"]),
                "case_fg_dice_std": safe_float(test_res["case_fg_dice_std"]),
                "case_fg_dice_outlier_threshold": safe_float(test_res["case_fg_dice_outlier_threshold"]),
            }
        )

    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_index", "class_name", "dice", "hd95", "nsd"],
        )
        writer.writeheader()
        for c, class_name in enumerate(class_names):
            dice_val = float(dice_pc[c].item()) if not torch.isnan(dice_pc[c]) else None
            hd95_val = None
            nsd_val = None
            if c > 0 and (c - 1) < hd_pc.numel():
                hd_v = hd_pc[c - 1]
                hd95_val = float(hd_v.item()) if not torch.isnan(hd_v) else None
            if c > 0 and (c - 1) < nsd_pc.numel():
                nsd_v = nsd_pc[c - 1]
                nsd_val = float(nsd_v.item()) if not torch.isnan(nsd_v) else None
            writer.writerow(
                {
                    "class_index": c,
                    "class_name": class_name,
                    "dice": dice_val,
                    "hd95": hd95_val,
                    "nsd": nsd_val,
                }
            )

    with open(case_ranking_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank_ascending_fg_dice", "case_id", "dice_mean_fg", "is_outlier", "hd95_per_class_excl_bg", "nsd_per_class_excl_bg"],
        )
        writer.writeheader()
        for row in case_rows:
            writer.writerow(
                {
                    "rank_ascending_fg_dice": int(row["rank_ascending_fg_dice"]),
                    "case_id": row["case_id"],
                    "dice_mean_fg": safe_float(row["dice_mean_fg"]),
                    "is_outlier": bool(row["is_outlier"]),
                    "hd95_per_class_excl_bg": json.dumps(row.get("hd95_per_class_excl_bg", []), ensure_ascii=False),
                    "nsd_per_class_excl_bg": json.dumps(row.get("nsd_per_class_excl_bg", []), ensure_ascii=False),
                }
            )

    return {
        "summary_csv": summary_csv,
        "per_class_csv": per_class_csv,
        "case_ranking_csv": case_ranking_csv,
    }


def main():
    args = parse_args()

    run_id = args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, f"btcv_run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    print("output_dir =", output_dir)

    save_code_snapshot(output_dir=output_dir, extra_rel_paths=list(args.snapshot_extra))

    print_config()
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print("MONAI root_dir =", root_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    seed_everything(args.seed)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[WARN] cudnn_benchmark=True may reduce reproducibility even with fixed seeds.")

    json_path = os.path.join(args.data_dir, args.split_json)

    pixdim = tuple(args.pixdim)
    roi_size = tuple(args.roi_size)
    num_classes = int(args.num_classes)
    class_names = get_class_names(num_classes)

    train_files, val_files, test_files = build_fixed_splits_from_json(
        json_path=json_path,
        train_num=args.train_num,
        eval_cases=FIXED_EVAL_CASES,
    )
    val_source = "fixed_test_as_val"

    print("\n================ SPLIT ================")
    print("[POOL] source_keys=['training','validation'] | split_seed=REMOVED")
    print(f"[VAL ] val_source={val_source} | val_start_iter={args.val_start_iter}")
    print(f"[SPLIT] train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")
    print("=======================================\n")

    cfg = vars(args).copy()
    cfg.update(
        {
            "output_dir": output_dir,
            "json_path": json_path,
            "pixdim": pixdim,
            "roi_size": roi_size,
            "device": str(device),
            "class_names": class_names,
            "class_labels": CLASS_LABELS,
            "best_selection_metric": "val/dice_mean_fg (foreground only)",
            "early_stop_metric": "val/dice_mean_fg (foreground only)",
            "test_report_metrics": [
                "test/dice_mean_fg (foreground only, single best.pt)",
                "test/hd95_mean_excl_bg (foreground only, single best.pt)",
                "test/nsd_mean_excl_bg (foreground only, single best.pt)",
            ],
            "sw_infer_mode": "gaussian",
            "code_snapshot_dir": os.path.join(output_dir, "snapshot"),
            "val_source": val_source,
            "val_start_iter": int(args.val_start_iter),
            "effective_split_sizes": {
                "train": int(len(train_files)),
                "val": int(len(val_files)),
                "test": int(len(test_files)),
            },
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    writer.add_text("meta/output_dir", output_dir, 0)
    writer.add_text("meta/json_path", json_path, 0)
    writer.add_text("meta/sw_infer_mode", "gaussian", 0)
    writer.add_text("meta/sw_overlap", str(args.sw_overlap), 0)
    writer.add_text("meta/code_snapshot_dir", os.path.join(output_dir, "snapshot"), 0)
    writer.add_text("meta/val_source", val_source, 0)
    writer.add_text("meta/val_start_iter", str(args.val_start_iter), 0)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.20,
                rotate_range=(rot, rot, rot),
                scale_range=(0.10, 0.10, 0.10),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.25),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
        ]
    )

    eval_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        ]
    )

    train_list_path = os.path.join(output_dir, "train_files_list.json")
    val_list_path = os.path.join(output_dir, "val_files_list.json")
    test_list_path = os.path.join(output_dir, "test_files_list.json")
    with open(train_list_path, "w") as f:
        json.dump(train_files, f, indent=2)
    with open(val_list_path, "w") as f:
        json.dump(val_files, f, indent=2)
    with open(test_list_path, "w") as f:
        json.dump(test_files, f, indent=2)

    g = torch.Generator()
    g.manual_seed(args.seed)

    cache_num_train = len(train_files) if args.cache_num_train <= 0 else min(args.cache_num_train, len(train_files))
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=cache_num_train,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers_train,
        pin_memory=True,
        collate_fn=list_data_collate,
        persistent_workers=(args.num_workers_train > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_ds = None
    val_loader = None
    test_ds = None
    test_loader = None

    cache_num_val = len(val_files) if args.cache_num_val <= 0 else min(args.cache_num_val, len(val_files))
    cache_num_test = len(test_files) if args.cache_num_test <= 0 else min(args.cache_num_test, len(test_files))

    def ensure_val_loader() -> DataLoader:
        nonlocal val_ds, val_loader
        if val_loader is not None:
            return val_loader
        val_ds = CacheDataset(
            data=val_files,
            transform=eval_transforms,
            cache_num=cache_num_val,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers_val,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers_val,
            pin_memory=True,
            persistent_workers=(args.num_workers_val > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
        print(f"[VAL] loader built (lazy) | size={len(val_files)} | cache_num={cache_num_val}")
        return val_loader

    def ensure_test_loader() -> DataLoader:
        nonlocal test_ds, test_loader, val_ds, val_loader
        if test_loader is not None:
            return test_loader

        if (
            (val_source == "test_as_val")
            and (val_loader is not None)
            and (args.num_workers_test == args.num_workers_val)
            and (cache_num_test == cache_num_val)
        ):
            test_loader = val_loader
            test_ds = val_ds
            print("[TEST] reuse val_loader as test_loader (val=test)")
            return test_loader

        test_ds = CacheDataset(
            data=test_files,
            transform=eval_transforms,
            cache_num=cache_num_test,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers_test,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers_test,
            pin_memory=True,
            persistent_workers=(args.num_workers_test > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
        print(f"[TEST] loader built (lazy) | size={len(test_files)} | cache_num={cache_num_test}")
        return test_loader

    model = VNet(n_channels=1, n_classes=num_classes).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    def build_param_groups(m: torch.nn.Module, weight_decay: float):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or n.endswith(".bias") or ("norm" in n.lower()) or ("bn" in n.lower()):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    param_groups = build_param_groups(model, weight_decay=float(args.wd))

    optimizer = torch.optim.SGD(
        param_groups,
        lr=float(args.lr),
        momentum=float(args.momentum),
    )

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    best_val_dice_fg = -1.0
    best_step = -1
    best_path = os.path.join(output_dir, "best.pt")

    eval_history: List[Dict[str, Any]] = []
    num_evals = 0
    bad_evals = 0
    stop_training = False
    early_stop_reason = ""

    pbar = tqdm(total=args.max_iterations, desc="Training", dynamic_ncols=True)
    t0 = time.time()

    while (global_step < args.max_iterations) and (not stop_training):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_iterations or stop_training:
                break

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_function(logits, y)

            writer.add_scalar("train/loss_total", float(loss.item()), global_step)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.update(1)

            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(lr_now), global_step)
            pbar.set_description(f"Training iter={global_step}/{args.max_iterations} loss={loss.item():.5f}")

            do_eval_now = ((global_step % args.eval_num == 0) or (global_step == args.max_iterations))
            if do_eval_now and (global_step >= int(args.val_start_iter)):
                prev_best = float(best_val_dice_fg)

                vloader = ensure_val_loader()
                val_res = run_evaluation_single_model(
                    model=model,
                    loader=vloader,
                    device=device,
                    roi_size=roi_size,
                    sw_batch_size=args.sw_batch_size,
                    sw_overlap=args.sw_overlap,
                    num_classes=num_classes,
                    class_names=class_names,
                    post_label=post_label,
                    post_pred=post_pred,
                    global_step=global_step,
                    writer=writer,
                    prefix="val",
                    compute_hd95=False,
                    compute_nsd=False,
                    pixdim=pixdim,
                    nsd_tol_mm=args.nsd_tol_mm,
                )
                model.train()
                dice_fg = float(val_res["dice_mean_fg"])
                dice_incl_bg = float(val_res["dice_mean_incl_bg"])

                eval_history.append(
                    {
                        "iter": int(global_step),
                        "train_loss_at_eval": float(loss.item()),
                        "val_dice_mean_fg": safe_float(dice_fg),
                        "val_dice_mean_incl_bg": safe_float(dice_incl_bg),
                        "val_source": val_source,
                        "val_dice_per_class_incl_bg": tensor_to_float_list(val_res["dice_per_class_incl_bg"]),
                    }
                )

                if dice_fg > best_val_dice_fg:
                    best_val_dice_fg = float(dice_fg)
                    best_step = int(global_step)
                    torch.save(model.state_dict(), best_path)
                    print(
                        f"[SAVE] best.pt @ iter={best_step} | best_val_dice_fg={best_val_dice_fg:.4f} "
                        f"| val_dice_incl_bg={dice_incl_bg:.4f}"
                    )
                else:
                    print(
                        f"[NOSAVE] best_val_dice_fg={best_val_dice_fg:.4f} | current_val_dice_fg={dice_fg:.4f} "
                        f"| val_dice_incl_bg={dice_incl_bg:.4f}"
                    )

                num_evals += 1
                improved_for_early_stop = (dice_fg > (prev_best + args.early_stop_min_delta))
                if improved_for_early_stop:
                    bad_evals = 0
                else:
                    if num_evals > args.early_stop_warmup:
                        bad_evals += 1

                writer.add_scalar("train/early_stop_bad_evals", bad_evals, global_step)
                writer.add_scalar("train/best_val_dice_mean_fg", float(best_val_dice_fg), global_step)

                if (num_evals > args.early_stop_warmup) and (bad_evals >= args.early_stop_patience):
                    stop_training = True
                    early_stop_reason = (
                        f"no improvement in val foreground Dice for {bad_evals} validations "
                        f"(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})"
                    )
                    print(f"[EARLY STOP] iter={global_step} | {early_stop_reason}")
                    writer.add_text("meta/early_stop_reason", early_stop_reason, global_step)
                    break

    pbar.close()
    dt = time.time() - t0
    print(f"Training finished. Time: {dt/60:.1f} min")
    writer.add_scalar("meta/train_minutes", dt / 60.0, global_step)

    if stop_training:
        print(f"Stopped early at iter={global_step}. Reason: {early_stop_reason}")

    test_csv_paths: Dict[str, str] = {}

    if os.path.isfile(best_path):
        tloader = ensure_test_loader()
        test_res = run_test_single_model_and_print_cases(
            model=model,
            ckpt_path=best_path,
            loader=tloader,
            device=device,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            num_classes=num_classes,
            class_names=class_names,
            post_label=post_label,
            post_pred=post_pred,
            global_step=best_step if best_step > 0 else global_step,
            writer=writer,
            prefix="test",
            compute_hd95=True,
            compute_nsd=True,
            pixdim=pixdim,
            nsd_tol_mm=args.nsd_tol_mm,
        )

        test_dice_fg = float(test_res["dice_mean_fg"])
        test_hd95 = float(test_res["hd95_mean_excl_bg"])
        test_nsd = float(test_res["nsd_mean_excl_bg"])
        dice_pc = test_res["dice_per_class_incl_bg"]
        hd_pc = test_res["hd95_per_class_excl_bg"]
        nsd_pc = test_res["nsd_per_class_excl_bg"]

        print("\n================ TEST (single best.pt) ================")
        print(f"best_iter                 : {best_step}")
        print(f"best_val_dice_fg          : {best_val_dice_fg:.6f}")
        print(f"best_path                 : {best_path}")
        print(f"TEST dice_mean_fg         : {test_dice_fg:.6f}")
        print(f"TEST hd95_mean_excl_bg    : {test_hd95:.6f}")
        print(f"TEST nsd_mean_excl_bg     : {test_nsd:.6f}")
        print("=======================================================\n")

        print("Per-class Dice (incl bg, mean-over-cases):")
        for c in range(num_classes):
            v = float(dice_pc[c].item()) if not torch.isnan(dice_pc[c]) else float("nan")
            print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

        print("\nPer-class HD95 (excl bg):")
        for c in range(1, num_classes):
            idx = c - 1
            if idx < hd_pc.numel():
                v = float(hd_pc[idx].item())
                print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

        print("\nPer-class NSD (excl bg):")
        for c in range(1, num_classes):
            idx = c - 1
            if idx < nsd_pc.numel():
                v = float(nsd_pc[idx].item())
                print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

        test_csv_paths = save_test_results_csv(
            output_dir=output_dir,
            test_res=test_res,
            best_path=best_path,
            best_step=best_step,
            best_val_dice_fg=best_val_dice_fg,
            class_names=class_names,
        )
    else:
        print(f"[WARN] best.pt not found, skip test. best_path={best_path}")

    with open(os.path.join(output_dir, "eval_history.json"), "w") as f:
        json.dump(eval_history, f, indent=2)

    writer.close()

    print("\nSaved artifacts:")
    print(" -", os.path.join(output_dir, "config.json"))
    print(" -", os.path.join(output_dir, "tb"))
    print(" -", os.path.join(output_dir, "snapshot"))
    print(" -", best_path)
    print(" -", os.path.join(output_dir, "eval_history.json"))
    print(" -", os.path.join(output_dir, "train_files_list.json"))
    print(" -", os.path.join(output_dir, "val_files_list.json"))
    print(" -", os.path.join(output_dir, "test_files_list.json"))
    if test_csv_paths:
        print(" -", test_csv_paths["summary_csv"])
        print(" -", test_csv_paths["per_class_csv"])
        print(" -", test_csv_paths["case_ranking_csv"])


if __name__ == "__main__":
    main()
