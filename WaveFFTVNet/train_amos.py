import os
import re
import json
import time
import math
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
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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
    "10": "pancreas",
    "11": "rad",
    "12": "lad",
    "13": "duodenum",
    "14": "bladder",
    "15": "prostate_or_uterus",
}

rot = np.deg2rad(30.0)


def parse_args():
    parser = argparse.ArgumentParser(
        "MONAI AMOS - Iteration Training + TensorBoard + custom split; "
    )

    # paths
    parser.add_argument("--data_dir", type=str, default="../data2/amos")
    parser.add_argument("--split_json", type=str, default="dataset_0.json")

    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default="", help="optional, if empty use timestamp")

    # preprocessing
    parser.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_classes", type=int, default=16)

    # dataset/cache
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2, help="RandCropByPosNegLabeld num_samples")
    parser.add_argument("--cache_num_train", type=int, default=160, help="<=0 means cache all")
    parser.add_argument("--cache_num_val", type=int, default=20, help="<=0 means cache all")
    parser.add_argument("--cache_num_test", type=int, default=20, help="<=0 means cache all")
    parser.add_argument("--cache_rate", type=float, default=1.0)

    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=6)
    parser.add_argument("--num_workers_test", type=int, default=6)

    # train/val schedule (iteration-based)
    parser.add_argument("--max_iterations", type=int, default=40000)
    parser.add_argument("--eval_num", type=int, default=400)
    parser.add_argument("--val_start_iter", type=int, default=11000, help="start val eval at this iter (inclusive)")
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3, help="keep top-k checkpoints by val foreground Dice")
    parser.add_argument(
        "--save_ckpt_every",
        type=int,
        default=4000,
        help="save periodic checkpoint every N iterations; <=0 disables it",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to a full training checkpoint for resuming",
    )
    # optim
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # PolyLR (per-iteration)
    parser.add_argument("--use_poly", action="store_true", help="use PolyLR per-iteration")
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=11,
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
    parser.add_argument(
        "--pool_keys",
        type=str,
        nargs="+",
        default=["training"],
        help="take labeled cases from these json keys and merge as pool (must contain image+label pairs)",
    )
    parser.add_argument("--train_num", type=int, default=160, help="number of train cases (default 160)")
    parser.add_argument(
        "--val_num",
        type=int,
        default=20,
        help="number of validation cases (ONLY used when --val_separate is set). Default 20.",
    )
    parser.add_argument("--test_num", type=int, default=20, help="number of test cases (default 20)")
    parser.add_argument(
        "--val_separate",
        action="store_true",
        help="if set, split into independent val and test sets using val_num/test_num. "
        "If not set (default), val set will be identical to test set.",
    )
    parser.add_argument(
        "--disable_json_split",
        action="store_true",
        help="disable predefined training/validation/test split from json and use random split arguments instead.",
    )
    parser.add_argument("--split_seed", type=int, default=123, help="seed for deterministic split shuffling")
    parser.add_argument(
        "--exclude_cases",
        type=str,
        nargs="*",
        default=[],
        help="exclude these case ids before splitting (e.g. 0008). default: [] (no exclusion)",
    )
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

    # 1) Save main script
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

    # 2) Save extra paths
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

    # 3) Git info (optional)
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
        import monai  # noqa: F401
        meta["packages"]["monai"] = getattr(monai, "__version__", "")
    except Exception:
        meta["packages"]["monai"] = ""

    with open(os.path.join(snapshot_dir, "snapshot_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SNAPSHOT] saved to: {snapshot_dir}")
    print(f"[SNAPSHOT] items: {len(copied)} (meta written: snapshot_meta.json)")


# ----------------------------
# helpers for excluding cases before split
# ----------------------------
def _normalize_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if cid.isdigit():
        return cid.zfill(4)
    return cid


def _extract_case_ids_from_basename(base: str) -> List[str]:
    ids: List[str] = []
    ids4 = re.findall(r"(?<!\d)(\d{4})(?!\d)", base)
    if ids4:
        return [x for x in ids4]
    for g in re.findall(r"\d+", base):
        if len(g) <= 4:
            ids.append(g.zfill(4))
    return ids


def _match_excluded_case(d: Dict[str, Any], excluded_set: set) -> bool:
    for k in ("image", "label"):
        p = str(d.get(k, "") or "")
        base = os.path.basename(p)
        cand = _extract_case_ids_from_basename(base)
        if any(c in excluded_set for c in cand):
            return True
    return False


def _replace_dir_token(path_str: str, old_token: str, new_token: str) -> str:
    if not path_str:
        return path_str
    pattern = rf'([/\\\\]){re.escape(old_token)}([/\\\\])'
    return re.sub(pattern, rf"\1{new_token}\2", path_str, count=1)


def _fix_single_pair_paths(d: Dict[str, Any], split_name: str = "test") -> Dict[str, Any]:
    """
    如果 json 里的 eval/test 路径写成 imagesTs/labelsTs，但本地真实文件在 imagesTr/labelsTr，
    则自动回退到 Tr 路径。
    """
    out = dict(d)

    img = str(out.get("image", "") or "")
    lab = str(out.get("label", "") or "")

    if img and (not os.path.exists(img)):
        cand = _replace_dir_token(img, "imagesTs", "imagesTr")
        if cand != img and os.path.exists(cand):
            print(f"[PATH_FIX][{split_name}] image: {img}  -->  {cand}")
            img = cand

    if lab and (not os.path.exists(lab)):
        cand = _replace_dir_token(lab, "labelsTs", "labelsTr")
        if cand != lab and os.path.exists(cand):
            print(f"[PATH_FIX][{split_name}] label: {lab}  -->  {cand}")
            lab = cand

    out["image"] = img
    out["label"] = lab
    return out


def _fix_split_pair_paths(files: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """
    对一个 split 的全部样本做路径修正，并在修正后做存在性检查。
    """
    fixed: List[Dict[str, Any]] = []
    changed = 0

    for d in files:
        nd = _fix_single_pair_paths(d, split_name=split_name)
        if (nd.get("image") != d.get("image")) or (nd.get("label") != d.get("label")):
            changed += 1
        fixed.append(nd)

    if changed > 0:
        print(f"[PATH_FIX][{split_name}] changed {changed}/{len(files)} entries")

    missing: List[Tuple[int, str, str]] = []
    for i, d in enumerate(fixed):
        for k in ("image", "label"):
            p = str(d.get(k, "") or "")
            if (not p) or (not os.path.exists(p)):
                missing.append((i, k, p))
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break

    if missing:
        msg = "\n".join([f"  [{i}] {k}: {p}" for i, k, p in missing])
        raise FileNotFoundError(
            f"[PATH_FIX][{split_name}] some files still do not exist after fallback:\n{msg}"
        )

    return fixed


# ----------------------------
# stable case_id helpers
# ----------------------------
def _strip_known_extensions(name: str) -> str:
    name = os.path.basename(str(name))
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7]
    stem, _ = os.path.splitext(name)
    return stem


def _derive_case_id_from_entry(d: Dict[str, Any]) -> str:
    """
    稳定生成 case_id，优先级：
    1) 样本字典里已有 case_id，则直接用
    2) 从 image / label 文件名中提取共同数字 id（优先 4 位）
    3) 从 image 文件名提取数字 id
    4) 从 label 文件名提取数字 id
    5) 退化为 image basename（去后缀）
    """
    explicit = str(d.get("case_id", "") or "").strip()
    if explicit:
        return explicit

    img_path = str(d.get("image", "") or "")
    lab_path = str(d.get("label", "") or "")

    img_base = _strip_known_extensions(img_path)
    lab_base = _strip_known_extensions(lab_path)

    img_ids = _extract_case_ids_from_basename(img_base)
    lab_ids = _extract_case_ids_from_basename(lab_base)

    if img_ids and lab_ids:
        lab_set = set(lab_ids)
        for cid in img_ids:
            if cid in lab_set:
                return cid

    if img_ids:
        return img_ids[0]
    if lab_ids:
        return lab_ids[0]

    if img_base:
        return img_base
    if lab_base:
        return lab_base

    return "unknown_case"


def attach_case_ids(files: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """
    给每个样本显式写入 case_id。
    这样后续不依赖 MONAI metadata，也能稳定拿到真实 case_id。
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}

    for d in files:
        nd = dict(d)
        cid = _derive_case_id_from_entry(nd)
        nd["case_id"] = cid
        out.append(nd)
        seen[cid] = seen.get(cid, 0) + 1

    dup_ids = [cid for cid, cnt in seen.items() if cnt > 1]
    if dup_ids:
        print(
            f"[CASE_ID][{split_name}][WARN] duplicated case_id detected: "
            f"{dup_ids[:10]}{' ...' if len(dup_ids) > 10 else ''}"
        )

    print(f"[CASE_ID][{split_name}] attached case_id for {len(out)} samples")
    return out


def _unwrap_singleton(x: Any) -> Any:
    """
    从 DataLoader collate 后的 list/tuple/numpy object 中取出单个值。
    主要用于 batch_size=1 的 val/test。
    """
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x.reshape(-1)[0]

    return x


def build_custom_splits_from_json(
    json_path: str,
    pool_keys: List[str],
    train_num: int,
    val_num: int,
    test_num: int,
    split_seed: int,
    exclude_cases: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pool: List[Dict[str, Any]] = []
    for k in pool_keys:
        part = load_decathlon_datalist(json_path, True, k)
        pool.extend(part)

    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in pool:
        key = (d.get("image", ""), d.get("label", ""))
        if not key[0] or not key[1]:
            continue
        uniq[key] = d
    pool = list(uniq.values())

    excluded_set = set()
    if exclude_cases:
        excluded_set = {_normalize_case_id(x) for x in exclude_cases}

    if excluded_set:
        before = len(pool)
        pool = [d for d in pool if not _match_excluded_case(d, excluded_set)]
        after = len(pool)
        print(f"[SPLIT] excluded cases={sorted(list(excluded_set))} | removed={before-after} | remain={after}")

    rng = random.Random(int(split_seed))
    rng.shuffle(pool)

    total = len(pool)
    val_num = max(0, int(val_num))
    test_num = max(0, int(test_num))
    train_num = int(train_num)

    if val_num + test_num > total:
        overflow = val_num + test_num - total
        test_num = max(0, test_num - overflow)

    remain = total - (val_num + test_num)
    if train_num < 0:
        train_num = remain
    else:
        train_num = min(train_num, remain)

    val_files = pool[:val_num]
    test_files = pool[val_num: val_num + test_num]
    train_files = pool[val_num + test_num: val_num + test_num + train_num]
    return train_files, val_files, test_files


def _get_case_id(batch: Dict[str, Any]) -> str:
    # 1) 最优先：直接读取我们显式注入的 case_id
    try:
        cid = _unwrap_singleton(batch.get("case_id", None))
        if cid is not None:
            cid = str(cid).strip()
            if cid:
                return cid
    except Exception:
        pass

    # 2) 兼容旧式 image_meta_dict
    try:
        md = batch.get("image_meta_dict", None)
        if isinstance(md, dict) and ("filename_or_obj" in md):
            fn = _unwrap_singleton(md["filename_or_obj"])
            if fn is not None:
                return _derive_case_id_from_entry({"image": str(fn)})
    except Exception:
        pass

    # 3) 兼容 MetaTensor.meta
    try:
        img = batch.get("image", None)
        if hasattr(img, "meta") and isinstance(img.meta, dict):
            fn = img.meta.get("filename_or_obj", None)
            fn = _unwrap_singleton(fn)
            if fn is not None:
                return _derive_case_id_from_entry({"image": str(fn)})
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
) -> Dict[str, Any]:
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)

    hd95_metric = None
    if compute_hd95:
        hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean_batch",
            percentile=95,
            get_not_nans=True,
        )

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
        if hd95_metric is not None:
            hd95_metric(y_pred=outputs_convert, y=labels_convert)

    dice_agg = dice_metric.aggregate()
    dice_metric.reset()

    dice_per_class = dice_agg[0] if isinstance(dice_agg, (tuple, list)) else dice_agg
    dice_per_class = dice_per_class.detach().float().cpu()

    if hd95_metric is not None:
        hd_agg = hd95_metric.aggregate()
        hd95_metric.reset()
        hd95_per_class = hd_agg[0] if isinstance(hd_agg, (tuple, list)) else hd_agg
        hd95_per_class = hd95_per_class.detach().float().cpu()
    else:
        hd95_per_class = torch.full((max(num_classes - 1, 0),), float("nan"), dtype=torch.float32)

    dice_mean_incl_bg = float(torch.nanmean(dice_per_class).item())
    dice_mean_fg = float(torch.nanmean(dice_per_class[1:]).item()) if num_classes > 1 else dice_mean_incl_bg
    hd95_mean_excl_bg = float(torch.nanmean(hd95_per_class).item()) if hd95_per_class.numel() > 0 else float("nan")

    if writer is not None:
        writer.add_scalar(f"{prefix}/dice_mean_fg", dice_mean_fg, global_step)
        writer.add_scalar(f"{prefix}/dice_mean_incl_bg", dice_mean_incl_bg, global_step)
        if compute_hd95:
            writer.add_scalar(f"{prefix}/hd95_mean_excl_bg", hd95_mean_excl_bg, global_step)

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

    return {
        "dice_mean_incl_bg": dice_mean_incl_bg,
        "dice_mean_fg": dice_mean_fg,
        "hd95_mean_excl_bg": hd95_mean_excl_bg,
        "dice_per_class_incl_bg": dice_per_class,
        "hd95_per_class_excl_bg": hd95_per_class,
    }


@torch.no_grad()
def run_test_ensemble_and_print_cases(
    model: torch.nn.Module,
    ckpt_paths: List[str],
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
) -> Dict[str, Any]:
    assert len(ckpt_paths) > 0, "ckpt_paths is empty"

    sum_dice = torch.zeros((num_classes,), dtype=torch.float64)
    cnt_dice = torch.zeros((num_classes,), dtype=torch.float64)

    hd95_metric = None
    if compute_hd95:
        hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean_batch",
            percentile=95,
            get_not_nans=True,
        )

    case_rows: List[Dict[str, Any]] = []

    pbar = tqdm(loader, desc=f"{prefix.upper()}(ens{len(ckpt_paths)})@{global_step}", dynamic_ncols=True)
    for batch in pbar:
        case_id = _get_case_id(batch)
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits_sum = None
        for p in ckpt_paths:
            sd = torch.load(p, map_location=device)
            model.load_state_dict(sd, strict=True)
            model.eval()

            out = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )
            logits_sum = out if logits_sum is None else (logits_sum + out)

        logits = logits_sum / float(len(ckpt_paths))

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

        if hd95_metric is not None:
            hd95_metric(y_pred=preds_1hot_list, y=labels_1hot_list)

        case_rows.append(
            {
                "case_id": case_id,
                "dice_mean_fg": dice_fg,
                "dice_per_class_incl_bg": tensor_to_float_list(dice_c.detach().cpu().float()),
            }
        )

        pbar.set_postfix({"case_fg_dice": f"{dice_fg:.4f}"})

    dice_per_class_mean = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid = cnt_dice > 0
    dice_per_class_mean[valid] = sum_dice[valid] / cnt_dice[valid]
    dice_per_class_mean_f32 = dice_per_class_mean.float()

    dice_mean_incl_bg = float(torch.nanmean(dice_per_class_mean_f32).item())
    dice_mean_fg = float(torch.nanmean(dice_per_class_mean_f32[1:]).item()) if num_classes > 1 else dice_mean_incl_bg

    if hd95_metric is not None:
        hd_agg = hd95_metric.aggregate()
        hd95_metric.reset()
        hd95_per_class = hd_agg[0] if isinstance(hd_agg, (tuple, list)) else hd_agg
        hd95_per_class = hd95_per_class.detach().float().cpu()
        hd95_mean_excl_bg = float(torch.nanmean(hd95_per_class).item()) if hd95_per_class.numel() > 0 else float("nan")
    else:
        determine = max(num_classes - 1, 0)
        hd95_per_class = torch.full((determine,), float("nan"), dtype=torch.float32)
        hd95_mean_excl_bg = float("nan")

    case_rows_sorted = sorted(case_rows, key=lambda r: r["dice_mean_fg"])
    dices = np.array([r["dice_mean_fg"] for r in case_rows_sorted], dtype=np.float32)
    mean_fg = float(np.mean(dices)) if dices.size > 0 else float("nan")
    std_fg = float(np.std(dices)) if dices.size > 0 else float("nan")
    thresh = mean_fg - 2.0 * std_fg if (not math.isnan(mean_fg) and not math.isnan(std_fg)) else float("-inf")

    print("\n================ TEST CASES (sorted by fg Dice) ================")
    print(f"Ensemble ckpts: {len(ckpt_paths)}")
    print(f"Case fgDice mean={mean_fg:.6f} std={std_fg:.6f} | outlier<thr={thresh:.6f}\n")
    for i, r in enumerate(case_rows_sorted):
        flag = "  <-- OUTLIER" if (r["dice_mean_fg"] < thresh) else ""
        print(f"[{i:02d}] fgDice={r['dice_mean_fg']:.6f} | {r['case_id']}{flag}")
    print("===============================================================\n")

    if writer is not None:
        writer.add_scalar(f"{prefix}/dice_mean_fg", dice_mean_fg, global_step)
        writer.add_scalar(f"{prefix}/dice_mean_incl_bg", dice_mean_incl_bg, global_step)
        if compute_hd95:
            writer.add_scalar(f"{prefix}/hd95_mean_excl_bg", hd95_mean_excl_bg, global_step)

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

    return {
        "dice_mean_incl_bg": dice_mean_incl_bg,
        "dice_mean_fg": dice_mean_fg,
        "hd95_mean_excl_bg": hd95_mean_excl_bg,
        "dice_per_class_incl_bg": dice_per_class_mean_f32,
        "hd95_per_class_excl_bg": hd95_per_class,
        "case_rows": case_rows_sorted,
        "ensemble_ckpts": ckpt_paths,
    }


def main():
    args = parse_args()

    # ----------------------------
    # Output dir
    # ----------------------------
    run_id = args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, f"btcv_run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    print("output_dir =", output_dir)

    # ----------------------------
    # Save CODE SNAPSHOT
    # ----------------------------
    save_code_snapshot(output_dir=output_dir, extra_rel_paths=list(args.snapshot_extra))

    # ----------------------------
    # MONAI temp/cache dir
    # ----------------------------
    print_config()
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print("MONAI root_dir =", root_dir)

    # ----------------------------
    # Device
    # ----------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # ----------------------------
    # Repro
    # ----------------------------
    seed_everything(args.seed)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[WARN] cudnn_benchmark=True may reduce reproducibility even with fixed seeds.")

    # ----------------------------
    # Config
    # ----------------------------
    json_path = os.path.join(args.data_dir, args.split_json)

    pixdim = tuple(args.pixdim)
    roi_size = tuple(args.roi_size)
    num_classes = int(args.num_classes)
    class_names = get_class_names(num_classes)

    # ----------------------------
    # Split priority:
    # 1) use predefined training/validation/test from json when available
    # 2) fallback to random split from pool_keys
    # ----------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        json_meta = json.load(f)

    def _has_labeled_pairs(key: str) -> bool:
        part = json_meta.get(key, [])
        return bool(part) and all(isinstance(d, dict) and d.get("image") and d.get("label") for d in part)

    if (not args.disable_json_split) and _has_labeled_pairs("training") and _has_labeled_pairs("validation") and _has_labeled_pairs("test"):
        train_files = load_decathlon_datalist(json_path, True, "training")
        val_files = load_decathlon_datalist(json_path, True, "validation")
        test_files = load_decathlon_datalist(json_path, True, "test")

        excluded_set = {_normalize_case_id(x) for x in args.exclude_cases} if args.exclude_cases else set()
        if excluded_set:
            train_files = [d for d in train_files if not _match_excluded_case(d, excluded_set)]
            val_files = [d for d in val_files if not _match_excluded_case(d, excluded_set)]
            test_files = [d for d in test_files if not _match_excluded_case(d, excluded_set)]
            print(f"[SPLIT] excluded cases={sorted(list(excluded_set))} | predefined json split filtered")
        val_source = "json_predefined"
    elif args.val_separate:
        train_files, val_files, test_files = build_custom_splits_from_json(
            json_path=json_path,
            pool_keys=args.pool_keys,
            train_num=args.train_num,
            val_num=args.val_num,
            test_num=args.test_num,
            split_seed=args.split_seed,
            exclude_cases=args.exclude_cases,
        )
        val_source = "separate_val"
    else:
        train_files, _empty_val, test_files = build_custom_splits_from_json(
            json_path=json_path,
            pool_keys=args.pool_keys,
            train_num=args.train_num,
            val_num=0,
            test_num=args.test_num,
            split_seed=args.split_seed,
            exclude_cases=args.exclude_cases,
        )
        val_files = list(test_files)
        val_source = "test_as_val"

    print("\n================ SPLIT ================")
    print(f"[POOL] keys={args.pool_keys} | split_seed={args.split_seed}")
    print(f"[EXCL] exclude_cases={args.exclude_cases if args.exclude_cases else '[] (no exclusion)'}")
    print(f"[VAL ] val_source={val_source} | val_start_iter={args.val_start_iter}")
    print(f"[SPLIT] train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")
    print("=======================================\n")

    # 路径修正
    train_files = _fix_split_pair_paths(train_files, split_name="train")
    val_files = _fix_split_pair_paths(val_files, split_name="val")
    test_files = _fix_split_pair_paths(test_files, split_name="test")

    # 显式注入稳定 case_id，不再依赖 MONAI metadata
    train_files = attach_case_ids(train_files, split_name="train")
    val_files = attach_case_ids(val_files, split_name="val")
    test_files = attach_case_ids(test_files, split_name="test")

    print("[CASE_ID][train] preview:", [d["case_id"] for d in train_files[:5]])
    print("[CASE_ID][val]   preview:", [d["case_id"] for d in val_files[:5]])
    print("[CASE_ID][test]  preview:", [d["case_id"] for d in test_files[:5]])

    # write config.json
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
                "test/dice_mean_fg (foreground only, ensemble logits avg)",
                "test/hd95_mean_excl_bg (foreground only, ensemble logits avg)",
            ],
            "sw_infer_mode": "gaussian",
            "exclude_cases": list(args.exclude_cases),
            "code_snapshot_dir": os.path.join(output_dir, "snapshot"),
            "val_source": val_source,
            "val_start_iter": int(args.val_start_iter),
            "save_ckpt_every": int(args.save_ckpt_every),
            "effective_split_sizes": {
                "train": int(len(train_files)),
                "val": int(len(val_files)),
                "test": int(len(test_files)),
            },
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ----------------------------
    # TensorBoard
    # ----------------------------
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    writer.add_text("meta/output_dir", output_dir, 0)
    writer.add_text("meta/json_path", json_path, 0)
    writer.add_text("meta/sw_infer_mode", "gaussian", 0)
    writer.add_text("meta/sw_overlap", str(args.sw_overlap), 0)
    writer.add_text("meta/topk", str(args.topk), 0)
    writer.add_text("meta/save_ckpt_every", str(args.save_ckpt_every), 0)
    writer.add_text("meta/exclude_cases", ",".join(list(args.exclude_cases)) if args.exclude_cases else "", 0)
    writer.add_text("meta/code_snapshot_dir", os.path.join(output_dir, "snapshot"), 0)
    writer.add_text("meta/val_source", val_source, 0)
    writer.add_text("meta/val_start_iter", str(args.val_start_iter), 0)

    # ----------------------------
    # Transforms
    # ----------------------------
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
                a_min=-125,
                a_max=275,
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
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.1, 0.1, 0.1),
            ),
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
                a_min=-125,
                a_max=275,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        ]
    )

    # ----------------------------
    # save lists
    # ----------------------------
    train_list_path = os.path.join(output_dir, "train_files_list.json")
    val_list_path = os.path.join(output_dir, "val_files_list.json")
    test_list_path = os.path.join(output_dir, "test_files_list.json")
    with open(train_list_path, "w") as f:
        json.dump(train_files, f, indent=2)
    with open(val_list_path, "w") as f:
        json.dump(val_files, f, indent=2)
    with open(test_list_path, "w") as f:
        json.dump(test_files, f, indent=2)

    # ----------------------------
    # DataLoader reproducibility
    # ----------------------------
    g = torch.Generator()
    g.manual_seed(args.seed)

    # ----------------------------
    # Train dataset/loader (build immediately)
    # ----------------------------
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

    # ----------------------------
    # val/test dataset/loader are lazy-built
    # ----------------------------
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

    scheduler = None
    if args.use_poly:
        base_lr = float(args.lr)

        def lr_lambda(step: int):
            t = min(max(step, 0), int(args.max_iterations))
            poly = (1.0 - t / float(args.max_iterations)) ** float(args.poly_power)
            return max(float(args.min_lr) / base_lr, poly)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    topk = max(1, int(args.topk))
    topk_dir = os.path.join(output_dir, f"top{topk}")
    os.makedirs(topk_dir, exist_ok=True)
    topk_list: List[Tuple[float, int, str]] = []

    periodic_ckpt_dir = os.path.join(output_dir, "periodic_checkpoints")
    os.makedirs(periodic_ckpt_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best.pt")
    last_resume_path = os.path.join(output_dir, "last_resume.pt")

    global_step = 0
    best_val_dice_fg = -1.0
    best_step = -1

    eval_history: List[Dict[str, Any]] = []
    num_evals = 0
    bad_evals = 0
    stop_training = False
    early_stop_reason = ""

    def maybe_update_topk(dice_fg: float, step: int) -> None:
        nonlocal topk_list
        qualifies = (len(topk_list) < topk) or (dice_fg > topk_list[-1][0])
        if not qualifies:
            return

        ckpt_path = os.path.join(topk_dir, f"iter{step:06d}_dice{dice_fg:.6f}.pt")
        torch.save(model.state_dict(), ckpt_path)

        topk_list.append((float(dice_fg), int(step), ckpt_path))
        topk_list.sort(key=lambda x: x[0], reverse=True)

        if len(topk_list) > topk:
            for _, _, p in topk_list[topk:]:
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass
            topk_list = topk_list[:topk]

        with open(os.path.join(output_dir, "topk_checkpoints.json"), "w") as f:
            json.dump(
                [{"rank": i + 1, "dice_fg": s, "iter": it, "path": p} for i, (s, it, p) in enumerate(topk_list)],
                f,
                indent=2,
            )

    def _build_full_training_checkpoint(step: int) -> Dict[str, Any]:
        ckpt = {
            "iter": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_dice_fg": float(best_val_dice_fg),
            "best_step": int(best_step),
            "eval_history": eval_history,
            "num_evals": int(num_evals),
            "bad_evals": int(bad_evals),
            "early_stop_reason": str(early_stop_reason),
            "topk_list": [(float(s), int(it), str(p)) for (s, it, p) in topk_list],
            "args": vars(args),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_generator": g.get_state(),
            },
        }
        return ckpt

    def save_periodic_checkpoint(step: int) -> None:
        ckpt_path = os.path.join(periodic_ckpt_dir, f"iter{step:06d}.pt")
        torch.save(_build_full_training_checkpoint(step), ckpt_path)
        print(f"[SAVE] periodic full checkpoint @ iter={step} -> {ckpt_path}")

        torch.save(_build_full_training_checkpoint(step), last_resume_path)
        print(f"[SAVE] last resume checkpoint updated -> {last_resume_path}")

    def try_resume_from_checkpoint(resume_path: str) -> None:
        nonlocal global_step, best_val_dice_fg, best_step
        nonlocal eval_history, num_evals, bad_evals, early_stop_reason, topk_list

        if not resume_path:
            return
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume file not found: {resume_path}")

        print(f"[RESUME] loading full checkpoint from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        if "model" not in ckpt:
            raise ValueError(
                "The resume checkpoint does not contain key 'model'. "
                "It looks like a model-only checkpoint, not a full training checkpoint."
            )

        model.load_state_dict(ckpt["model"], strict=True)

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        global_step = int(ckpt.get("iter", 0))
        best_val_dice_fg = float(ckpt.get("best_val_dice_fg", -1.0))
        best_step = int(ckpt.get("best_step", -1))

        eval_history = list(ckpt.get("eval_history", []))
        num_evals = int(ckpt.get("num_evals", 0))
        bad_evals = int(ckpt.get("bad_evals", 0))
        early_stop_reason = str(ckpt.get("early_stop_reason", ""))

        raw_topk = ckpt.get("topk_list", [])
        restored_topk: List[Tuple[float, int, str]] = []
        for item in raw_topk:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                s, it, p = item
                p = str(p)
                if os.path.isfile(p):
                    restored_topk.append((float(s), int(it), p))
        restored_topk.sort(key=lambda x: x[0], reverse=True)
        topk_list = restored_topk[:topk]

        rng_state = ckpt.get("rng_state", {})
        try:
            if "python" in rng_state and rng_state["python"] is not None:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state and rng_state["numpy"] is not None:
                np.random.set_state(rng_state["numpy"])
            if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
                torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and ("torch_cuda" in rng_state) and (rng_state["torch_cuda"] is not None):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            if "loader_generator" in rng_state and rng_state["loader_generator"] is not None:
                g.set_state(rng_state["loader_generator"])
        except Exception as e:
            print(f"[RESUME][WARN] failed to fully restore RNG state: {type(e).__name__}: {e}")

        print(
            f"[RESUME] success | iter={global_step} | "
            f"best_val_dice_fg={best_val_dice_fg:.6f} | best_step={best_step} | "
            f"num_evals={num_evals} | bad_evals={bad_evals} | topk_kept={len(topk_list)}"
        )

    if args.resume.strip():
        try_resume_from_checkpoint(args.resume.strip())

    pbar = tqdm(
        total=args.max_iterations,
        initial=global_step,
        desc="Training",
        dynamic_ncols=True,
    )
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

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            pbar.update(1)

            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(lr_now), global_step)
            pbar.set_description(f"Training iter={global_step}/{args.max_iterations} loss={loss.item():.5f}")

            if args.save_ckpt_every > 0 and (global_step % args.save_ckpt_every == 0):
                save_periodic_checkpoint(global_step)

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

                maybe_update_topk(dice_fg=dice_fg, step=global_step)

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
        torch.save(_build_full_training_checkpoint(global_step), last_resume_path)
        print(f"[SAVE] final last resume checkpoint -> {last_resume_path}")

    test_metrics: Dict[str, Any] = {}

    ckpt_paths = [p for (_, _, p) in topk_list if os.path.isfile(p)]
    if len(ckpt_paths) == 0 and os.path.isfile(best_path):
        ckpt_paths = [best_path]

    if len(ckpt_paths) > 0:
        tloader = ensure_test_loader()
        test_res = run_test_ensemble_and_print_cases(
            model=model,
            ckpt_paths=ckpt_paths,
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
        )

        test_dice_fg = float(test_res["dice_mean_fg"])
        test_hd95 = float(test_res["hd95_mean_excl_bg"])
        dice_pc = test_res["dice_per_class_incl_bg"]
        hd_pc = test_res["hd95_per_class_excl_bg"]

        print("\n================ TEST (top-k ensemble) ================")
        print(f"best_iter                 : {best_step}")
        print(f"best_val_dice_fg          : {best_val_dice_fg:.6f}")
        print(f"TEST dice_mean_fg         : {test_dice_fg:.6f}")
        print(f"TEST hd95_mean_excl_bg    : {test_hd95:.6f}")
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

        test_metrics = {
            "ensemble_ckpts": ckpt_paths,
            "ensemble_k": int(len(ckpt_paths)),
            "best_path": best_path if os.path.isfile(best_path) else "",
            "best_iter": int(best_step),
            "best_selection_metric": "val/dice_mean_fg (foreground only)",
            "best_val_dice_mean_fg": safe_float(best_val_dice_fg),
            "test_metric_definition": "dice_mean_fg = mean Dice over classes 1..C-1 (mean-over-cases per class, ignore empty)",
            "test_dice_mean_fg": safe_float(test_dice_fg),
            "test_hd95_mean_excl_bg": safe_float(test_hd95),
            "test_dice_per_class_incl_bg": tensor_to_float_list(dice_pc),
            "test_hd95_per_class_excl_bg": tensor_to_float_list(hd_pc),
            "test_case_rows_sorted_by_fg_dice": test_res["case_rows"],
            "class_names": class_names,
            "class_labels": CLASS_LABELS,
            "early_stopped": bool(stop_training),
            "early_stop_reason": early_stop_reason,
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
            "early_stop_warmup": int(args.early_stop_warmup),
            "val_start_iter": int(args.val_start_iter),
            "split": {
                "pool_keys": list(args.pool_keys),
                "train_num_arg": int(args.train_num),
                "val_num_arg": int(args.val_num),
                "test_num_arg": int(args.test_num),
                "val_separate": bool(args.val_separate),
                "val_source": val_source,
                "split_seed": int(args.split_seed),
                "exclude_cases": list(args.exclude_cases),
                "train_size": int(len(train_files)),
                "val_size": int(len(val_files)),
                "test_size": int(len(test_files)),
            },
            "sw_infer": {"mode": "gaussian", "overlap": float(args.sw_overlap)},
        }

        with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
    else:
        print(f"[WARN] No checkpoints found for testing. topk_dir={topk_dir} best_path={best_path}")

    with open(os.path.join(output_dir, "eval_history.json"), "w") as f:
        json.dump(eval_history, f, indent=2)

    writer.close()

    print("\nSaved artifacts:")
    print(" -", os.path.join(output_dir, "config.json"))
    print(" -", os.path.join(output_dir, "tb"))
    print(" -", os.path.join(output_dir, "snapshot"))
    print(" -", best_path)
    print(" -", periodic_ckpt_dir)
    print(" -", os.path.join(output_dir, "topk_checkpoints.json"))
    print(" -", os.path.join(output_dir, "test_metrics.json"))
    print(" -", os.path.join(output_dir, "eval_history.json"))
    print(" -", os.path.join(output_dir, "train_files_list.json"))
    print(" -", os.path.join(output_dir, "val_files_list.json"))
    print(" -", os.path.join(output_dir, "test_files_list.json"))


if __name__ == "__main__":
    main()