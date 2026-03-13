import os
import json
import math
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from monai.config import print_config
from monai.utils import set_determinism
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

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

#python eval_best_btcv.py --run_dir ./outputs/btcv_run_20260310_120000
def parse_args():
    parser = argparse.ArgumentParser("BTCV best.pt sliding-window evaluation on test_files_list.json")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="训练输出目录，例如 ./outputs/btcv_run_20260310_120000",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="",
        help="测试集列表 json 路径；为空时默认使用 <run_dir>/test_files_list.json",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="checkpoint 路径；为空时默认使用 <run_dir>/best.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="config.json 路径；为空时默认使用 <run_dir>/config.json",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="覆盖 config 中的 num_workers_test")
    parser.add_argument("--cache_num", type=int, default=None, help="覆盖 config 中的 cache_num_test")
    parser.add_argument("--cache_rate", type=float, default=None, help="覆盖 config 中的 cache_rate")
    parser.add_argument("--device", type=str, default="", help="例如 cuda, cuda:0, cpu；为空则自动选择")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 config 中的 seed")
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="结果保存路径；为空时默认使用 <run_dir>/test_metrics_best_only.json",
    )
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
        names.append(CLASS_LABELS.get(str(i), f"class_{i}"))
    return names


def _strip_known_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _extract_case_id_from_path(path: str) -> str:
    base = os.path.basename(str(path))
    digits4 = []
    import re

    digits4 = re.findall(r"(?<!\d)(\d{4})(?!\d)", base)
    if len(digits4) > 0:
        return digits4[0]
    return _strip_known_ext(base)


def _get_case_id(batch: Dict[str, Any]) -> str:
    fn = batch["image"].meta["filename_or_obj"]
    if isinstance(fn, (list, tuple)):
        fn = fn[0]
    return _extract_case_id_from_path(fn)


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
    compute_hd95: bool = True,
) -> Dict[str, Any]:
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

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

    pbar = tqdm(loader, desc="TEST(best.pt)", dynamic_ncols=True)
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
        hd95_per_class = torch.full((max(num_classes - 1, 0),), float("nan"), dtype=torch.float32)
        hd95_mean_excl_bg = float("nan")

    case_rows_sorted = sorted(case_rows, key=lambda r: r["dice_mean_fg"])
    dices = np.array([r["dice_mean_fg"] for r in case_rows_sorted], dtype=np.float32)
    mean_fg = float(np.mean(dices)) if dices.size > 0 else float("nan")
    std_fg = float(np.std(dices)) if dices.size > 0 else float("nan")
    thresh = mean_fg - 2.0 * std_fg if (not math.isnan(mean_fg) and not math.isnan(std_fg)) else float("-inf")

    print("\n================ TEST CASES (sorted by fg Dice) ================")
    print("Checkpoint: 1 (best.pt)")
    print(f"Case fgDice mean={mean_fg:.6f} std={std_fg:.6f} | outlier<thr={thresh:.6f}\n")
    for i, r in enumerate(case_rows_sorted):
        flag = "  <-- OUTLIER" if r["dice_mean_fg"] < thresh else ""
        print(f"[{i:02d}] fgDice={r['dice_mean_fg']:.6f} | {r['case_id']}{flag}")
    print("===============================================================\n")

    print("Per-class Dice (incl bg, mean-over-cases):")
    for c in range(num_classes):
        v = float(dice_per_class_mean_f32[c].item()) if not torch.isnan(dice_per_class_mean_f32[c]) else float("nan")
        print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

    print("\nPer-class HD95 (excl bg):")
    for c in range(1, num_classes):
        idx = c - 1
        if idx < hd95_per_class.numel():
            v = float(hd95_per_class[idx].item())
            print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

    return {
        "dice_mean_incl_bg": dice_mean_incl_bg,
        "dice_mean_fg": dice_mean_fg,
        "hd95_mean_excl_bg": hd95_mean_excl_bg,
        "dice_per_class_incl_bg": dice_per_class_mean_f32,
        "hd95_per_class_excl_bg": hd95_per_class,
        "case_rows": case_rows_sorted,
        "ckpt_path": ckpt_path,
    }


def build_eval_transforms(pixdim: Tuple[float, float, float]):
    return Compose(
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


def main():
    args = parse_args()

    run_dir = os.path.abspath(args.run_dir)
    test_list_path = os.path.abspath(args.test_list) if args.test_list else os.path.join(run_dir, "test_files_list.json")
    ckpt_path = os.path.abspath(args.ckpt) if args.ckpt else os.path.join(run_dir, "best.pt")
    config_path = os.path.abspath(args.config) if args.config else os.path.join(run_dir, "config.json")
    save_json_path = os.path.abspath(args.save_json) if args.save_json else os.path.join(run_dir, "test_metrics_best_only.json")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")
    if not os.path.isfile(test_list_path):
        raise FileNotFoundError(f"test_files_list.json not found: {test_list_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"best.pt not found: {ckpt_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(test_list_path, "r", encoding="utf-8") as f:
        test_files = json.load(f)

    print_config()

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 123))
    seed_everything(seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    pixdim = tuple(cfg.get("pixdim", [1.5, 1.5, 2.0]))
    roi_size = tuple(cfg.get("roi_size", [96, 96, 96]))
    num_classes = int(cfg.get("num_classes", 14))
    sw_batch_size = int(cfg.get("sw_batch_size", 2))
    sw_overlap = float(cfg.get("sw_overlap", 0.5))
    cache_rate = float(args.cache_rate if args.cache_rate is not None else cfg.get("cache_rate", 1.0))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers_test", 6))
    cache_num_cfg = int(cfg.get("cache_num_test", len(test_files)))
    cache_num = int(args.cache_num if args.cache_num is not None else cache_num_cfg)
    cache_num = len(test_files) if cache_num <= 0 else min(cache_num, len(test_files))

    class_names = get_class_names(num_classes)

    print("\n================ EVAL CONFIG ================")
    print(f"run_dir                  : {run_dir}")
    print(f"config                   : {config_path}")
    print(f"test_list                : {test_list_path}")
    print(f"checkpoint               : {ckpt_path}")
    print(f"num_test_cases           : {len(test_files)}")
    print(f"pixdim                   : {pixdim}")
    print(f"roi_size                 : {roi_size}")
    print(f"num_classes              : {num_classes}")
    print(f"sw_batch_size            : {sw_batch_size}")
    print(f"sw_overlap               : {sw_overlap}")
    print(f"cache_num                : {cache_num}")
    print(f"cache_rate               : {cache_rate}")
    print(f"num_workers              : {num_workers}")
    print("=============================================\n")

    eval_transforms = build_eval_transforms(pixdim=pixdim)

    g = torch.Generator()
    g.manual_seed(seed)

    test_ds = CacheDataset(
        data=test_files,
        transform=eval_transforms,
        cache_num=cache_num,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = VNet(n_channels=1, n_classes=num_classes).to(device)

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    test_res = run_test_single_model_and_print_cases(
        model=model,
        ckpt_path=ckpt_path,
        loader=test_loader,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        sw_overlap=sw_overlap,
        num_classes=num_classes,
        class_names=class_names,
        post_label=post_label,
        post_pred=post_pred,
        compute_hd95=True,
    )

    test_dice_fg = float(test_res["dice_mean_fg"])
    test_hd95 = float(test_res["hd95_mean_excl_bg"])

    print("\n================ TEST (best.pt only) ================")
    print(f"TEST dice_mean_fg         : {test_dice_fg:.6f}")
    print(f"TEST hd95_mean_excl_bg    : {test_hd95:.6f}")
    print("=====================================================\n")

    result = {
        "eval_mode": "single_best_checkpoint",
        "checkpoint": ckpt_path,
        "run_dir": run_dir,
        "config_path": config_path,
        "test_list_path": test_list_path,
        "num_test_cases": int(len(test_files)),
        "seed": int(seed),
        "device": str(device),
        "metric_definition": "dice_mean_fg = mean Dice over classes 1..C-1 (mean-over-cases per class, ignore empty)",
        "sw_infer": {
            "mode": "gaussian",
            "roi_size": list(roi_size),
            "sw_batch_size": int(sw_batch_size),
            "overlap": float(sw_overlap),
        },
        "preprocess": {
            "pixdim": list(pixdim),
            "orientation": "RAS",
            "intensity": {"a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1.0, "clip": True},
            "crop_foreground": True,
        },
        "test_dice_mean_incl_bg": safe_float(float(test_res["dice_mean_incl_bg"])),
        "test_dice_mean_fg": safe_float(test_dice_fg),
        "test_hd95_mean_excl_bg": safe_float(test_hd95),
        "test_dice_per_class_incl_bg": tensor_to_float_list(test_res["dice_per_class_incl_bg"]),
        "test_hd95_per_class_excl_bg": tensor_to_float_list(test_res["hd95_per_class_excl_bg"]),
        "test_case_rows_sorted_by_fg_dice": test_res["case_rows"],
        "class_names": class_names,
        "class_labels": CLASS_LABELS,
    }

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"saved json: {save_json_path}")


if __name__ == "__main__":
    main()
