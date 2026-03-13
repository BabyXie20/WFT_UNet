import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from monai.config import print_config
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    load_decathlon_datalist,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

print_config()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"Root directory: {root_dir}")

# =========================
# 1. 路径与基础参数
# =========================
data_dir = "/path/to/dataset"
datalist_json = "/path/to/dataset_0.json"

max_iterations = 25000
eval_num = 500
roi_size = (96, 96, 96)
num_classes = 14  # 1 background + 13 organs
batch_size = 2
num_workers = 4
val_pixdim = (1.5, 1.5, 2.0)

excel_path = os.path.join(root_dir, "validation_metrics.xlsx")
best_model_path = os.path.join(root_dir, "best_metric_model.pth")

class_names = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
]

assert len(class_names) == num_classes - 1, "class_names 数量必须等于 num_classes - 1"

# =========================
# 2. NSD / Surface Dice 参数
# =========================
# 这里先给一个“可运行”的占位版本：每个前景类别阈值都设为 1.0
# 正式实验时建议按器官分别设定更合理的容忍边界误差
nsd_class_thresholds = [1.0] * (num_classes - 1)

# use_subvoxels=True 时更接近 DeepMind surface-distance 的边界定义
nsd_use_subvoxels = True

assert len(nsd_class_thresholds) == num_classes - 1, (
    "当 include_background=False 时，"
    "nsd_class_thresholds 长度必须等于 num_classes - 1"
)

# =========================
# 3. 数据变换
# =========================
def get_train_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=val_pixdim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=val_pixdim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),
        ]
    )


# =========================
# 4. 数据集与 DataLoader
# =========================
def get_dataloaders():
    train_files = load_decathlon_datalist(
        datalist_json,
        is_segmentation=True,
        data_list_key="training",
        base_dir=data_dir,
    )
    val_files = load_decathlon_datalist(
        datalist_json,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=data_dir,
    )

    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_num=min(24, len(train_files)),
        cache_rate=1.0,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_num=min(6, len(val_files)),
        cache_rate=1.0,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, val_ds, train_loader, val_loader


def visualize_val_case(val_ds, case_num=0):
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }

    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]

    print(f"image shape: {img.shape}, label shape: {label.shape}")

    if img_name not in slice_map:
        print(f"{img_name} 不在 slice_map 中，跳过可视化。")
        return

    slice_idx = slice_map[img_name]

    plt.figure("image_label", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, slice_idx].detach().cpu(), cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, slice_idx].detach().cpu())

    plt.show()


# =========================
# 5. 模型、损失、优化器
# =========================
def build_model():
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=roi_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
    return model


model = build_model()
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)


# =========================
# 6. 工具函数
# =========================
def metric_tensor_to_numpy(metric_value):
    """
    MONAI aggregate() 有时可能返回 tensor / tuple / list。
    这里统一转为 1D numpy，并把 inf/-inf 变成 nan，便于后续 nanmean 聚合。
    """
    if isinstance(metric_value, (tuple, list)):
        metric_value = metric_value[0]

    if isinstance(metric_value, torch.Tensor):
        metric_value = metric_value.detach().cpu().float().numpy()

    metric_value = np.asarray(metric_value, dtype=np.float32)
    metric_value = np.squeeze(metric_value)
    metric_value[~np.isfinite(metric_value)] = np.nan
    return metric_value


def get_batch_filenames(batch, default_start_idx=0):
    batch_size_local = int(batch["image"].shape[0])
    filenames = None

    # 优先从 MetaTensor 取
    meta = getattr(batch["image"], "meta", None)
    if isinstance(meta, dict) and "filename_or_obj" in meta:
        raw = meta["filename_or_obj"]
        if isinstance(raw, (list, tuple)):
            filenames = [os.path.basename(str(x)) for x in raw]
        else:
            filenames = [os.path.basename(str(raw))]

    # 兼容旧版 meta_dict
    if (not filenames) and ("image_meta_dict" in batch) and ("filename_or_obj" in batch["image_meta_dict"]):
        raw = batch["image_meta_dict"]["filename_or_obj"]
        if isinstance(raw, (list, tuple)):
            filenames = [os.path.basename(str(x)) for x in raw]
        else:
            filenames = [os.path.basename(str(raw))]

    if (not filenames) or (len(filenames) != batch_size_local):
        filenames = [f"case_{default_start_idx + i:04d}" for i in range(batch_size_local)]

    return filenames


def summarize_validation_metrics(dice_case_matrix, hd95_case_matrix, nsd_case_matrix):
    """
    dice_case_matrix: [N_cases, N_organs]
    hd95_case_matrix: [N_cases, N_organs]
    nsd_case_matrix:  [N_cases, N_organs]
    """
    mean_dice = float(np.nanmean(dice_case_matrix))
    mean_hd95 = float(np.nanmean(hd95_case_matrix))
    mean_nsd = float(np.nanmean(nsd_case_matrix))

    per_class_mean_dice = np.nanmean(dice_case_matrix, axis=0)
    per_class_mean_hd95 = np.nanmean(hd95_case_matrix, axis=0)
    per_class_mean_nsd = np.nanmean(nsd_case_matrix, axis=0)

    per_class_df = pd.DataFrame(
        {
            "class_index": list(range(1, len(class_names) + 1)),
            "class_name": class_names,
            "dice": per_class_mean_dice.astype(np.float32),
            "hd95": per_class_mean_hd95.astype(np.float32),
            "nsd": per_class_mean_nsd.astype(np.float32),
        }
    )

    summary = {
        "mean_dice": mean_dice,
        "mean_hd95": mean_hd95,
        "mean_nsd": mean_nsd,
    }
    return summary, per_class_df


def print_val_metrics(summary, per_class_df):
    print("\n========== Validation Metrics ==========")
    print(f"Mean Dice (all organs over val_ds): {summary['mean_dice']:.4f}")
    print(f"Mean HD95 (all organs over val_ds): {summary['mean_hd95']:.4f}")
    print(f"Mean NSD  (all organs over val_ds): {summary['mean_nsd']:.4f}")
    print("\nPer-class metrics:")
    for _, row in per_class_df.iterrows():
        print(
            f"[{int(row['class_index']):02d}] "
            f"{row['class_name']:30s} "
            f"Dice: {row['dice']:.4f} | "
            f"HD95: {row['hd95']:.4f} | "
            f"NSD: {row['nsd']:.4f}"
        )
    print("========================================\n")


def save_metrics_to_excel(
    excel_file,
    overall_history,
    per_class_history,
    best_summary=None,
    best_per_class_df=None,
    best_case_df=None,
):
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pd.DataFrame(overall_history).to_excel(
            writer,
            sheet_name="val_overall_history",
            index=False,
        )

        pd.DataFrame(per_class_history).to_excel(
            writer,
            sheet_name="val_per_class_history",
            index=False,
        )

        if best_summary is not None:
            pd.DataFrame([best_summary]).to_excel(
                writer,
                sheet_name="best_model_summary",
                index=False,
            )

        if best_per_class_df is not None:
            best_per_class_df.to_excel(
                writer,
                sheet_name="best_model_per_class",
                index=False,
            )

        if best_case_df is not None:
            best_case_df.to_excel(
                writer,
                sheet_name="best_model_case_metrics",
                index=False,
            )


# =========================
# 7. 验证：Dice + HD95 + NSD
# =========================
def evaluate_val_dataset(model, val_loader, global_step=None):
    """
    用 MONAI 的 DiceMetric / HausdorffDistanceMetric / SurfaceDiceMetric
    按 case 计算，然后对整个 val_ds 聚合。
    """
    model.eval()

    dice_case_list = []
    hd95_case_list = []
    nsd_case_list = []
    case_records = []
    case_counter = 0

    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc="Validate", dynamic_ncols=True)

        for batch in val_iterator:
            case_names = get_batch_filenames(batch, default_start_idx=case_counter)

            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)

            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=model,
            )

            val_labels_list = decollate_batch(val_labels)
            val_outputs_list = decollate_batch(val_outputs)

            for local_idx, (val_label_tensor, val_pred_tensor) in enumerate(
                zip(val_labels_list, val_outputs_list)
            ):
                # 转 one-hot，形状为 [C, H, W, D]
                true_case = post_label(val_label_tensor)
                pred_case = post_pred(val_pred_tensor)

                # Dice
                tmp_dice = DiceMetric(
                    include_background=False,
                    reduction="none",
                    get_not_nans=False,
                    ignore_empty=True,
                )
                tmp_dice(y_pred=[pred_case], y=[true_case])
                dice_case = metric_tensor_to_numpy(tmp_dice.aggregate())
                tmp_dice.reset()

                # HD95
                tmp_hd95 = HausdorffDistanceMetric(
                    include_background=False,
                    reduction="none",
                    percentile=95,
                    get_not_nans=False,
                )
                tmp_hd95(y_pred=[pred_case], y=[true_case], spacing=val_pixdim)
                hd95_case = metric_tensor_to_numpy(tmp_hd95.aggregate())
                tmp_hd95.reset()

                # NSD / Surface Dice
                # SurfaceDiceMetric 需要 batch-first one-hot tensor: [B, C, H, W, D]
                tmp_nsd = SurfaceDiceMetric(
                    class_thresholds=nsd_class_thresholds,
                    include_background=False,
                    distance_metric="euclidean",
                    reduction="none",
                    get_not_nans=False,
                    use_subvoxels=nsd_use_subvoxels,
                )
                tmp_nsd(
                    y_pred=pred_case.unsqueeze(0),
                    y=true_case.unsqueeze(0),
                    spacing=val_pixdim,
                )
                nsd_case = metric_tensor_to_numpy(tmp_nsd.aggregate())
                tmp_nsd.reset()

                dice_case_list.append(dice_case)
                hd95_case_list.append(hd95_case)
                nsd_case_list.append(nsd_case)

                record = {
                    "global_step": global_step if global_step is not None else -1,
                    "case_name": case_names[local_idx] if local_idx < len(case_names) else f"case_{case_counter:04d}",
                    "mean_dice": float(np.nanmean(dice_case)),
                    "mean_hd95": float(np.nanmean(hd95_case)),
                    "mean_nsd": float(np.nanmean(nsd_case)),
                }

                for i, organ_name in enumerate(class_names):
                    record[f"dice_{organ_name}"] = float(dice_case[i]) if i < len(dice_case) else np.nan
                    record[f"hd95_{organ_name}"] = float(hd95_case[i]) if i < len(hd95_case) else np.nan
                    record[f"nsd_{organ_name}"] = float(nsd_case[i]) if i < len(nsd_case) else np.nan

                case_records.append(record)
                case_counter += 1

            if global_step is None:
                val_iterator.set_description("Validate")
            else:
                val_iterator.set_description(f"Validate (step={global_step}/{max_iterations})")

    dice_case_matrix = np.vstack(dice_case_list).astype(np.float32)
    hd95_case_matrix = np.vstack(hd95_case_list).astype(np.float32)
    nsd_case_matrix = np.vstack(nsd_case_list).astype(np.float32)

    summary, per_class_df = summarize_validation_metrics(
        dice_case_matrix=dice_case_matrix,
        hd95_case_matrix=hd95_case_matrix,
        nsd_case_matrix=nsd_case_matrix,
    )
    case_df = pd.DataFrame(case_records)

    return summary, per_class_df, case_df


# =========================
# 8. 训练主循环
# =========================
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_function,
    max_iterations,
    eval_num,
    excel_file,
):
    global_step = 0
    dice_val_best = -1.0
    global_step_best = 0

    epoch_loss_values = []
    overall_history = []
    per_class_history = []

    best_summary_cache = None
    best_per_class_df_cache = None
    best_case_df_cache = None

    while global_step < max_iterations:
        model.train()
        epoch_loss = 0.0
        step_in_epoch = 0

        train_iterator = tqdm(
            train_loader,
            desc="Training",
            dynamic_ncols=True,
        )

        for batch in train_iterator:
            step_in_epoch += 1

            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            global_step += 1

            train_iterator.set_description(
                f"Training (step={global_step}/{max_iterations}) (loss={loss.item():.5f})"
            )

            if (global_step % eval_num == 0) or (global_step == max_iterations):
                mean_epoch_loss = epoch_loss / step_in_epoch
                epoch_loss_values.append(mean_epoch_loss)

                val_summary, val_per_class_df, val_case_df = evaluate_val_dataset(
                    model=model,
                    val_loader=val_loader,
                    global_step=global_step,
                )

                overall_history.append(
                    {
                        "global_step": global_step,
                        "train_loss_avg_since_epoch_start": mean_epoch_loss,
                        "mean_dice": val_summary["mean_dice"],
                        "mean_hd95": val_summary["mean_hd95"],
                        "mean_nsd": val_summary["mean_nsd"],
                    }
                )

                val_per_class_with_step = val_per_class_df.copy()
                val_per_class_with_step.insert(0, "global_step", global_step)
                per_class_history.extend(val_per_class_with_step.to_dict("records"))

                print_val_metrics(val_summary, val_per_class_df)

                if val_summary["mean_dice"] > dice_val_best:
                    dice_val_best = val_summary["mean_dice"]
                    global_step_best = global_step
                    torch.save(model.state_dict(), best_model_path)

                    best_summary_cache = {
                        "selected_by": "mean_dice",
                        "global_step": global_step,
                        "mean_dice": val_summary["mean_dice"],
                        "mean_hd95": val_summary["mean_hd95"],
                        "mean_nsd": val_summary["mean_nsd"],
                    }
                    best_per_class_df_cache = val_per_class_df.copy()
                    best_case_df_cache = val_case_df.copy()

                    print(
                        f"Model was saved. "
                        f"Best Mean Dice: {dice_val_best:.4f}, "
                        f"Current Mean HD95: {val_summary['mean_hd95']:.4f}, "
                        f"Current Mean NSD: {val_summary['mean_nsd']:.4f}"
                    )
                else:
                    print(
                        f"Model was not saved. "
                        f"Best Mean Dice: {dice_val_best:.4f}, "
                        f"Current Mean Dice: {val_summary['mean_dice']:.4f}, "
                        f"Current Mean HD95: {val_summary['mean_hd95']:.4f}, "
                        f"Current Mean NSD: {val_summary['mean_nsd']:.4f}"
                    )

                save_metrics_to_excel(
                    excel_file=excel_file,
                    overall_history=overall_history,
                    per_class_history=per_class_history,
                    best_summary=best_summary_cache,
                    best_per_class_df=best_per_class_df_cache,
                    best_case_df=best_case_df_cache,
                )
                print(f"Excel updated: {excel_file}")

            if global_step >= max_iterations:
                break

    print(f"Training finished. Best metric: {dice_val_best:.4f} at step {global_step_best}")
    return (
        epoch_loss_values,
        overall_history,
        per_class_history,
        dice_val_best,
        global_step_best,
    )


# =========================
# 9. main
# =========================
def main():
    train_ds, val_ds, train_loader, val_loader = get_dataloaders()

    # 可选：查看一个验证样本
    # visualize_val_case(val_ds, case_num=0)

    (
        epoch_loss_values,
        overall_history,
        per_class_history,
        dice_val_best,
        global_step_best,
    ) = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        max_iterations=max_iterations,
        eval_num=eval_num,
        excel_file=excel_path,
    )

    # 重新加载最佳模型，再做一次完整验证，确保 Excel 中 best_model_* 对应 checkpoint 本身
    if Path(best_model_path).exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from: {best_model_path}")

        best_summary, best_per_class_df, best_case_df = evaluate_val_dataset(
            model=model,
            val_loader=val_loader,
            global_step=global_step_best,
        )
        best_summary = {
            "selected_by": "mean_dice",
            "global_step": global_step_best,
            "mean_dice": best_summary["mean_dice"],
            "mean_hd95": best_summary["mean_hd95"],
            "mean_nsd": best_summary["mean_nsd"],
        }

        save_metrics_to_excel(
            excel_file=excel_path,
            overall_history=overall_history,
            per_class_history=per_class_history,
            best_summary=best_summary,
            best_per_class_df=best_per_class_df,
            best_case_df=best_case_df,
        )

        print("\nBest model validation metrics:")
        print_val_metrics(best_summary, best_per_class_df)
        print(f"Final Excel saved to: {excel_path}")
    else:
        print("No saved best model found.")

    return {
        "excel_path": excel_path,
        "best_model_path": best_model_path,
        "best_mean_dice": dice_val_best,
        "best_global_step": global_step_best,
    }


if __name__ == "__main__":
    main()