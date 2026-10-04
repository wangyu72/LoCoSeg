import csv
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import PatchesMultiLabelDataset, AGGC2022Dataset
try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from utils import (
    calculate_dice_coefficient,
    calculate_iou,
    calculate_accuracy,
    compute_batch_metrics_macro,
    DC_and_Focal_loss,
)


try:
    from network.SelfRegUNet.selfreg_unet import SelfRegKDLoss
except ImportError:
    SelfRegKDLoss = None
    logging.warning("SelfRegKDLoss not available. SelfRegUNet will use standard loss only.")


def get_base_model(model):

    return model.module if isinstance(model, torch.nn.DataParallel) else model




def get_dataloader(args, split="train", batch_size=1, shuffle=False):

    base_seed = args.seed

    generator = torch.Generator()
    generator.manual_seed(base_seed)

    target_size = (args.img_size, args.img_size) if args.img_size is not None else None
    dataset_label_folders = [f for f in args.label_folders if f.lower() != "background"]
    if args.dataset == 'aggc2022':
        DatasetClass = AGGC2022Dataset
        logging.info("Using AGGC2022Dataset (Nested structure: masks/g3/...)")
    else:
        DatasetClass = PatchesMultiLabelDataset
        logging.info("Using PatchesMultiLabelDataset (Flat structure: masks_g3/...)")

    db_data = DatasetClass(
        list_dir=args.list_dir,
        split=split,
        label_folders=dataset_label_folders,
        target_size=target_size,
    )

    logging.info(
        "The length of {} {} set is: {}".format(args.dataset, split, len(db_data))
    )

    def worker_init_fn(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)

    dataloader = DataLoader(
        db_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        prefetch_factor=2,
    )
    return dataloader





def save_loss_curve(train_losses, val_metrics_dict, save_path):

    if plt is None:
        logging.warning("matplotlib is not available; skipping loss curve plotting.")
        return
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", marker="o", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = [
        "tab:green", "tab:orange", "tab:brown", "tab:red"
    ]
    metric_names = ["dice", "iou", "sensitivity", "mcc"]

    for idx, metric_name in enumerate(metric_names):
        if metric_name in val_metrics_dict:
            vals = val_metrics_dict[metric_name]
            valid_epochs = [
                ep + 1 for ep, v in enumerate(vals) if v is not None and not np.isnan(v)
            ]
            valid_vals = [v for v in vals if v is not None and not np.isnan(v)]
            if valid_vals:
                ax2.plot(
                    valid_epochs,
                    valid_vals,
                    label=f"Val {metric_name.capitalize()}",
                    marker="s",
                    color=colors[idx % len(colors)],
                    linewidth=1.5,
                )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Validation Metrics")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Training Progress", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def validate_wsi(args, model, valloader, loss_fn):

    if len(valloader) == 0:
        logging.warning("Validation loader is empty.")
        return {}, {}, np.nan, {}

    logging.info("%d val iterations per epoch", len(valloader))
    model.eval()

    global_tp = np.zeros(args.num_classes)
    global_fp = np.zeros(args.num_classes)
    global_fn = np.zeros(args.num_classes)
    global_tn = np.zeros(args.num_classes)

    total_loss = 0.0
    batches = 0

    for sampled_batch in tqdm(valloader, total=len(valloader), desc="Validating", leave=False):
        image = sampled_batch["image"].cuda()
        label = sampled_batch["mask"].cuda().float()

        with torch.no_grad():
            if args.network in ("DHUnet", "mynet"):
                raw_outputs = model(image, image)
            else:
                raw_outputs = model(image)


            if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2:
                first_elem = raw_outputs[0]
                if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                    if isinstance(first_elem[0], torch.Tensor):
                        raw_outputs = first_elem

            if isinstance(raw_outputs, torch.Tensor):
                primary_output = raw_outputs

            elif isinstance(raw_outputs, (list, tuple)):
                first = raw_outputs[0]

                if isinstance(first, (list, tuple)):
                    primary_output = None
                    for cand in first:
                        if isinstance(cand, torch.Tensor) and cand.dim() == 4 and cand.shape[2:] == label.shape[2:]:
                            primary_output = cand
                            break
                    if primary_output is None:
                        for cand in first:
                            if isinstance(cand, torch.Tensor) and cand.dim() == 4:
                                primary_output = cand
                                break
                        else:
                            raise RuntimeError("Could not find tensor output in model's first element to use as primary_output")
                else:
                    if isinstance(first, torch.Tensor):
                        primary_output = first
                    else:
                        raise RuntimeError("Unexpected model output structure: first element is not a tensor or list of tensors")
            else:
                raise RuntimeError("Unexpected model output type: %s" % type(raw_outputs))

            annotation_valid = sampled_batch.get('annotation_valid', None)
            if annotation_valid is not None and loss_fn is not None:
                loss_fn._annotation_valid = annotation_valid.cuda()
            elif loss_fn is not None:
                loss_fn._annotation_valid = None

            if loss_fn is not None:
                loss = loss_fn(primary_output, label)
            else:
                loss = torch.tensor(0.0)

        total_loss += loss.item()
        batches += 1

        batch_annotation_valid = annotation_valid.cuda() if annotation_valid is not None else None
        batch_metrics = compute_batch_metrics_macro(
            primary_output, label,
            threshold=args.thr,
            annotation_valid=batch_annotation_valid,
        )

        for c in range(args.num_classes):
            global_tp[c] += batch_metrics['tp_per_class'][c]
            global_fp[c] += batch_metrics['fp_per_class'][c]
            global_fn[c] += batch_metrics['fn_per_class'][c]
            global_tn[c] += batch_metrics['tn_per_class'][c]

    final_dice_per_class = []
    final_iou_per_class = []
    final_sensitivity_per_class = []
    final_mcc_per_class = []

    eps = 1e-5

    for c in range(args.num_classes):
        tp = global_tp[c]
        fp = global_fp[c]
        fn = global_fn[c]
        tn = global_tn[c]

        total_pixels = tp + fp + fn + tn
        if total_pixels == 0:
            final_dice_per_class.append(float('nan'))
            final_iou_per_class.append(float('nan'))
            final_sensitivity_per_class.append(float('nan'))
            final_mcc_per_class.append(float('nan'))
        else:
            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            sensitivity = (tp + eps) / (tp + fn + eps)
            denom_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc = (tp * tn - fp * fn) / (np.sqrt(denom_mcc) + eps) if denom_mcc > 0 else 0.0
            final_dice_per_class.append(dice)
            final_iou_per_class.append(iou)
            final_sensitivity_per_class.append(sensitivity)
            final_mcc_per_class.append(mcc)

    macro_dice = np.nanmean(final_dice_per_class)
    macro_iou = np.nanmean(final_iou_per_class)
    macro_sensitivity = np.nanmean(final_sensitivity_per_class)
    macro_mcc = np.nanmean(final_mcc_per_class)

    macro = {
        "mDice": macro_dice,
        "mIoU": macro_iou,
        "mSensitivity": macro_sensitivity,
        "mMCC": macro_mcc,
    }

    per_class = {
        "dice": final_dice_per_class,
        "iou": final_iou_per_class,
        "sensitivity": final_sensitivity_per_class,
        "mcc": final_mcc_per_class,
    }

    micro = {}
    val_loss = total_loss / batches if batches > 0 else np.nan

    label_names = args.label_folders or [f"label_{i}" for i in range(args.num_classes)]
    for idx in range(len(label_names)):
        label_name = label_names[idx] if idx < len(label_names) else f"label_{idx}"
        logging.info(
            "%s: Dice %.4f | IoU %.4f | Sens %.4f | MCC %.4f",
            label_name,
            per_class["dice"][idx],
            per_class["iou"][idx],
            per_class["sensitivity"][idx],
            per_class["mcc"][idx]
        )

    logging.info(
        "[Macro Average] mDice: %.4f | mIoU: %.4f | mSensitivity: %.4f | mMCC: %.4f",
        macro["mDice"], macro["mIoU"], macro["mSensitivity"], macro["mMCC"]
    )

    return macro, micro, val_loss, per_class



def log_split_metrics(split_name, epoch, label_names, per_class, macro, micro, save_dir):

    import csv
    csv_path = os.path.join(save_dir, f"{split_name}_metrics.csv")

    row_data = {"epoch": epoch}

    for idx, label_name in enumerate(label_names):
        for metric_key, values in per_class.items():
            if idx < len(values):
                row_data[f"{label_name}_{metric_key}"] = values[idx]

    for key, value in macro.items():
        if isinstance(value, (int, float)):
            row_data[f"macro_{key}"] = value

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def train_with_val_and_test(args, model, snapshot_path, config=None):

    base_lr = args.base_lr

    optimizer = optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=0.01
    )

    trainloader = get_dataloader(args, split="train", batch_size=args.batch_size, shuffle=True)
    valloader = get_dataloader(args, split="val", batch_size=args.val_batch_size, shuffle=False)
    testloader = get_dataloader(args, split="test", batch_size=args.val_batch_size, shuffle=False)

    if len(trainloader) == 0:
        raise RuntimeError("Training loader is empty, cannot proceed.")

    patience = getattr(args, "patience", 20)
    min_best_epoch = max(2, int(getattr(args, "min_best_epoch", 2)))
    test_every_epoch = int(getattr(args, "test_every_epoch", 1)) > 0

    train_losses = []
    val_losses = []
    val_metrics = {k: [] for k in ["dice", "iou", "sensitivity", "mcc"]}

    logging.info(f"Training config: Patience {patience} epochs, min_best_epoch {min_best_epoch}, test_every_epoch {test_every_epoch}")

    top_k = 3
    best_list = []
    epochs_no_improve = 0
    last_model_path = os.path.join(snapshot_path, "last_epoch.pth")

    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)

    logging.info(f"Start training: {args.max_epochs} epochs, {len(trainloader)} iter/epoch, Total: {max_iterations}")

    label_names = args.label_folders or [f"label_{i}" for i in range(args.num_classes)]

    use_selfreg = (args.network == "SelfRegUNet")
    kd_loss_fn = None
    if use_selfreg and SelfRegKDLoss is not None:
        kd_loss_fn = SelfRegKDLoss(lambda_x=0.01).cuda()



    for epoch_idx in range(args.max_epochs):
        loss_fn = DC_and_Focal_loss(batch_dice=True, smooth=1.0, dice_weight=1.0, focal_weight=0.5,
                                    alpha=0.25, gamma=2.0).cuda()
        loss_fn._annotation_valid = None

        model.train()
        epoch_loss_sum = 0.0

        iterator = tqdm(trainloader, ncols=100, desc=f"Epoch {epoch_idx+1}/{args.max_epochs}")

        for sampled_batch in iterator:
            image_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["mask"].cuda().float()

            annotation_valid = sampled_batch.get('annotation_valid', None)
            if annotation_valid is not None:
                loss_fn._annotation_valid = annotation_valid.cuda()
            else:
                loss_fn._annotation_valid = None

            optimizer.zero_grad()

            if args.network in ("ViTexSwin", "ViTexConvNeXt"):
                outputs = model(image_batch)

                enable_cls = getattr(args, 'enable_classification', False)

                if enable_cls and isinstance(outputs, tuple) and len(outputs) == 2:
                    seg_outputs, cls_logits = outputs

                    if len(seg_outputs) == 5:
                        outputs0, outputs1, outputs2, outputs3, outputs4 = seg_outputs

                        loss0 = loss_fn(outputs0, label_batch)
                        loss1 = loss_fn(outputs1, label_batch)
                        loss2 = loss_fn(outputs2, label_batch)
                        loss3 = loss_fn(outputs3, label_batch)
                        loss4 = loss_fn(outputs4, label_batch)
                        seg_loss = 0.6*loss0 + 0.2*loss2 + 0.2*loss4
                    else:
                        seg_loss = loss_fn(seg_outputs[0], label_batch)

                    B, C, H, W = label_batch.shape

                    pixel_counts = label_batch.view(B, C, -1).sum(dim=2)
                    y_classification = (pixel_counts > 20).float()
                    cls_loss_fn = torch.nn.BCEWithLogitsLoss()
                    cls_loss = cls_loss_fn(cls_logits, y_classification)

                    cls_weight = getattr(args, 'cls_loss_weight', 0.2)
                    loss = seg_loss + cls_weight * cls_loss
                else:
                    if len(outputs) == 5:
                        outputs0, outputs1, outputs2, outputs3, outputs4 = outputs
                        loss0 = loss_fn(outputs0, label_batch)
                        loss1 = loss_fn(outputs1, label_batch)
                        loss2 = loss_fn(outputs2, label_batch)
                        loss3 = loss_fn(outputs3, label_batch)
                        loss4 = loss_fn(outputs4, label_batch)
                        loss = 0.6*loss0 + 0.2*loss2 + 0.2*loss4
                    else:
                        loss = loss_fn(outputs[0], label_batch)

            elif args.network in "DHUnet":
                outputs = model(image_batch)
                if isinstance(outputs, (list, tuple)):
                    loss = 0
                    for o in outputs:
                        loss += loss_fn(o, label_batch)
                else:
                    loss = loss_fn(outputs, label_batch)
            elif args.network == "ConDSeg":
                outputs = model(image_batch)
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    mask_pred, fg_pred, bg_pred, uc_pred = outputs
                    loss_mask = loss_fn(mask_pred, label_batch)

                    condseg_cfg = getattr(config.MODEL, 'CONDSEG', None) if config else None
                    use_aux = bool(getattr(condseg_cfg, 'AUX_LOSS', True)) if condseg_cfg else True

                    if use_aux:
                        loss_fg = loss_fn(fg_pred, label_batch)
                        bg_label = 1.0 - label_batch
                        loss_bg = loss_fn(bg_pred, bg_label)

                        beta1 = 1.0 / (torch.tanh(torch.sigmoid(fg_pred).sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
                        beta2 = 1.0 / (torch.tanh(torch.sigmoid(bg_pred).sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)

                        prob_fg = torch.sigmoid(fg_pred)
                        prob_bg = torch.sigmoid(bg_pred)
                        prob_uc = torch.sigmoid(uc_pred)
                        preds_stack = torch.stack([prob_fg, prob_bg, prob_uc], dim=1)
                        probs_soft = F.softmax(preds_stack, dim=1)
                        p_fg, p_bg, p_uc = probs_soft[:, 0], probs_soft[:, 1], probs_soft[:, 2]
                        num_pixels = p_fg.shape[0] * p_fg.shape[2] * p_fg.shape[3]
                        loss_comp = ((p_fg * p_bg).sum() + (p_fg * p_uc).sum() + (p_bg * p_uc).sum()) / num_pixels

                        aux_weight = float(getattr(condseg_cfg, 'AUX_LOSS_WEIGHT', 1.0)) if condseg_cfg else 1.0
                        loss = loss_mask + aux_weight * (beta1 * loss_fg + beta2 * loss_bg + loss_comp)
                    else:
                        loss = loss_mask
                else:
                    loss = loss_fn(outputs, label_batch)
            else:
                outputs = model(image_batch)

                if isinstance(outputs, (list, tuple)):
                    loss = 0
                    for o in outputs:
                        loss += loss_fn(o, label_batch)
                else:
                    loss = loss_fn(outputs, label_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            iter_num += 1
            progress = iter_num / max_iterations
            lr_ = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
            lr_ = max(lr_, base_lr * 0.01)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            epoch_loss_sum += loss.item()
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_:.6f}"})

        epoch_train_loss = epoch_loss_sum / len(trainloader)
        train_losses.append(epoch_train_loss)
        logging.info("Epoch %d train loss %.6f", epoch_idx + 1, epoch_train_loss)


        if len(valloader) > 0:
            macro, micro, val_loss, per_class = validate_wsi(args, model, valloader, loss_fn)
            val_losses.append(val_loss)
            current_dice = macro["mDice"]
            val_metrics["dice"].append(current_dice)
            val_metrics["iou"].append(macro["mIoU"])
            val_metrics["sensitivity"].append(macro["mSensitivity"])
            val_metrics["mcc"].append(macro["mMCC"])
            log_split_metrics("val", epoch_idx + 1, label_names, per_class, macro, micro, snapshot_path)

            if np.isnan(current_dice):
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    logging.info("Early stopping triggered")
                    break
            elif (epoch_idx + 1) < min_best_epoch:
                logging.info(
                    "Epoch %d < min_best_epoch=%d, skip best-model selection (not saved/tested).",
                    epoch_idx + 1,
                    min_best_epoch,
                )
            else:
                base_model = get_base_model(model)
                checkpoint = {
                    "model_state_dict": base_model.state_dict(),
                    "epoch": epoch_idx + 1,
                    "best_dice": current_dice,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                }
                if hasattr(base_model, "L_encoder"):
                    checkpoint["L_encoder_state"] = base_model.L_encoder.state_dict()
                if hasattr(base_model, "G_encoder"):
                    checkpoint["G_encoder_state"] = base_model.G_encoder.state_dict()

                ckpt_cpu = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                checkpoint["model_state_dict"] = ckpt_cpu
                if "L_encoder_state" in checkpoint:
                    checkpoint["L_encoder_state"] = {
                        k: v.detach().cpu().clone()
                        for k, v in checkpoint["L_encoder_state"].items()
                    }
                if "G_encoder_state" in checkpoint:
                    checkpoint["G_encoder_state"] = {
                        k: v.detach().cpu().clone()
                        for k, v in checkpoint["G_encoder_state"].items()
                    }
                best_list.append((current_dice, epoch_idx + 1, checkpoint))
                best_list.sort(key=lambda x: x[0], reverse=True)
                best_list = best_list[:top_k]

                for rank, (dice, ep, ckpt) in enumerate(best_list):
                    top_path = os.path.join(snapshot_path, f"best_model_top{rank+1}_ep{ep:03d}.pth")
                    torch.save(ckpt, top_path)
                logging.info("Top-3 best models: %s",
                             ", ".join([f"ep{ep}(Dice={dice:.4f})" for dice, ep, _ in best_list]))

                if current_dice >= best_list[0][0]:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if patience > 0 and epochs_no_improve >= patience:
                        logging.info("Early stopping triggered")
                        break
        else:
            val_losses.append(np.nan)

        if test_every_epoch and len(testloader) > 0:
            test_macro_ep, test_micro_ep, test_loss_ep, test_per_class_ep = validate_wsi(
                args, model, testloader, loss_fn
            )
            log_split_metrics(
                "test", epoch_idx + 1, label_names,
                test_per_class_ep, test_macro_ep, test_micro_ep, snapshot_path,
            )
            logging.info(
                "[Epoch %d Test] mDice: %.4f | mIoU: %.4f | mSensitivity: %.4f | mMCC: %.4f",
                epoch_idx + 1,
                test_macro_ep["mDice"],
                test_macro_ep["mIoU"],
                test_macro_ep["mSensitivity"],
                test_macro_ep["mMCC"],
            )

        save_loss_curve(train_losses, val_metrics, os.path.join(snapshot_path, "loss_curve.png"))

    base_model = get_base_model(model)
    last_checkpoint = {
        "model_state_dict": base_model.state_dict(),
        "epoch": args.max_epochs,
        "args": vars(args),
    }
    torch.save(last_checkpoint, last_model_path)
    logging.info("Saved last epoch model to %s", last_model_path)

    if len(testloader) > 0 and len(best_list) > 0:
        logging.info("=" * 60)
        logging.info("Final test evaluation on Top-%d best models..." % len(best_list))
        logging.info("=" * 60)

        for rank, (dice, ep, ckpt) in enumerate(best_list):
            logging.info("-" * 40)
            logging.info("Loading Top-%d model (epoch %d, val Dice %.4f)", rank + 1, ep, dice)
            logging.info("-" * 40)

            base_model.load_state_dict(ckpt["model_state_dict"])
            if "L_encoder_state" in ckpt and hasattr(base_model, "L_encoder"):
                base_model.L_encoder.load_state_dict(ckpt["L_encoder_state"])
            if "G_encoder_state" in ckpt and hasattr(base_model, "G_encoder"):
                base_model.G_encoder.load_state_dict(ckpt["G_encoder_state"])

            test_macro, test_micro, test_loss, test_per_class = validate_wsi(
                args, model, testloader, loss_fn
            )
            log_split_metrics("test_top%d_ep%03d" % (rank + 1, ep), ep,
                              label_names, test_per_class, test_macro, test_micro, snapshot_path)

            logging.info("Top-%d (epoch %d) test results - mDice: %.4f | mIoU: %.4f | mSensitivity: %.4f | mMCC: %.4f",
                         rank + 1, ep,
                         test_macro["mDice"], test_macro["mIoU"],
                         test_macro["mSensitivity"], test_macro["mMCC"])

    logging.info("Training Finished. Top-3 Val Dice: %s",
                 ", ".join([f"ep{ep}(Dice={dice:.4f})" for dice, ep, _ in best_list]))
    return "Training Finished!"


def trainer(args, model, snapshot_path, config=None):

    return train_with_val_and_test(args, model, snapshot_path, config=config)
