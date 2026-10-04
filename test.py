import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset import PatchesMultiLabelDataset, AGGC2022Dataset

from network.LoCoSeg import build_locoseg

from utils import (
    compute_batch_metrics_macro,
    DC_and_Focal_loss,
    find_optimal_thresholds,
)

try:
    from transition_metrics import (
        SCIPY_AVAILABLE as TRANSITION_SCIPY_AVAILABLE,
        TransitionBandEvaluator,
        group_patches_by_wsi,
        make_wsi_canvas,
        parse_patch_position,
    )
except ImportError:
    TRANSITION_SCIPY_AVAILABLE = False

def _normalize_label_folders(raw):
    return [] if not raw else [f.strip() for f in raw.split(",") if f.strip()]

def _read_list_file(list_dir, sample_name):
    path = os.path.join(list_dir, sample_name + ".txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def infer_multilabel_folders(list_dir):

    label_candidates = set()
    for sample in ("train_images", "val_images", "test_images"):
        sample_list = _read_list_file(list_dir, sample)
        for img_path in sample_list:
            wsi_dir = os.path.dirname(os.path.dirname(img_path))
            if not os.path.isdir(wsi_dir):
                continue
            try:
                for item in os.listdir(wsi_dir):
                    if item.startswith("masks_") and os.path.isdir(
                        os.path.join(wsi_dir, item)
                    ):
                        label_candidates.add(item[len("masks_") :])
            except FileNotFoundError:
                continue
    return sorted(label_candidates)

def get_dataloader(args, split="test", batch_size=1, shuffle=False):
    base_seed = args.seed
    generator = torch.Generator()
    generator.manual_seed(base_seed)

    target_size = (args.img_size, args.img_size) if args.img_size is not None else None

    dataset_label_folders = [f for f in args.label_folders if f.lower() != "background"]

    if args.dataset == "aggc2022":
        DatasetClass = AGGC2022Dataset
        logging.info("Using AGGC2022Dataset (Nested structure: masks/g3/...)")
    else:
        DatasetClass = PatchesMultiLabelDataset
        logging.info("Using PatchesMultiLabelDataset (Flat structure: masks_g3/...)")

    dataset = DatasetClass(
        list_dir=args.list_dir,
        split=split,
        label_folders=dataset_label_folders,
        target_size=target_size,
    )
    logging.info("The length of %s %s set is: %d" % (args.dataset, split, len(dataset)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        prefetch_factor=4,
    )

def build_model(args, config):

    if args.network == "LoCoSeg":
        locoseg_cfg = getattr(config.MODEL, 'LOCOSEG', None)

        if locoseg_cfg:
            drop_path_rate = float(getattr(locoseg_cfg, 'DROP_PATH_RATE', 0.1))
            use_ema = bool(getattr(locoseg_cfg, 'USE_EMA', True))
            deep_supervision = bool(getattr(locoseg_cfg, 'DEEP_SUPERVISION', False))
            pretrained_path = getattr(locoseg_cfg, 'PRETRAINED', None)
        else:
            drop_path_rate = 0.1
            use_ema = True
            deep_supervision = False
            pretrained_path = None

        dataset_prompt_map = {
            "patches_output": "./text_prompt/text_prompt.csv",
            "aggc2022":       "./text_prompt/text_prompts.csv",
            "NEPTUNE":        "./text_prompt/text_prompt1.csv",
        }
        prompt_csv = dataset_prompt_map.get(args.dataset, "./text_prompt/text_prompt.csv")

        net = build_locoseg(
            num_classes=args.num_classes,
            in_chans=3,
            classnames=args.label_folders,
            drop_path_rate=drop_path_rate,
            use_ema=use_ema,
            deep_supervision=deep_supervision,
            pretrained=pretrained_path,
            prompt_csv=prompt_csv,
            conch_cfg="conch_ViT-B-16",
            conch_ckpt="./pretrained_ckpt/pytorch_model.bin",
        )
        net.load_from(config)
    else:
        raise NotImplementedError("Unknown network %s" % args.network)
    return net

def _grouped_batch_iter_by_wsi(testloader, wsi_groups, batch_size, num_workers=4):

    dataset = testloader.dataset
    for wsi_name, info in wsi_groups.items():
        subset = torch.utils.data.Subset(dataset, info["indices"])
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for batch in tqdm(loader, desc=f"Testing {wsi_name}", leave=False):
            batch["_wsi_name"] = wsi_name
            batch["_wsi_ok"] = info["ok"]
            yield batch

def inference(args, model):

    if not isinstance(model, torch.nn.DataParallel):
        model = model.cuda()

    testloader = get_dataloader(args, split="test", batch_size=args.batch_size, shuffle=False)
    if len(testloader) == 0:
        logging.warning("No test samples found for %s", args.dataset)
        return

    logging.info("Starting inference on %d samples (batch_size=%d)..." % (len(testloader.dataset), args.batch_size))
    logging.info(f"{len(testloader)} test iterations")

    eval_transition = bool(getattr(args, "eval_transition", 1))
    transition_eval = None
    wsi_groups = None
    if eval_transition and TRANSITION_SCIPY_AVAILABLE:
        wsi_groups = group_patches_by_wsi(testloader.dataset.image_list)
        if len(wsi_groups) == 0:
            logging.warning("[transition] 无法解析 patch 文件名坐标, 跳过过渡带评估")
            wsi_groups = None
        else:
            n_ok = sum(1 for info in wsi_groups.values() if info["ok"])
            transition_eval = TransitionBandEvaluator(
                num_classes=args.num_classes,
                radii=getattr(args, "transition_radii", [0, 16]),
            )
            logging.info(
                "[transition] 过渡带评估已启用: radii=%s px (推理分辨率) | %d 张片 (%d 张可拼接评估)",
                getattr(args, "transition_radii", [0, 16]), len(wsi_groups), n_ok,
            )
    elif eval_transition:
        logging.warning("[transition] scipy 不可用, 跳过过渡带评估 (可 pip install scipy)")

    loss_fn = DC_and_Focal_loss(
        batch_dice=True,
        smooth=1e-6,
        dice_weight=1.0,
        focal_weight=0.5,
        alpha=0.25,
        gamma=2.0,
    ).cuda()
    loss_fn._annotation_valid = None

    model.eval()

    global_tp = np.zeros(args.num_classes)
    global_fp = np.zeros(args.num_classes)
    global_fn = np.zeros(args.num_classes)
    global_tn = np.zeros(args.num_classes)

    total_loss = 0.0
    batches = 0

    if transition_eval is not None:
        batch_iter = _grouped_batch_iter_by_wsi(testloader, wsi_groups, args.batch_size)
    else:
        batch_iter = iter(testloader)

    current_wsi = None
    pred_canvas = gt_canvas = valid_canvas = None
    canvas_geom = None

    for sampled_batch in tqdm(batch_iter, desc="Testing", leave=False):
        wsi_name = sampled_batch.get("_wsi_name", None)
        if transition_eval is not None and wsi_name is not None and wsi_name != current_wsi:
            if current_wsi is not None and gt_canvas is not None:
                transition_eval.add_slide(current_wsi, gt_canvas, pred_canvas, valid_canvas)
                logging.info("[transition] 已完成 %s 的过渡带统计", current_wsi)
            current_wsi = wsi_name
            pred_canvas = gt_canvas = valid_canvas = None
            canvas_geom = None
        image = sampled_batch["image"].cuda()
        label = sampled_batch["mask"].cuda().float()

        with torch.no_grad():
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
                    if isinstance(first, torch.Tensor):
                        primary_output = first
                    else:
                        raise RuntimeError("Unexpected model output structure")
            else:
                raise RuntimeError("Unexpected model output type: %s" % type(raw_outputs))

            annotation_valid = sampled_batch.get('annotation_valid', None)
            if annotation_valid is not None:
                loss_fn._annotation_valid = annotation_valid.cuda()
            else:
                loss_fn._annotation_valid = None

            loss = loss_fn(primary_output, label)

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

        if (transition_eval is not None and wsi_name is not None
                and sampled_batch.get("_wsi_ok", False)):
            probs = torch.sigmoid(primary_output) if primary_output.min() < 0 else primary_output
            if isinstance(args.thr, (list, tuple)):
                thr_t = torch.as_tensor(args.thr, device=probs.device, dtype=probs.dtype).view(1, -1, 1, 1)
            else:
                thr_t = args.thr
            pred_bin = probs > thr_t
            gt_bin = label > 0.5
            if pred_bin.shape[-2:] != gt_bin.shape[-2:]:
                pred_bin = torch.nn.functional.interpolate(
                    pred_bin.float(), size=gt_bin.shape[-2:], mode="nearest"
                ) > 0.5
            if pred_canvas is None:
                ph, pw = gt_bin.shape[-2], gt_bin.shape[-1]
                pred_canvas, gt_canvas, valid_canvas, canvas_geom = make_wsi_canvas(
                    wsi_groups[wsi_name]["coords"], (ph, pw), args.num_classes
                )
            r0, c0, ph, pw = canvas_geom
            pred_np = pred_bin.byte().cpu().numpy()
            gt_np = gt_bin.byte().cpu().numpy()
            case_names = sampled_batch.get("case_name", None)
            for b in range(pred_np.shape[0]):
                if case_names is None:
                    break
                pos = parse_patch_position(case_names[b])
                if pos is None:
                    continue
                y = (pos[0] - r0) * ph
                x = (pos[1] - c0) * pw
                pred_canvas[:, y:y + ph, x:x + pw] = pred_np[b]
                gt_canvas[:, y:y + ph, x:x + pw] = gt_np[b]
                if annotation_valid is not None:
                    valid_canvas[:, y:y + ph, x:x + pw] = (
                        annotation_valid[b].cpu().numpy().astype(bool).reshape(-1, 1, 1)
                    )

    if transition_eval is not None and current_wsi is not None and gt_canvas is not None:
        transition_eval.add_slide(current_wsi, gt_canvas, pred_canvas, valid_canvas)
        logging.info("[transition] 已完成 %s 的过渡带统计", current_wsi)

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

    per_class = {
        "dice": final_dice_per_class,
        "iou": final_iou_per_class,
        "sensitivity": final_sensitivity_per_class,
        "mcc": final_mcc_per_class,
    }

    macro = {
        "mDice": macro_dice,
        "mIoU": macro_iou,
        "mSensitivity": macro_sensitivity,
        "mMCC": macro_mcc,
    }

    micro = {}
    test_loss = total_loss / batches if batches > 0 else np.nan

    label_names = args.label_folders or [f"label_{i}" for i in range(args.num_classes)]

    logging.info("\n" + "="*70)
    logging.info("TEST RESULTS (Global Accumulation)")
    logging.info("="*70)
    logging.info(f"Test Loss: {test_loss:.6f}")
    if isinstance(args.thr, list):
        logging.info(f"Per-class Thresholds: {[f'{t:.2f}' for t in args.thr]}")
    else:
        logging.info(f"Threshold: {args.thr}")
    logging.info("")

    logging.info("Per-Class Metrics:")
    logging.info("-"*70)
    for idx in range(args.num_classes):
        label_name = label_names[idx] if idx < len(label_names) else f"label_{idx}"
        thr_str = f"(thr={args.thr[idx]:.2f})" if isinstance(args.thr, list) else ""

        logging.info(
            "%s %s: Dice %.4f | IoU %.4f | Sens %.4f | MCC %.4f",
            label_name, thr_str,
            per_class["dice"][idx],
            per_class["iou"][idx],
            per_class["sensitivity"][idx],
            per_class["mcc"][idx],
        )

    logging.info("")
    logging.info("Macro Average:")
    logging.info("-"*70)
    logging.info("mDice %.4f | mIoU %.4f | mSensitivity %.4f | mMCC %.4f",
                 macro["mDice"], macro["mIoU"], macro["mSensitivity"], macro["mMCC"])

    logging.info("")
    logging.info("="*70 + "\n")

    save_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)

    from trainer import log_split_metrics
    log_split_metrics("test", "final", label_names, per_class, macro, micro, save_dir)

    if transition_eval is not None:
        summary = transition_eval.summary()
        if isinstance(args.thr, (list, tuple)):
            thr_desc = ",".join(f"{t:.2f}" for t in args.thr)
        else:
            thr_desc = f"{float(args.thr):.2f}"
        transition_eval.log_summary(summary, label_names, thr_desc=thr_desc)
        model_subdir = str(getattr(args, "transition_subdir", "") or "").strip()
        if not model_subdir:
            model_subdir = str(getattr(args, "network", "") or "model")
        transition_dir = os.path.join(save_dir, model_subdir)
        logging.info("[transition] 过渡带表格保存目录: %s", transition_dir)
        transition_eval.write_csv(
            summary, label_names, transition_dir, tag=getattr(args, "ckpt_name", "")
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="patches_output", help="dataset name")
    parser.add_argument("--root_path",type=str,default="/home/wangys/1/data/patches_output",help="root dir for data")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_WS10x")
    parser.add_argument("--output_dir", type=str, default="model", help="output dir")
    parser.add_argument("--batch_size", type=int, default=8, help="test batch size")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=str, default="configs/DHUnet_224.yaml", metavar="FILE")
    parser.add_argument("--network", type=str, default="LoCoSeg")
    parser.add_argument("--resume", type=str, help="checkpoint path to load")
    parser.add_argument("--label_folders", type=str, default="", help="comma separated list of mask folders")
    parser.add_argument(
        "--thr",
        type=str,
        default="0.5",
        help="threshold for predictions. Single value (e.g., '0.5') or comma-separated per-class values (e.g., '0.3,0.5,0.6,0.4,0.5')",
    )
    parser.add_argument(
        "--use_dynamic_threshold",
        action="store_true",
        default=False,
        help="search optimal per-class thresholds on validation set before testing",
    )
    parser.add_argument(
        "--eval_transition",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否启用过渡带 Dice (M_r) 评估: 1=启用(默认), 0=关闭",
    )
    parser.add_argument(
        "--transition_radii",
        type=str,
        default="0,16",
        help="过渡带半径(像素, 相对推理分辨率), 逗号分隔; 默认 '0,16' (0=双标签像素, 16 对应约 20μm)",
    )
    parser.add_argument(
        "--transition_subdir",
        type=str,
        default="",
        help="过渡带表格的子目录名 (默认为空: 用网络名, 即 predictions/<network>/, 避免不同模型互相覆盖)",
    )

    args = parser.parse_args()

    if isinstance(args.thr, str):
        if ',' in args.thr:
            args.thr = [float(t.strip()) for t in args.thr.split(',') if t.strip()]
        else:
            args.thr = float(args.thr)

    args.transition_radii = [
        float(x) for x in str(args.transition_radii).split(',') if x.strip()
    ] or [0.0, 16.0]

    args.label_folders = _normalize_label_folders(args.label_folders)
    config = get_config(args)

    args.output_dir = os.path.join(args.output_dir, "all")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_config = {
        "patches_output": {
            "root_path": args.root_path,
            "list_dir": "./lists/lists_WS10x",
        },
    }
    default_list_dir = "./lists/lists_WS10x"
    user_specified_list_dir = args.list_dir != default_list_dir

    if args.dataset in dataset_config and not user_specified_list_dir:
        entry = dataset_config[args.dataset]
        args.list_dir = entry["list_dir"]

    args.label_folders = args.label_folders or infer_multilabel_folders(args.list_dir)
    if not args.label_folders:
        raise ValueError("Label folders could not be determined; pass --label_folders explicitly.")

    args.num_classes = len(args.label_folders)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Test Configuration: {args}")
    logging.info(f"Using label folders: {args.label_folders}")

    net = build_model(args, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    candidates = []
    if args.resume:
        candidates.append(args.resume)

    candidates.append(os.path.join(args.output_dir, "best_model.pth"))
    candidates.append(os.path.join(os.path.dirname(args.output_dir), args.network, "log", "best_model.pth"))
    candidates.append(os.path.join(args.output_dir, args.network, "log", "best_model.pth"))

    candidates.append(os.path.join("model", args.network, "log", "best_model.pth"))
    candidates.append(os.path.join(args.output_dir, "last_epoch.pth"))

    checkpoint_path = None
    for c in candidates:
        if not c:
            continue
        if os.path.exists(c):
            checkpoint_path = c
            break

    if checkpoint_path is None:
        tried = ", ".join(c for c in candidates if c)
        raise FileNotFoundError(
            f"Checkpoint not found. Tried the following paths: {tried}. Provide --resume pointing to a valid .pth file."
        )

    logging.info(f"Loading weights from {checkpoint_path}")
    args.ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    msg = net.load_state_dict(new_state_dict, strict=False)
    logging.info(f"Weights loaded. Missing/Unexpected keys: {msg}")

    if args.use_dynamic_threshold:
        logging.info("=" * 60)
        logging.info("Searching optimal per-class thresholds on validation set...")
        logging.info("=" * 60)

        valloader = get_dataloader(args, split="val", batch_size=args.batch_size, shuffle=False)

        if len(valloader) > 0:
            optimal_thresholds = find_optimal_thresholds(
                model=net,
                dataloader=valloader,
                num_classes=args.num_classes,
                network=args.network,
                device=device,
                label_names=args.label_folders,
            )
            args.thr = optimal_thresholds
            logging.info(f"Using optimized thresholds: {[f'{t:.2f}' for t in args.thr]}")
        else:
            logging.warning("Validation set is empty, using default threshold")

    inference(args, net)


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    main()
