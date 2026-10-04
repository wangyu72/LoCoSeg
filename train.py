import os



import argparse
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network.LoCoSeg import build_locoseg




from trainer import trainer
from config import get_config
import sys
def worker_init_fn(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_base_model(model):

    return model.module if isinstance(model, torch.nn.DataParallel) else model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/home/wangys/1/data/patches_output",
    help="root dir for data",
)
parser.add_argument(
    "--dataset", type=str, default="patches_output", help="experiment_name"
)
parser.add_argument(
    "--list_dir", type=str, default="./lists/lists_WS10x", help="list dir"
)
parser.add_argument(
    "--num_classes", type=int, default=7, help="output channel of network"
)
parser.add_argument("--output_dir", type=str, default="model", help="output dir")
parser.add_argument(
    "--max_epochs", type=int, default=150, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument(
    "--val_batch_size", type=int, default=8, help="validation/test batch_size per gpu"
)

parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.0001, help="segmentation network learning rate"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument(
    "--cfg",
    type=str,
    required=False,
    default="configs/DHUnet_224.yaml",
    metavar="FILE",
    help="path to config file",
)

parser.add_argument(
    "--label_folders",
    type=str,
    default="",
    help="comma separated list of mask folders for multi-label dataset, auto detected when omitted",
)
parser.add_argument(
    "--thr",
    type=float,
    default=0.5,
    help="threshold for predictions",
)
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="early stopping patience in epochs (<=0 disables)",
)
parser.add_argument(
    "--min_best_epoch",
    type=int,
    default=2,
    help="epochs with epoch number < min_best_epoch are not eligible as best models for saving/testing (>=2, default: 2, i.e. epoch 1 is excluded)",
)
parser.add_argument(
    "--test_every_epoch",
    type=int,
    default=1,
    help="if >0, evaluate the test set every epoch and append results to test_metrics.csv; 0 disables (default: 1)",
)

parser.add_argument(
    "--network", type=str, default="LoCoSeg", help="the model of network"
)
parser.add_argument(
    "--n_gpu",
    type=int,
    default=1,
    help="number of GPUs to use (default: 1, use all available if > available count)",
)
parser.add_argument(
    "--enable_classification",
    action="store_true",
    default=False,
    help="enable classification head for ViTexVMamba (patch-level classification)",
)
parser.add_argument(
    "--cls_num_classes",
    type=int,
    default=1,
    help="number of classes for classification head (1=binary, >1=multi-label)",
)
parser.add_argument(
    "--cls_loss_weight",
    type=float,
    default=0.2,
    help="weight for classification loss (default: 0.2, following MFTrans)",
)
args = parser.parse_args()
args.label_folders = (
    []
    if not args.label_folders
    else [f.strip() for f in args.label_folders.split(",") if f.strip()]
)
args.val_batch_size = max(1, args.val_batch_size)
config = get_config(args)


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


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "patches_output": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 7,
        },
        "aggc2022": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 5,
            "label_folders": ["g3", "g4", "g5", "normal", "stroma"],
        },
        "NEPTUNE": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 5,
            "label_folders": ["PTC_lumen", "capsule", "distal", "proximal", "tuft"],
        },
    }

    if "aggc2022" in args.list_dir.lower() and dataset_name == "patches_output":
        print("Auto-detected dataset: aggc2022 (based on list_dir path)")
        dataset_name = "aggc2022"
        args.dataset = "aggc2022"

    if "NEPTUNE" in args.root_path and dataset_name == "patches_output":
        print("Auto-detected dataset: NEPTUNE (based on root_path)")
        dataset_name = "NEPTUNE"
        args.dataset = "NEPTUNE"

    dataset_entry = dataset_config.get(dataset_name)
    if dataset_entry is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}', please add an entry to dataset_config."
        )
    args.root_path = dataset_entry["root_path"]
    args.list_dir = dataset_entry["list_dir"]

    if not args.label_folders and "label_folders" in dataset_entry:
        args.label_folders = dataset_entry["label_folders"]

    foreground_folders = args.label_folders or infer_multilabel_folders(args.list_dir)
    if args.root_path == parser.get_default("root_path"):
        args.root_path = dataset_entry["root_path"]
    if args.list_dir == parser.get_default("list_dir"):
        args.list_dir = dataset_entry["list_dir"]

    if args.label_folders:
        foreground_folders = args.label_folders
    elif "label_folders" in dataset_entry:
        foreground_folders = dataset_entry["label_folders"]
    else:
        foreground_folders = infer_multilabel_folders(args.list_dir)

    if not foreground_folders:
        raise ValueError(
            "No label folders detected; pass --label_folders explicitly so training knows the channel order."
        )

    args.label_folders = foreground_folders
    args.num_classes = len(foreground_folders)

    n_gpu = torch.cuda.device_count()
    if args.n_gpu > 1 and n_gpu > 1:
        args.n_gpu = min(args.n_gpu, n_gpu)
    else:
        args.n_gpu = 1

    method_dir = os.path.join(args.output_dir, args.network)
    args.output_dir = os.path.join(method_dir, "log")
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=args.output_dir + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Using multi-label folders: {args.label_folders}")

    if args.network == "LoCoSeg":

        locoseg_cfg = getattr(config.MODEL, 'LOCOSEG', None)

        if locoseg_cfg:
            drop_path_rate = float(getattr(locoseg_cfg, 'DROP_PATH_RATE', 0.1))
            use_ema = bool(getattr(locoseg_cfg, 'USE_EMA', True))
            deep_supervision = bool(getattr(locoseg_cfg, 'DEEP_SUPERVISION', False))
            pretrained_path = getattr(locoseg_cfg, 'PRETRAINED', None)
        else:
            drop_path_rate = 0.1
            deep_supervision = False
            pretrained_path = "./pretrained_ckpt/vmamba_tiny_e292.pth"

        prompt_csv = getattr(locoseg_cfg, 'PROMPT_CSV', None) if locoseg_cfg else None
        if not prompt_csv:
            dataset_prompt_map = {
                "patches_output": "./text_prompt/text_prompt.csv",
                "aggc2022":       "./text_prompt/text_prompts.csv",
                "NEPTUNE":        "./text_prompt/text_prompt1.csv",
            }
            prompt_csv = dataset_prompt_map.get(args.dataset, "./text_prompt/text_prompt.csv")
            logging.info(f"Auto-selected prompt file for dataset '{args.dataset}': {prompt_csv}")

        net = build_locoseg(
            config=config,
            num_classes=args.num_classes,
            in_chans=3,
            classnames=args.label_folders,
            drop_path_rate=drop_path_rate,
            deep_supervision=deep_supervision,
            pretrained=pretrained_path,
            prompt_csv=prompt_csv,
            conch_cfg="conch_ViT-B-16",
            conch_ckpt="./pretrained_ckpt/pytorch_model.bin",
        ).cuda()

        net.load_from(config)

        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info(f"LoCoSeg: Swin-Tiny Encoder + Dynamic_MSAA Decoder")
        logging.info(f"  Total params: {total_params:,}")
        logging.info(f"  Trainable params: {trainable_params:,}")

    if args.n_gpu > 1:
        logging.info(
            f"Using {args.n_gpu} GPUs for training (out of {torch.cuda.device_count()} available)"
        )
    else:
        logging.info("Using single GPU training")

    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        logging.info(f"Model wrapped with DataParallel on GPUs: {gpu_ids}")
        logging.info(
            f"Effective batch size: {args.batch_size} x {args.n_gpu} = {args.batch_size * args.n_gpu}"
        )
    else:
        logging.info("Single GPU training mode")

    print(str(net))
    trainer(args, net, args.output_dir, config=config)
