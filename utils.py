import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def get_base_model(model):

    return model.module if isinstance(model, torch.nn.DataParallel) else model



def TimeOnCuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=None, eps=1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight
        self.eps = eps

    def _dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.eps) / (z_sum + y_sum + self.eps)
        return 1 - loss

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        target_tensor = target.float()
        weights = self.weight or [1.0] * self.n_classes
        assert inputs.size() == target_tensor.size(), 'predict {} & target {} shape do not match'.format(
            inputs.size(), target_tensor.size()
        )
        total_loss = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)
        total_weight = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target_tensor[:, i])
            if dice is None:
                continue
            w_i = weights[i]
            total_loss = total_loss + dice * w_i
            total_weight += w_i
        if total_weight == 0:
            return torch.zeros(1, device=inputs.device, dtype=inputs.dtype)
        return total_loss / total_weight

class FocalLoss(nn.Module):


    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple, np.ndarray)):
            self.register_buffer("alpha", torch.tensor(alpha).float())
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight_map = alpha_t

        loss = weight_map * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MemoryEfficientSoftDiceLoss(nn.Module):

    def __init__(self, batch_dice=False, smooth=1e-5):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, probs, targets):

        assert probs.shape == targets.shape
        B, C, H, W = probs.shape

        if self.batch_dice:
            intersect = (probs * targets).sum(dim=(0, 2, 3))
            pred_sum = probs.sum(dim=(0, 2, 3))
            target_sum = targets.sum(dim=(0, 2, 3))
            dice = (2 * intersect + self.smooth) / (pred_sum + target_sum + self.smooth)
            loss = (1 - dice).mean()
        else:
            intersect = (probs * targets).sum(dim=(2, 3))
            pred_sum = probs.sum(dim=(2, 3))
            target_sum = targets.sum(dim=(2, 3))
            dice = (2 * intersect + self.smooth) / (pred_sum + target_sum + self.smooth)
            loss = (1 - dice).mean()

        return loss


class DC_and_Focal_loss(nn.Module):

    def __init__(self, batch_dice=False, smooth=1e-5, dice_weight=1.0, focal_weight=0.5,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        if isinstance(alpha, (list, tuple, np.ndarray)):
            self.register_buffer("alpha", torch.tensor(alpha).float())
        else:
            self.alpha = alpha

    def _get_indicator(self, targets):

        has_fg = targets.view(targets.size(0), targets.size(1), -1).sum(dim=2) > 0
        return has_fg.float()

    def forward(self, logits, targets):

        targets = targets.float()
        probs = torch.sigmoid(logits)
        B, C, H, W = logits.shape

        if (hasattr(self, '_annotation_valid') and self._annotation_valid is not None
                and self._annotation_valid.shape[0] == B):
            indicator = self._annotation_valid.float()
        else:
            indicator = self._get_indicator(targets)

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_mod = (1 - p_t) ** self.gamma

        bce_all = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.view(1, C, 1, 1) * targets + (1 - self.alpha.view(1, C, 1, 1)) * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_all = alpha_t * focal_mod * bce_all

        focal_per_pair = focal_all.view(B, C, -1).mean(dim=2)
        focal_masked = focal_per_pair * indicator
        num_valid = indicator.sum().clamp(min=1)
        focal_loss = focal_masked.sum() / num_valid

        indicator_4d = indicator.view(B, C, 1, 1)

        intersect = (probs * targets * indicator_4d).sum(dim=(0, 2, 3))
        pred_sum  = (probs * indicator_4d).sum(dim=(0, 2, 3))
        target_sum = (targets * indicator_4d).sum(dim=(0, 2, 3))

        dice = (2 * intersect + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_per_class = 1 - dice

        class_has_any = indicator.any(dim=0).float()
        num_valid_classes = class_has_any.sum().clamp(min=1)
        dice_loss = (dice_per_class * class_has_any).sum() / num_valid_classes

        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss


class CombinedLoss(nn.Module):


    def __init__(
        self,
        n_classes,
        dice_weight=1,
        focal_weight=0.5,
        dice_class_weights=None,
        focal_alpha=0.25,
        focal_gamma=2.0,
        smooth=1e-5,
    ):
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(n_classes, weight=dice_class_weights, eps=smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets.float())
        dice_loss = self.dice_loss(inputs, targets.float())

        loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return loss


class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):

        probs = torch.sigmoid(inputs)
        targets_float = targets.float()

        probs_pos = probs
        probs_neg = (probs + self.clip).clamp(max=1.0)

        pos_loss = targets_float * torch.log(probs_pos.clamp(min=self.eps)) * ((1 - probs_pos) ** self.gamma_pos)

        neg_loss = (1 - targets_float) * torch.log((1 - probs_neg).clamp(min=self.eps)) * (probs_neg ** self.gamma_neg)

        loss = -(pos_loss + neg_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):

    def __init__(self, boundary_width=3, weight_type='binary', boundary_weight=10.0, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.boundary_width = boundary_width
        self.weight_type = weight_type
        self.boundary_weight = boundary_weight
        self.reduction = reduction

    def get_boundary_mask(self, targets):

        kernel_size = 2 * self.boundary_width + 1
        padding = self.boundary_width

        dilated = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=padding)

        eroded = -F.max_pool2d(-targets, kernel_size=kernel_size, stride=1, padding=padding)

        boundary_mask = dilated - eroded

        return boundary_mask

    def forward(self, inputs, targets):

        probs = torch.sigmoid(inputs)
        targets_float = targets.float()

        boundary_mask = self.get_boundary_mask(targets_float)

        if self.weight_type == 'binary':
            weight_map = torch.where(boundary_mask > 0,
                                    torch.tensor(self.boundary_weight, device=inputs.device),
                                    torch.tensor(1.0, device=inputs.device))
        else:
            weight_map = 1.0 + boundary_mask * (self.boundary_weight - 1.0)

        bce = F.binary_cross_entropy_with_logits(inputs, targets_float, reduction='none')
        weighted_bce = bce * weight_map

        if self.reduction == 'mean':
            return weighted_bce.mean()
        elif self.reduction == 'sum':
            return weighted_bce.sum()
        else:
            return weighted_bce



def calculate_dice_coefficient(pred, target, eps=1e-5):

    pred = pred.float()
    target = target.float()

    pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=(0, 2))
    pred_sum = pred_flat.sum(dim=(0, 2))
    target_sum = target_flat.sum(dim=(0, 2))

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    denom = (pred_sum + target_sum)
    dice = torch.where(denom > 0, dice, torch.tensor(float('nan'), device=dice.device))

    return dice


def calculate_iou(pred, target, eps=1e-5):

    pred = pred.float()
    target = target.float()

    pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=(0, 2))
    union = pred_flat.sum(dim=(0, 2)) + target_flat.sum(dim=(0, 2)) - intersection

    iou = (intersection + eps) / (union + eps)
    iou = torch.where(union > 0, iou, torch.tensor(float('nan'), device=iou.device))
    return iou


def calculate_accuracy(pred, target, eps=1e-5):
    pred = pred.float()
    target = target.float()

    pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)

    correct = (pred_flat == target_flat).float().sum(dim=(0, 2))
    total = pred_flat.size(0) * pred_flat.size(2)
    accuracy = correct / (total + eps)
    return accuracy


def validate_single_patch_multilabel(
    image, label, net, classes, threshold=0.5, network="DHUnet"
):

    image = image.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)

    net.eval()
    with torch.no_grad():
        if network in ["DHUnet", "ViTexVMamba"]:
            raw_outputs = net(image, image)
        else:
            raw_outputs = net(image)

        if isinstance(raw_outputs, (list, tuple)):
            logits = raw_outputs[0]
        else:
            logits = raw_outputs

        probs = torch.sigmoid(logits)
        prediction = (probs > threshold).byte()

    if label.dim() == 3:
        label = label.unsqueeze(0)
    target = (label > 0.5).byte()

    dice = calculate_dice_coefficient(prediction, target)
    iou = calculate_iou(prediction, target)
    accuracy = calculate_accuracy(prediction, target)


    return {
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
    }



def compute_batch_metrics_macro(predictions, labels, threshold=0.5, eps=1e-5, annotation_valid=None):

    probs = torch.sigmoid(predictions) if predictions.min() < 0 else predictions

    num_classes = labels.size(1)
    device = predictions.device

    if isinstance(threshold, (list, tuple)):
        threshold_tensor = torch.tensor(threshold, device=device, dtype=predictions.dtype)
    elif isinstance(threshold, np.ndarray):
        threshold_tensor = torch.from_numpy(threshold).to(device=device, dtype=predictions.dtype)
    elif isinstance(threshold, torch.Tensor):
        threshold_tensor = threshold.to(device=device, dtype=predictions.dtype)
    else:
        threshold_tensor = torch.full((num_classes,), threshold, device=device, dtype=predictions.dtype)

    if threshold_tensor.dim() == 1:
        threshold_tensor = threshold_tensor.view(1, -1, 1, 1)

    pred_binary = (probs > threshold_tensor).float()

    dice_per_class = []
    iou_per_class = []
    accuracy_per_class = []

    tp_per_class = []
    fp_per_class = []
    tn_per_class = []
    fn_per_class = []

    for c in range(num_classes):
        if annotation_valid is not None:
            valid_mask = annotation_valid[:, c]
            if valid_mask.sum() == 0:
                dice_per_class.append(float('nan'))
                iou_per_class.append(float('nan'))
                accuracy_per_class.append(float('nan'))
                tp_per_class.append(0.0)
                fp_per_class.append(0.0)
                tn_per_class.append(0.0)
                fn_per_class.append(0.0)
                continue

            pred_c = pred_binary[valid_mask, c]
            label_c = labels[valid_mask, c]
        else:
            pred_c = pred_binary[:, c]
            label_c = labels[:, c]

        pred_flat = pred_c.reshape(-1)
        label_flat = label_c.reshape(-1)
        target_flat = (label_flat > 0.5).float()

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        tp_per_class.append(tp.item())
        fp_per_class.append(fp.item())
        tn_per_class.append(tn.item())
        fn_per_class.append(fn.item())

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

        dice_per_class.append(dice.item())
        iou_per_class.append(iou.item())
        accuracy_per_class.append(accuracy.item())

    dice_per_class_tensor = torch.tensor(dice_per_class)
    iou_per_class_tensor = torch.tensor(iou_per_class)
    accuracy_per_class_tensor = torch.tensor(accuracy_per_class)

    macro_dice = dice_per_class_tensor.nanmean().item()
    macro_iou = iou_per_class_tensor.nanmean().item()
    macro_accuracy = accuracy_per_class_tensor.nanmean().item()

    result = {
        'dice_per_class': dice_per_class,
        'iou_per_class': iou_per_class,
        'accuracy_per_class': accuracy_per_class,
        'tp_per_class': tp_per_class,
        'fp_per_class': fp_per_class,
        'tn_per_class': tn_per_class,
        'fn_per_class': fn_per_class,

        'macro_dice': macro_dice,
        'macro_iou': macro_iou,
        'macro_accuracy': macro_accuracy,
    }

    return result


def find_optimal_thresholds(model, dataloader, num_classes, network="ViTexVMamba",
                            device='cuda', threshold_range=None, label_names=None):

    from tqdm import tqdm
    import logging

    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.91, 0.05)

    if label_names is None:
        label_names = [f"class_{i}" for i in range(num_classes)]

    model.eval()


    logging.info("Starting per-class threshold optimization on validation set...")
    logging.info(f"Threshold search range: {threshold_range[0]:.2f} to {threshold_range[-1]:.2f}")

    num_thresholds = len(threshold_range)
    tp_accum = np.zeros((num_thresholds, num_classes), dtype=np.float64)
    pred_sum_accum = np.zeros((num_thresholds, num_classes), dtype=np.float64)
    target_sum_accum = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Threshold Search", leave=False):
            img = batch['image'].to(device)
            mask = batch['mask'].to(device).float()

            if network == "DHUnet":
                output = model(img, img)
                if isinstance(output, (tuple, list)):
                    output = output[0]
            else:
                output = model(img)
                if isinstance(output, tuple) and len(output) == 2:
                    seg_out, _ = output
                    if isinstance(seg_out, (list, tuple)):
                        output = seg_out[0]
                    else:
                        output = seg_out
                elif isinstance(output, (tuple, list)):
                    output = output[0]

            prob = torch.sigmoid(output)

            for t_idx, t in enumerate(threshold_range):
                pred_binary = (prob > t).float()

                for c in range(num_classes):
                    pred_c = pred_binary[:, c].reshape(-1)
                    target_c = (mask[:, c] > 0.5).float().reshape(-1)

                    tp = (pred_c * target_c).sum().item()
                    pred_sum = pred_c.sum().item()

                    tp_accum[t_idx, c] += tp
                    pred_sum_accum[t_idx, c] += pred_sum

            for c in range(num_classes):
                target_c = (mask[:, c] > 0.5).float().reshape(-1)
                target_sum_accum[c] += target_c.sum().item()

    best_thresholds = []
    eps = 1e-7

    logging.info("=" * 60)
    logging.info("Per-class optimal threshold results:")
    logging.info("=" * 60)

    for c in range(num_classes):
        best_dice = -1.0
        best_t = 0.5
        best_t_idx = -1

        target_sum_c = target_sum_accum[c]

        for t_idx, t in enumerate(threshold_range):
            tp = tp_accum[t_idx, c]
            pred_sum = pred_sum_accum[t_idx, c]

            dice = (2 * tp + eps) / (pred_sum + target_sum_c + eps)

            if dice > best_dice:
                best_dice = dice
                best_t = t
                best_t_idx = t_idx

        best_thresholds.append(float(best_t))

        total_pixels = pred_sum_accum[0, c] / (threshold_range[0] + eps) if pred_sum_accum[0, c] > 0 else 0
        foreground_ratio = target_sum_c / (target_sum_c + total_pixels + eps) if total_pixels > 0 else 0

        label_name = label_names[c] if c < len(label_names) else f"class_{c}"
        logging.info(f"  {label_name:15s}: threshold={best_t:.2f}, Dice={best_dice:.4f}")

    logging.info("=" * 60)
    logging.info(f"Optimal thresholds: {[f'{t:.2f}' for t in best_thresholds]}")

    return best_thresholds
