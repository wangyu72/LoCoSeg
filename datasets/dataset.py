import os
import random
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from collections import Counter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, falling back to PIL (slower image loading)")

_mask_data_cache = {}
_MASK_CACHE_MAX = 5000


def transform_convert(img_tensor, transform):
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, T.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

class RandomVerticalFlip(object):
    def __init__(self, vertical_prob):
        self.vertical_prob = vertical_prob

    def __call__(self, image, mask=None):
        if random.random() < self.vertical_prob:
            image = F.vflip(image)
            if mask is not None:
                if isinstance(mask, (list, tuple)):
                    mask = [F.vflip(m) for m in mask]
                else:
                    mask = F.vflip(mask)
        return image, mask

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                if isinstance(mask, (list, tuple)):
                    mask = [F.hflip(m) for m in mask]
                else:
                    mask = F.hflip(mask)
        return image, mask

class RandomRotation(object):
    def __init__(self, rotate_prob, max_angle=90):
        self.rotate_prob = rotate_prob
        self.max_angle = max_angle

    def __call__(self, image, mask=None):
        if random.random() < self.rotate_prob:
            angle = random.randint(-self.max_angle, self.max_angle)
            image = image.rotate(angle, resample=Image.BILINEAR)
            if mask is not None:
                if isinstance(mask, (list, tuple)):
                    mask = [m.rotate(angle, resample=Image.NEAREST, fillcolor=0) for m in mask]
                else:
                    mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return image, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, mask=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            if isinstance(mask, (list, tuple)):
                mask = [F.resize(m, self.size, interpolation=InterpolationMode.NEAREST) for m in mask]
            else:
                mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = F.to_tensor(image)
        if mask is not None:
            if isinstance(mask, (list, tuple)):
                mask_arrs = []
                for m in mask:
                    arr = np.asarray(m)
                    arr = (arr > 0).astype(np.uint8)
                    mask_arrs.append(arr)
                mask_np = np.stack(mask_arrs, axis=0)
                mask = torch.from_numpy(mask_np)
            else:
                mask = torch.from_numpy(np.asarray(mask))
        return image, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class PatchesMultiLabelDataset(Dataset):

    def __init__(self, list_dir, split, label_folders=None, target_size=None):
        self.split = split
        self.list_dir = list_dir
        self.target_size = target_size

        if self.split == "test":
            self.image_list = self.get_sample_list("test_images")
        elif self.split == "val":
            self.image_list = self.get_sample_list("val_images")
        else:
            self.image_list = self.get_sample_list("train_images")

        if label_folders is None or len(label_folders) == 0:
            candidates_set = set()
            for img_path in self.image_list:
                wsi_dir = os.path.dirname(os.path.dirname(img_path))
                try:
                    for d in os.listdir(wsi_dir):
                        if d.startswith('masks_') and os.path.isdir(os.path.join(wsi_dir, d)):
                            candidates_set.add(d[len('masks_'):])
                except FileNotFoundError:
                    continue
            self.label_folders = sorted(list(candidates_set))
        else:
            self.label_folders = [f for f in label_folders if f.lower() != "background"]

        self._build_mask_cache()

        if self.split == "train":
            self.img_transform = Compose([
                Resize(self.target_size) if self.target_size is not None else (lambda img, msk: (img, msk)),
                RandomHorizontalFlip(flip_prob=0.5),
                RandomVerticalFlip(vertical_prob=0.5),
                RandomRotation(rotate_prob=0.5),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = Compose([
                Resize(self.target_size) if self.target_size is not None else (lambda img, msk: (img, msk)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def _build_mask_cache(self):

        self._mask_cache = {}
        for img_path in self.image_list:
            wsi_dir = os.path.dirname(os.path.dirname(img_path))
            wsi_name = os.path.basename(wsi_dir)
            if wsi_name not in self._mask_cache:
                cache = {}
                for name in self.label_folders:
                    mask_dir = os.path.join(wsi_dir, 'masks_' + name)
                    if os.path.isdir(mask_dir):
                        try:
                            cache[name] = len(os.listdir(mask_dir)) > 0
                        except OSError:
                            cache[name] = False
                    else:
                        cache[name] = False
                self._mask_cache[wsi_name] = cache

    def __len__(self):
        return len(self.image_list)

    def get_sample_list(self, sample_name):
        path = os.path.join(self.list_dir, sample_name + '.txt')
        if not os.path.exists(path):
            print(f"Warning: List file not found: {path}")
            return []
        with open(path) as f:
            sample_list = [line.strip('\n') for line in f if line.strip()]
        return sample_list

    def rgb_loader(self, path):
        if CV2_AVAILABLE:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')

    def _get_wsi_name(self, img_path):

        return os.path.basename(os.path.dirname(os.path.dirname(img_path)))

    def _load_mask_via_cache(self, mask_path):

        global _mask_data_cache
        if mask_path in _mask_data_cache:
            return _mask_data_cache[mask_path]

        if CV2_AVAILABLE:
            m_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            with open(mask_path, 'rb') as f:
                m_arr = np.asarray(Image.open(f).convert('L'))

        if m_arr is None:
            return None

        if len(_mask_data_cache) >= _MASK_CACHE_MAX:
            _mask_data_cache.clear()
        _mask_data_cache[mask_path] = m_arr
        return m_arr

    def load_mask_channels(self, img_path, size):

        fname = os.path.basename(img_path)
        wsi_name = self._get_wsi_name(img_path)
        wsi_cache = self._mask_cache.get(wsi_name, {})
        masks = []
        annotation_valid = []

        for name in self.label_folders:
            if wsi_cache.get(name, False):
                wsi_dir = os.path.dirname(os.path.dirname(img_path))
                mask_path = os.path.join(wsi_dir, 'masks_' + name, fname)
                m_arr = self._load_mask_via_cache(mask_path)
                if m_arr is not None:
                    m = Image.fromarray(m_arr, mode='L')
                    masks.append(m)
                    annotation_valid.append(True)
                    continue

            m = Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8))
            masks.append(m)
            annotation_valid.append(False)

        return masks, annotation_valid

    def __getitem__(self, idx):
        img_path = self.image_list[idx]

        image = self.rgb_loader(img_path)

        masks, annotation_valid = self.load_mask_channels(img_path, image.size)

        image, masks = self.img_transform(image, masks)

        sample = {
            'image': image,
            'mask': masks,
            'annotation_valid': torch.tensor(annotation_valid, dtype=torch.bool),
            'case_name': img_path.split('/', 1)[-1],
        }
        return sample


class AGGC2022Dataset(PatchesMultiLabelDataset):

    def __init__(self, list_dir, split, label_folders=None, target_size=None):
        super().__init__(list_dir, split, label_folders, target_size)

        self.wsi_available_labels = self._scan_wsi_labels()

    def _build_mask_cache(self):

        self._mask_cache = {}
        for img_path in self.image_list:
            wsi_dir = os.path.dirname(os.path.dirname(img_path))
            wsi_name = os.path.basename(wsi_dir)
            if wsi_name not in self._mask_cache:
                cache = {}
                masks_root = os.path.join(wsi_dir, 'masks')
                if os.path.isdir(masks_root):
                    for name in self.label_folders:
                        label_dir = os.path.join(masks_root, name)
                        if os.path.isdir(label_dir):
                            try:
                                cache[name] = len(os.listdir(label_dir)) > 0
                            except OSError:
                                cache[name] = False
                        else:
                            cache[name] = False
                else:
                    for name in self.label_folders:
                        cache[name] = False
                self._mask_cache[wsi_name] = cache

    def _scan_wsi_labels(self):

        wsi_labels = {}
        for img_path in self.image_list:
            wsi_name = self._get_wsi_name(img_path)
            if wsi_name not in wsi_labels:
                available = set()
                for name, has_mask in self._mask_cache.get(wsi_name, {}).items():
                    if has_mask:
                        available.add(name)
                wsi_labels[wsi_name] = available
        return wsi_labels

    def load_mask_channels(self, img_path, size):

        fname = os.path.basename(img_path)
        wsi_name = self._get_wsi_name(img_path)
        wsi_cache = self._mask_cache.get(wsi_name, {})

        masks = []
        annotation_valid = []
        for name in self.label_folders:
            if wsi_cache.get(name, False):
                wsi_dir = os.path.dirname(os.path.dirname(img_path))
                mask_path = os.path.join(wsi_dir, 'masks', name, fname)
                m_arr = self._load_mask_via_cache(mask_path)
                if m_arr is not None:
                    m = Image.fromarray(m_arr, mode='L')
                    masks.append(m)
                    annotation_valid.append(True)
                    continue

            zero_mask = Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8))
            masks.append(zero_mask)
            annotation_valid.append(False)

        return masks, annotation_valid
