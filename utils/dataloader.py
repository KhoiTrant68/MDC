import json
import os
from pathlib import Path

import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms


class CreateImageDataset(Dataset):
    def __init__(self, mode: str, dataset_path: Path, transform):
        """
        Custom dataset for image data.

        Args:
            mode (str): Dataset mode ("train", "val", or "test").
            dataset_path (Path): Path to the dataset.
            transform (torchvision.transforms.Compose): Image transformations.
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.root = self.dataset_path if mode == "test" else os.path.join(self.dataset_path, mode)
        self.imgs_path = sorted(Path(self.root).rglob("*.*"))

        assert len(self.imgs_path) > 0, f"No images found in {self.dataset_path}"

        scores_folder = Path(self.dataset_path.parent) / f"{self.dataset_path.name}_scores"
        if not scores_folder.exists():
            raise RuntimeError(f"Scores folder '{scores_folder}' does not exist. Please run the generate_scores_file.py first.")
        scores_path = scores_folder / f"{mode}.pt"
        self.scores = torch.load(scores_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        orig_img = Image.open(img_path).convert("RGB")
        orig_shape = orig_img.size
        total_score = self.scores[idx]
        img = self.transform(orig_img)
        return img, orig_shape, total_score


def get_image_dataset(mode: str, dataset_path: Path) -> Dataset:
    """
    Get an image dataset.

    Args:
        mode (str): Dataset mode ("train", "val", or "test").
        dataset_path (Path): Dataset path.
    """
    assert mode in ["train", "val", "test"], "Mode must be one of ['train', 'val', 'test']"

    if mode == "train":
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)
    elif mode == "val":
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)
    else:
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)

    dataset = CreateImageDataset(
        dataset_path=dataset_path, mode=mode, transform=transform
    )
    return dataset
