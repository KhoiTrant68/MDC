import os
import torch
from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CreateImageDataset(Dataset):
    def __init__(
        self, mode: str, dataset_path: Path, total_scores_path: Path, transform
    ):
        """
        Custom dataset for image data.

        Args:
            mode (str): Dataset mode ("train", "val", or "test").
            dataset_path (Path): Path to the image dataset.
            total_scores_path (Path): Path to the corresponding total_scores csv file.
            transform (torchvision.transforms.Compose): Image transformations.
        """
        self.dataset_path = dataset_path
        exp_name = self.dataset_path.split("/")[-1]
        self.transform = transform

        self.root = (
            self.dataset_path
            if mode == "test"
            else os.path.join(self.dataset_path, mode)
        )
        self.imgs_path = sorted(Path(self.root).rglob("*.*"))
        assert len(self.imgs_path) > 0, f"No images found in {dataset_path}"

        self.total_scores_root = (
            os.path.join(total_scores_path, exp_name)
            if mode == "test"
            else os.path.join(total_scores_path, exp_name, mode)
        )

        self.total_scores = torch.load(
            os.path.join(self.total_scores_root, "total_scores.pt")
        )

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        orig_img = Image.open(img_path).convert("RGB")
        orig_shape = orig_img.size
        total_score = self.total_scores[idx]
        img = self.transform(orig_img)
        return img, orig_shape, total_score


def get_image_dataset(mode: str, args) -> Dataset:
    """
    Get an image dataset.

    Args:
        mode (str): Dataset mode ("train", "val", or "test").
        args (dict, optional): config.
    """
    assert mode in [
        "train",
        "val",
        "test",
    ], "Mode must be one of ['train', 'val', 'test']"

    if mode == "train":
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        t.append(transforms.ToTensor())
        # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)
    elif mode == "val":
        t = list()
        t.append(
            transforms.Resize((224, 224), interpolation=Image.BICUBIC)
        )  # to maintain same ratio w.r.t 224 images
        t.append(transforms.ToTensor())
        # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)
    else:
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)

    dataset = CreateImageDataset(
        mode=mode,
        dataset_path=args.dataset_path,
        total_scores_path=args.total_scores_path,
        transform=transform,
    )
    return dataset
