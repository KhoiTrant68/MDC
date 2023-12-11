import os
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


def get_texture_map(img):
    """
    Generate a texture map from the input image using edge detection.

    Args:
        img (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Texture map generated using edge detection.
    """
    texture_map = img.filter(ImageFilter.FIND_EDGES)
    return texture_map


def get_structure_map(img):
    """
    Generate a structure map from the input image.

    Args:
        img (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Structure map generated based on pixel intensity range.
    """
    structure_map = img.point(
        lambda x: 255
        if x in range(85, 125) or x in range(30, 50) or x in range(180, 200)
        else 0
    )
    return structure_map


def calculate_patch_score(img, size=16, step=16):
    """
    Calculate patch scores from the input image.

    Args:
        img (PIL.Image.Image): Input image.
        size (int): Size of the patches (default is 16).
        step (int): Step size for moving the patches (default is 16).

    Returns:
        torch.Tensor: Patch scores calculated from the input image.
    """
    img = transforms.ToTensor()(img.resize((224, 224)))
    patch_img = img.unfold(1, size, step).unfold(2, size, step).unfold(3, size, step)
    patch_img_reshape = patch_img.squeeze(0).squeeze(2)
    img_score = patch_img_reshape.mean(dim=(2, 3)).flatten()
    return img_score


def get_total_score(img, size=16, step=16):
    """
    Calculate the total score based on texture and structure maps.

    Args:
        img (PIL.Image.Image): Input image.
        size (int): Size of the patches (default is 16).
        step (int): Step size for moving the patches (default is 16).

    Returns:
        torch.Tensor: Total score calculated from texture and structure maps.
    """
    img_gray = img.convert("L")
    t_map = get_texture_map(img_gray)
    s_map = get_structure_map(img_gray)
    t_score = calculate_patch_score(t_map, size=size, step=step)
    s_score = calculate_patch_score(s_map, size=size, step=step)
    total_score = t_score * s_score
    total_score = (total_score - total_score.min()) / (
        total_score.max() - total_score.min()
    )
    return total_score


def write_total_score(mode, args):
    assert mode in ["train", "val", "test"]

    # Read the dataset path
    dataset_path = args.dataset_path
    exp_name = dataset_path.split("/")[-1]

    # Create directory to store total_score for each dataset
    total_score_path = os.path.join(args.total_scores_path, exp_name)

    # Iteration
    if mode != "test":
        type_dataset_path = os.path.join(dataset_path, mode)
        type_total_score_path = os.path.join(total_score_path, mode)
        os.makedirs(type_total_score_path, exist_ok=True)
        list_name_file = sorted(os.listdir(type_dataset_path))
        list_total_score = list()
        for name_file in list_name_file:
            img_file = os.path.join(type_dataset_path, name_file)
            img = Image.open(img_file).convert("RGB")
            total_score = get_total_score(img)
            list_total_score.append(total_score)
        list_total_score = torch.stack(list_total_score)
        print("Shape of list_total_score: ", list_total_score.shape)
        torch.save(
            list_total_score, os.path.join(type_total_score_path, "total_scores.pt")
        )
    else:
        os.makedirs(total_score_path, exist_ok=True)
        list_name_file = sorted(os.listdir(dataset_path))
        list_total_score = list()
        for name_file in list_name_file:
            img_file = os.path.join(dataset_path, name_file)
            img = Image.open(img_file).convert("RGB")
            total_score = get_total_score(img)
            list_total_score.append(total_score)
        list_total_score = torch.stack(list_total_score)
        print("Shape of list_total_score: ", list_total_score.shape)
        torch.save(list_total_score, os.path.join(total_score_path, "total_scores.pt"))
    print("Done")
