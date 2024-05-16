from pathlib import Path

_root = (Path(__file__).parent / '../datasets').resolve()

known_datasets = {
    # Kodak images: http://r0k.us/graphics/kodak
    'train': _root / 'train',

    # CLIC dataset: http://www.compression.cc
    'clic2022-test': _root / 'clic/test-2022',

    # Tecnick TESTIMAGES: https://testimages.org
    'tecnick-rgb-1200': _root / 'tecnick/TESTIMAGES/RGB/RGB_OR_1200x1200',

    # COCO dataset: http://cocodataset.org
    'coco-train2017': _root / 'coco/train2017',
    'coco-val2017': _root / 'coco/val2017',

    # ImageNet dataset: http://www.image-net.org
    'imagenet-train': _root / 'imagenet/train',
    'imagenet-val': _root / 'imagenet/val',

    # Vimeo-90k dataset: http://toflow.csail.mit.edu/
    'vimeo-90k': _root / 'vimeo-90k/sequences',

    # UVG dataset: http://ultravideo.fi/#testsequences
    'uvg-1080p': _root / 'video/uvg/1080p-frames'
}


def get_dataset_path(name):
    if name in known_datasets:
        return known_datasets[name]
    else:
        import warnings
        warnings.warn(f"Dataset '{name}' not found in known_datasets.")
        return None
