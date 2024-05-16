import torch


def de_normalize(batch):
    # De normalize for regular normalization
    batch = (batch + 1.0) / 2.0 * 255.0
    return batch


def normalize_batch(batch):
    # Normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= mean
    batch = batch / std
    return batch