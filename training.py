import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models.Compression.MDC import MCM
from models.Compression.common import scaler, distributed, model_utils, pos_embed
from models.Compression.loss import rd_loss
from utils.dataloader import get_image_dataset
from utils.engine import train_one_epoch, val_one_epoch


def save_checkpoint(state, is_best, filename):
    filename1 = filename[:-8] + '_epoch' + str(state['epoch']) + filename[-8:]
    torch.save(state, filename1)
    if is_best:
        shutil.copyfile(filename1, filename[:-8] + "_best" + filename[-8:])


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE fine-tuning for image compression', add_help=False)

    # Dataset
    parser.add_argument("-d", "--dataset", type=str,
                        required=True, help="Training dataset path")

    # Training Options
    parser.add_argument("-e", "--epochs", default=100, type=int,
                        help="Number of epochs (default: %(default)s)")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Accumulate gradient iterations for effective batch size")
    parser.add_argument("--learning-rate", "-lr", default=1e-4, type=float,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-4,
                        help="Bit-rate distortion parameter (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=8,
                        help="Test batch size (default: %(default)s)")
    parser.add_argument("--aux-learning-rate", default=1e-4, type=float,
                        help="Auxiliary loss learning rate (default: %(default)s)")

    # Miscellaneous Options
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save model to disk")
    parser.add_argument("--save_path", type=str,
                        default="ckpt/model.pth.tar", help="Where to Save model")
    parser.add_argument("--seed", default=0, type=int,
                        help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float,
                        help="Gradient clipping max norm (default: %(default)s")

    # Checkpoints and Logging
    parser.add_argument("--checkpoint", type=str, default='',
                        help="Path to a checkpoint")
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='Path where to store TensorBoard logs')

    # Hardware and Data Loading Options
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Model Options
    parser.add_argument('--num_keep_patches', type=int, default=144,
                        help='Number of keep patches to input the model')

    # Augmentation parameters
    parser.add_argument("--input_size", type=int, default=224, required=True,
                        help="Size of the input image")
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    return parser


def main(args):
    print('Job directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # Set up device and random seed
    device = torch.device(args.device)
    seed = args.seed + distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data preprocessing and dataset creation
    cudnn.benchmark = True
    train_dataset = get_image_dataset(mode="train",
                                      dataset_path=args.dataset)

    val_dataset = get_image_dataset(mode="val",
                                    dataset_path=args.dataset)

    # Setting up data samplers
    num_tasks = distributed.get_world_size()  # GPU's RAM
    global_rank = distributed.get_rank()  # GPU
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    # Configuring a writer for logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    # Set up dataloader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=sampler_val,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mcm = MCM(num_keep_patches=args.num_keep_patches)

    '''load mae encoder model'''
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['model']

        state_dict = mcm.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed.interpolate_pos_embed(mcm, checkpoint_model)
        # msg = mcm.load_state_dict(checkpoint_model)
        # print(msg)

    mcm.to(device)
    loss_scaler = scaler.NativeScaler()
    optimizer, aux_optimizer = model_utils.configure_optimizers(mcm, args)
    model_utils.load_model(args=args, model=mcm, optimizer=optimizer,
                           aux_optimizer=aux_optimizer, loss_scaler=loss_scaler)

    criterion = rd_loss.RateDistortionLoss(lmbda=args.lmbda)
    best_loss = 1e10

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs, 5):
        train_one_epoch(mcm, criterion, train_dataloader, optimizer, aux_optimizer, epoch, loss_scaler,
                        args.clip_max_norm, writer=writer, args=args)
        out = val_one_epoch(epoch, val_dataloader, mcm, criterion)

        if args.output_dir:
            if out['loss'] < best_loss:
                model_utils.save_model(args=args, epoch=epoch, model=mcm, optimizer=optimizer,
                                       aux_optimizer=aux_optimizer,
                                       loss_scaler=loss_scaler)
                best_loss = out['loss']


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
