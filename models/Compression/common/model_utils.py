from pathlib import Path

import torch
import torch.optim as optim

from models.Compression.common.distributed import get_rank


def load_model(args, model, optimizer, aux_optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model.load_state_dict(checkpoint['model'])

        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            # lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print("With optim & sched!")


def save_model(args, epoch, model, optimizer, aux_optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    if not Path(output_dir).exists():
        Path.mkdir(output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        # checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        checkpoint_paths = [output_dir / ('best_model.pth')]

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'aux_optimizer': aux_optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                # 'scheduler': lr_scheduler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        # model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" %
        #                                                     epoch_name, client_state=client_state)
        model.save_checkpoint(save_dir=args.output_dir, tag="best_model", client_state=client_state)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_main_process():
    return get_rank() == 0


def configure_optimizers(mcm, args):
    """Return two optimizers"""
    # MAE
    parameters = {n for n, p in mcm.named_parameters(
    ) if not n.endswith(".quantiles") and p.requires_grad}

    # LIC
    aux_parameters = {n for n, p in mcm.named_parameters(
    ) if n.endswith(".quantiles") and p.requires_grad}

    params_dict = dict(mcm.named_parameters())
    inter_params = parameters & aux_parameters

    assert len(inter_params) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer
