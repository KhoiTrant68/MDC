import torch
import sys

sys.path.append('.\\models\\MAE')
import os
import models_mae_new


class Mae:
    def __init__(self, arch='mae_vit_large_patch16'):
        self.arch = arch

    def prepare_model(self, gan_loss=False):
        # Get model weight path
        model_path = self.get_model_path(gan_loss)

        # Check if the checkpoint file exists, if not, download it
        if not os.path.exists(model_path):
            download_link = self.get_download_link(model_path)
            os.system(f"wget -nc -P weights\\mae\\ {download_link}")

        # Build model
        model = getattr(models_mae_new, self.arch)()

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

        return model

    @staticmethod
    def get_model_path(gan_loss):
        if gan_loss:
            return "weights\\mae\\mae_visualize_vit_large_ganloss.pth"
        else:
            return "weights\\mae\\mae_visualize_vit_large.pth"

    @staticmethod
    def get_download_link(chkpt_dir):
        if "ganloss" in chkpt_dir:
            return "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth"
        else:
            return "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth"
