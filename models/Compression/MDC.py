import warnings

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv3x3, subpel_conv3x3
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from timm.models.vision_transformer import PatchEmbed, Block

from common.mage_modules import Block, LabelSmoothingCrossEntropy, BertEmbeddings, MlmLayer
from models.Compression.common.pos_embed import get_2d_sincos_pos_embed
from models.Compression.loss.vgg import cal_features_loss
from taming.models.vqgan import VQModel

warnings.filterwarnings("ignore")


class MDC(CompressionModel):
    """
    Masked Generative Compression Model with Vision Transformer backbone

    This class inherits from MAsked Generative Encoder in *LTH14/mage* class.
    See the original paper and the `MAGE' documentation
    <https://github.com/LTH14/mage/tree/main> for an introduction.
    """

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 encoder_embed_dim=1024,
                 encoder_depth=24,
                 encoder_num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 latent_depth=384,
                 hyperprior_depth=192,
                 num_slices=12,
                 mask_ratio_min=0.5,
                 mask_ratio_max=1.0,
                 mask_ratio_mu=0.55,
                 mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt'):
        super().__init__()

        # Initialize frozen stage
        self.frozen_stages = -1

        # Entropy model
        self.entropy_bottleneck = EntropyBottleneck(hyperprior_depth)
        self.gaussian_conditional = GaussianConditional(None)
        self.max_support_slices = self.num_slices // 2

        # Compression Modules
        # G_a Module
        self.g_a = nn.Sequential(
            nn.Conv2d(encoder_embed_dim,
                      int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 3 / 4),
                      kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 3 / 4),
                      int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 2 / 4),
                      kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 2 / 4),
                      decoder_embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(decoder_embed_dim, latent_depth,
                      kernel_size=1, stride=1, padding=0),
        )

        # G_s Module
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                latent_depth, decoder_embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_embed_dim,
                               int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 2 / 4),
                               kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 2 / 4),
                               int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 3 / 4),
                               kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(int(decoder_embed_dim + (encoder_embed_dim - decoder_embed_dim) * 3 / 4),
                               encoder_embed_dim, kernel_size=1, stride=1, padding=0),
        )

        # H_a Module
        self.h_a = nn.Sequential(
            conv3x3(latent_depth, latent_depth, stride=1),
            nn.GELU(),
            conv3x3(latent_depth, int(hyperprior_depth + (latent_depth - hyperprior_depth) * 3 / 4), stride=1),
            nn.GELU(),
            conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) * 3 / 4),
                    int(hyperprior_depth + (latent_depth - hyperprior_depth) * 2 / 4), stride=2),
            nn.GELU(),
            conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) * 2 / 4),
                    int(hyperprior_depth + (latent_depth - hyperprior_depth) / 4), stride=1),
            nn.GELU(),
            conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) / 4),
                    hyperprior_depth, stride=2),
        )

        # H_s Module
        self.h_s_mean = nn.Sequential(
            conv3x3(hyperprior_depth, int(hyperprior_depth + (latent_depth - hyperprior_depth) / 4), stride=1),
            nn.GELU(),
            subpel_conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) / 4),
                           int(hyperprior_depth + (latent_depth - hyperprior_depth) * 2 / 4), r=2),
            nn.GELU(),
            conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) * 2 / 4),
                    int(hyperprior_depth + (latent_depth - hyperprior_depth) * 3 / 4), stride=1),
            nn.GELU(),
            subpel_conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) * 3 / 4),
                           latent_depth, r=2),
            nn.GELU(),
            conv3x3(latent_depth, latent_depth, stride=1),
        )

        self.h_s_scale = nn.Sequential(
            conv3x3(hyperprior_depth, int(hyperprior_depth + (latent_depth - hyperprior_depth) / 4), stride=1),
            nn.GELU(),
            subpel_conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) / 4),
                           int(hyperprior_depth + (latent_depth - hyperprior_depth) * 2 / 4), r=2),
            nn.GELU(),
            conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) * 2 / 4),
                    int(hyperprior_depth + (latent_depth - hyperprior_depth) * 3 / 4), stride=1),
            nn.GELU(),
            subpel_conv3x3(int(hyperprior_depth + (latent_depth - hyperprior_depth) * 3 / 4),
                           latent_depth, r=2),
            nn.GELU(),
            conv3x3(latent_depth, latent_depth, stride=1),
        )

        # CC_Transform Module
        self.cc_transform_mean = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(latent_depth + (latent_depth // num_slices) * min(i, num_slices // 2)),
                          int(latent_depth // num_slices * (num_slices // 2 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 3 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 3 / 4 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 2 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 2 / 4 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 1 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 1 / 4 + 1)),
                          int(latent_depth // num_slices),
                          kernel_size=3, stride=1, padding=1),
            ) for i in range(num_slices)
        ])

        self.cc_transform_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(latent_depth + (latent_depth // num_slices) * min(i, num_slices // 2)),
                          int(latent_depth // num_slices * (num_slices // 2 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 3 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 3 / 4 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 2 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 2 / 4 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 1 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 1 / 4 + 1)),
                          int(latent_depth // num_slices),
                          kernel_size=3, stride=1, padding=1),
            ) for i in range(num_slices)
        ])

        # LRP Transform Module
        self.lrp_transform = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(latent_depth + (latent_depth // num_slices) * min(i + 1, num_slices // 2 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 3 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 3 / 4 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 2 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 2 / 4 + 1)),
                          int(latent_depth // num_slices * (num_slices // 2 * 1 / 4 + 1)),
                          kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(latent_depth // num_slices * (num_slices // 2 * 1 / 4 + 1)),
                          int(latent_depth // num_slices),
                          kernel_size=3, stride=1, padding=1),
            ) for i in range(num_slices)
        ])

        # Initialize freeze stage status
        self._freeze_stages()

        # ------------------------------------ Initialize MAGE layers ------------------------------------
        # VQGAN specifics
        config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        self.codebook_size = config.params.n_embed
        vocab_size = self.codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        self.fake_class_label = self.codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=encoder_embed_dim,
                                        max_position_embeddings=256 + 1,
                                        dropout=0.1)

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu, scale=mask_ratio_std
        )

        # Define encoder blocks
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)

        # Define decoder blocks
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # ----------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=encoder_embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # Initialize weight
        self.initialize_weights()

    def _freeze_stages(self):
        """
        Freeze the stages
        """
        if self.frozen_stages >= 0:
            self.encoder_embed.eval()
            for param in self.encoder_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(x)

        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)
        gt_indices = token_indices.clone().detach().long()

        # masking
        bsz, seq_len = token_indices.size()
        mask_ratio_min = self.mask_ratio_min
        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens - 1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens - 1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz * num_dropped_tokens and token_all_mask.sum() == bsz * num_masked_tokens:
                break
            else:
                print("Rerandom the noise!")
        # print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        # print("Masekd num token:", torch.sum(token_indices == self.mask_token_label, dim=1))

        # concate class token
        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = self.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        token_indices = token_indices.long()
        # bert embedding
        input_embeddings = self.token_emb(token_indices)
        # print("Input embedding shape:", input_embeddings.shape)
        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)

        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        # print("Logits shape:", x.shape)

        return x

    @staticmethod
    def forward_loss(imgs, preds):
        # Calculate SSIM loss
        ssim = SSIM(
            win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
        )
        ssim_loss = 1 - ssim(preds, imgs)

        # Calculate L1 loss
        l1_loss = F.l1_loss(preds, imgs)

        # Calculate feature loss (ensure 'cal_features_loss' is defined appropriately)
        feature_loss = cal_features_loss(preds, imgs)

        return {
            "ssim_loss": ssim_loss,
            "l1_loss": l1_loss,
            "feature_loss": feature_loss
        }

    def forward(self, imgs):
        # Encoder
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs)

        # LIC
        y = (latent.view(-1,
                         int(self.num_keep_patches ** 0.5),
                         int(self.num_keep_patches ** 0.5),
                         self.encoder_embed_dim).permute(0, 3, 1, 2).contiguous())

        # Apply G_a module
        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        # Apply H_a module
        z = self.h_a(y)

        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        # Apply H_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Compress using slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihoods = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices
                              if self.max_support_slices < 0
                              else y_hat_slices[: self.max_support_slices])

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            # Calculate y_slice_likelihood
            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, sigma, mu)
            y_likelihoods.append(y_slice_likelihood)

            # Calculate y_hat_slice
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            # Calculate lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)

        # Apply G_s module
        y_hat = self.g_s(y_hat)
        y_hat = (y_hat.permute(0, 2, 3, 1).contiguous()
                 .view(-1, self.num_keep_patches, self.encoder_embed_dim))

        # Decoder
        logits = self.forward_decoder(y_hat, token_drop_mask, token_all_mask)
        with torch.no_grad():
            x_hat = self.vqgan.decode(logits)
        loss = self.forward_loss(imgs, x_hat)

        return {
            "loss": loss,
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
            "x_hat": x_hat,
        }

    def compress(self, imgs):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for autoregressive models. The entropy coder is run sequentially "
                "on GPU "
            )

        # Encoder MCM
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs)

        # LIC
        y = (latent.view(-1,
                         int(self.num_keep_patches ** 0.5),
                         int(self.num_keep_patches ** 0.5),
                         self.encoder_embed_dim).permute(0, 3, 1, 2).contiguous())

        # Apply G_a module
        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        # Apply H_a module
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Apply H_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Compress using slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        # CDF
        cdfs = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(
            -1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        # BufferedRansEncoder Module
        encoder = BufferedRansEncoder()

        # Compress using slices
        y_strings = []
        symbols_list = []
        indexes_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices
                              if self.max_support_slices < 0
                              else y_hat_slices[: self.max_support_slices])

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(sigma)
            y_q_slice = self.gaussian_conditional.quantize(
                y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            # Calculate lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdfs, cdf_lengths, offsets
        )

        # Get y_string
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "string": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "gt_indices": gt_indices,
            "token_drop_mask": token_drop_mask,
            "token_all_mask": token_all_mask
        }

    def decompress(self, strings, shape, token_drop_mask, token_all_mask):
        assert isinstance(strings, list) and len(strings) == 2

        # Decompress
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        # Apply h_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]
        y_hat_slices = []

        # Cdf
        cdfs = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(
            -1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        # Decoder bit stream
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Decompress using slices
        for slice_index in range(self.num_slices):
            # for slice_index in range(3):
            support_slices = (y_hat_slices
                              if self.max_support_slices < 0
                              else y_hat_slices[: self.max_support_slices])

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            # Index
            index = self.gaussian_conditional.build_indexes(sigma)

            # Revert string indices
            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdfs, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            # Lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        # Apply G_s module
        y_hat = self.g_s(y_hat)
        y_hat = (
            y_hat.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.num_keep_patches, self.encoder_embed_dim)
        )

        # Decoder
        logits = self.forward_decoder(y_hat, token_drop_mask, token_all_mask)
        with torch.no_grad():
            x_hat = self.vqgan.decode(logits)

        return {"x_hat": x_hat}
