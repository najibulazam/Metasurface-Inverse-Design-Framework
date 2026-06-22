"""
simmim.py

Masked-image-modeling wrapper around the small ViT, adapted from the
original repo's model/simmim.py. Trains the encoder to reconstruct masked
parts of a Jones-matrix "image" given the unmasked parts -- this is the
self-supervised pretraining stage that lets the model later be fine-tuned
to predict structural parameters with much less labeled data.

A "mask" here is a binary [n_wave, 6] array: 1 = masked (hidden from the
model, the part it must reconstruct), 0 = visible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vit_small import VisionTransformerSmall


class VisionTransformerForSimMIM(VisionTransformerSmall):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_para == 0, "SimMIM encoder must be built with num_para=0 (no prediction head)."
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask):
        _, _, H, W = x.shape
        x = self.patch_embed(x)
        B, C, H_patch, W_patch = x.shape
        x = x.flatten(2).transpose(1, 2)
        L = x.shape[1]

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, 1:]  # drop cls token
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)
        return x


class SimMIM(nn.Module):
    """loss_type: 0 = loss over the whole image, 1 = loss over masked region
    only (the typical/recommended choice -- forces the model to actually
    use context to reconstruct, rather than just copying visible pixels),
    2 = loss over unmasked region only (rarely useful, kept for parity
    with the original repo's option).
    """
    def __init__(self, encoder, loss_type=1):
        super().__init__()
        self.encoder = encoder
        self.in_chans = encoder.in_chans
        self.patch_size = encoder.patch_size
        self.loss_type = loss_type
        self.decoder = nn.Conv2d(encoder.embed_dim, encoder.in_chans, kernel_size=1)

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        mask_expanded = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2)

        if self.loss_type == 0:
            loss = F.l1_loss(x, x_rec, reduction="mean")
        elif self.loss_type == 1:
            loss_map = F.l1_loss(x, x_rec, reduction="none")
            loss = (loss_map * mask_expanded).sum() / (mask_expanded.sum() + 1e-5) / self.in_chans
        elif self.loss_type == 2:
            loss_map = F.l1_loss(x, x_rec, reduction="none")
            inv_mask = 1 - mask_expanded
            loss = (loss_map * inv_mask).sum() / (inv_mask.sum() + 1e-5) / self.in_chans
        else:
            raise ValueError("loss_type must be 0, 1, or 2")

        return loss, x_rec


def build_simmim(encoder_or_n_wave, embed_dim=256, depth=6, num_heads=8, loss_type=1):
    if isinstance(encoder_or_n_wave, VisionTransformerSmall):
        encoder = encoder_or_n_wave
    else:
        n_wave = encoder_or_n_wave
        encoder = VisionTransformerForSimMIM(
            img_size=(n_wave, 6), patch_size=1, in_chans=1, num_para=0,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        )
    return SimMIM(encoder, loss_type=loss_type)
