import random
from math import ceil
from functools import partial
from itertools import zip_longest

import torch
from torch import nn
import torch.nn.functional as F
# from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

import pdb

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class SelfResidualVQ(nn.Module):
    """ Quantizes the residuals using the same codebook iteratively """
    def __init__(
        self,
        *,
        dim,
        num_quantizers = 1,
        codebook_dim = None,
        use_cosine_sim = True,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, 'self residual vq is not compatible with multi-headed codes'
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap

        self.layer = VectorQuantize(dim=codebook_dim, codebook_dim=codebook_dim, use_cosine_sim = use_cosine_sim, accept_image_fmap=accept_image_fmap, **kwargs)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    def forward(
        self,
        x,
        indices=None,
        return_all_codes=False, # TODO: Extend code when need to return all codes.
        sample_codebook_temp=None,
        freeze_codebook=False,
        mask=None,
    ):
        residual = self.project_in(x)
        all_indices = []
        commit_losses = []
        quantized = 0.

        for _ in range(self.num_quantizers):
            quantized_output, indices, commit_loss = self.layer(
                residual,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=freeze_codebook,
                mask=mask
            )
            residual = residual - quantized_output

            all_indices.append(indices)
            commit_losses.append(commit_loss)
            quantized = quantized + quantized_output

        commit_losses = torch.stack(commit_losses, dim=-1)
        all_indices = torch.stack(all_indices, dim=-1)

        quantized = self.project_out(quantized)

        return quantized, commit_losses, all_indices #[4, 50, 50], [4, 50, 1], [1, 1]

if __name__ == '__main__':
    self_residual_vq = SelfResidualVQ(
        dim = 256,
        num_quantizers = 8,       # times to do quantize
        codebook_size = 1024,    # codebook size
    )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = self_residual_vq(x)

    print(quantized.shape, indices.shape, commit_loss.shape)