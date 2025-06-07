# thanos_transformer.py

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SpatioTemporalAttentionBlock(nn.Module):
    """
    Applies factorized spatiotemporal attention to tokenized features from a single FPN level.
    It expects tokens to already have positional embeddings added.
    We follow the process:
    1. Reshape [B, T, Ns, C] -> [B*T, Ns, C] for spatial attention.
    2. Apply Spatial Transformer Encoder.
    3. Reshape [B*T, Ns, C] -> [B, T, Ns, C] -> [B*Ns, T, C] for temporal attention.
    4. Apply Temporal Transformer Encoder.
    5. Reshape back to [B, T, Ns, C].
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int,
                 num_spatial_layers: int,
                 num_temporal_layers: int,
                 dropout_rate: float,
                 use_gradient_checkpointing: bool = False,
                 ):
        """
        :param d_model: Feature dimension (same as input token C_features and PE d_model).
        :param n_head: Number of attention heads for both spatial and temporal transformers.
        :param dim_feedforward: Dimension of the feedforward network model in nn.TransformerEncoderLayer.
        :param num_spatial_layers: Number of nn.TransformerEncoderLayer for spatial attention.
        :param num_temporal_layers: Number of nn.TransformerEncoderLayer for temporal attention.
        :param dropout_rate: Dropout rate for nn.TransformerEncoderLayer.
        :param use_gradient_checkpointing: Whether to use gradient checkpointing for transformer blocks.
        """
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # set up SPATIAL transformer
        if num_spatial_layers > 0:
            spatial_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                batch_first=True  # [batch, seq_len, features]
            )
            self.spatial_transformer = nn.TransformerEncoder(
                encoder_layer=spatial_encoder_layer,
                num_layers=num_spatial_layers
            )
        else:
            self.spatial_transformer = nn.Identity()  # pass through if no spatial layers

        # set up TEMPORAL transformer
        if num_temporal_layers > 0:
            temporal_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                batch_first=True
            )
            self.temporal_transformer = nn.TransformerEncoder(
                encoder_layer=temporal_encoder_layer,
                num_layers=num_temporal_layers
            )
        else:
            self.temporal_transformer = nn.Identity()

        # self.norm_spatial_out = nn.LayerNorm(d_model) if num_spatial_layers > 0 else nn.Identity()
        # self.norm_temporal_out = nn.LayerNorm(d_model) if num_temporal_layers > 0 else nn.Identity()


    def _checkpointed_forward(self, module_to_ckpt: nn.Module, *args_for_module):
        """Helper to apply checkpointing to a given module's forward pass"""
        if self.use_gradient_checkpointing and self.training and not isinstance(module_to_ckpt, nn.Identity):
            return checkpoint(module_to_ckpt, *args_for_module, use_reentrant=False)
        else:
            return module_to_ckpt(*args_for_module)


    def forward(self, tokens_b_t_ns_c: torch.Tensor) -> torch.Tensor:
        """
        :param tokens_b_t_ns_c: Input tokens with positional embeddings already added.
                                Shape: [B, T, Ns, C_features(d_model)]. (Ns = N_spatial_tokens)
        :returns: Enhanced tokens [B, T, Ns, C_features].
        """

        B, T, Ns, C = tokens_b_t_ns_c.shape
        if C != self.d_model:
            raise ValueError(f"Input token feature dimension {C} does not match block's d_model {self.d_model}")

        # 1. spatial self-attention
        spatial_input = tokens_b_t_ns_c.contiguous().view(B * T, Ns, C) # [B, T, Ns, C] -> [B*T, Ns, C]

        spatial_output = self._checkpointed_forward(self.spatial_transformer, spatial_input)

        # 2. temporal self-attention
        temporal_input_reshaped = spatial_output.view(B, T, Ns, C) # [B*T, Ns, C] -> [B, T, Ns, C]
        temporal_input = temporal_input_reshaped.permute(0, 2, 1, 3).contiguous().view(B * Ns, T, C) # -> [B*Ns, T, C]

        temporal_output = self._checkpointed_forward(self.temporal_transformer, temporal_input)  # [B*Ns, T, C]

        # 3. reshape back to original: [B*Ns, T, C] -> [B, Ns, T, C] -> [B, T, Ns, C]
        return temporal_output.view(B, Ns, T, C).permute(0, 2, 1, 3).contiguous()


