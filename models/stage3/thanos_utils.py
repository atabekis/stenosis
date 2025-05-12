# thanos_utils.py

import torch
import torch.nn as nn

class LearnablePositionalEmbeddings(nn.Module):
    """
    Generates learnable 1D spatial and 1D temporal positional embeddings.
        - Spatial embeddings are for flattened tokens from an HxW grid.
        - Temporal embeddings are for T time steps.
    These are then broadcast and added to input tokens [B, T, N_spatial, C_features].
    """

    def __init__(self, d_model: int, max_spatial_tokens: int, max_temporal_tokens: int):
        """
        :param d_model: The feature dimension of the tokens (and embeddings).
        :param max_spatial_tokens: Maximum number of spatial tokens expected (e.g., max_H * max_W).
        :param max_temporal_tokens (int): Maximum number of temporal tokens (e.g., max T_clip).
        """
        super().__init__()
        self.d_model = d_model
        self.max_spatial_tokens = max_spatial_tokens
        self.max_temporal_tokens = max_temporal_tokens

        # learnable embedding for spatial tokens 1D seq. flattened HxW. each spatial position (0 to N_spatial_tokens-1) gets a d_model vector
        self.spatial_token_embed = nn.Embedding(max_spatial_tokens, d_model)

        # same for temporal tokens. each time step (0 to T_clip-1) gets a d_model vector
        self.temporal_embed = nn.Embedding(max_temporal_tokens, d_model)

        # initialize embeddings, e.g., truncated normal
        self.spatial_token_embed.weight.data.normal_(mean=0.0, std=0.02)
        self.temporal_embed.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self, B: int, T: int, Ns: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates spatial and temporal positional embeddings.
        :param B: batch size.
        :param T: number of temporal steps in the current input .
        :param Ns: number of spatial tokens in the current input (H * W).

        :return:
            - spatial_pe_to_add: Shape [1, 1, Ns, C_features] (or [B, 1, Ns, C_features] if batch-specific needed, usually shared)
            - temporal_pe_to_add: Shape [1, T, 1, C_features] (or [B, T, 1, C_features])
        """
        if Ns > self.max_spatial_tokens:
            raise ValueError(
                f"Number of spatial tokens {Ns} exceeds max_spatial_tokens {self.max_spatial_tokens} defined for embedding table.")
        if T > self.max_temporal_tokens:
            raise ValueError(
                f"Number of temporal steps {T} exceeds max_temporal_tokens {self.max_temporal_tokens} defined for embedding table.")

        # indices for lookup
        spatial_indices = torch.arange(Ns, device=device)  # [Ns]
        temporal_indices = torch.arange(T, device=device)  # [T]

        # lookup embeddings
        spatial_pe = self.spatial_token_embed(spatial_indices)  # [Ns, d_model]
        temporal_pe = self.temporal_embed(temporal_indices)  # [T, d_model]

        # reshape for broadcasting and addition to tokens [B, T, Ns, d_model]
        # original tokens: [B, T, Ns, d_model]
        # spatial pe: [Ns, d_model] -> [1, 1, Ns, d_model]
        spatial_pe_to_add = spatial_pe.unsqueeze(0).unsqueeze(0)

        # same for temporal
        # original: [B, T, Ns, d_model]
        # temporal pe: [T, d_model] -> [1, T, 1, d_model]
        temporal_pe_to_add = temporal_pe.unsqueeze(0).unsqueeze(2)

        return spatial_pe_to_add, temporal_pe_to_add

    def add_to_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Adds the generated positional embeddings to the input tokens.

        :param tokens: input tokens [B, T, Ns, C_features].
        :return: tokens with positional embeddings added.
        """
        B, T, Ns, C = tokens.shape
        if C != self.d_model:
            raise ValueError(f"Token feature dimension {C} does not match embedding d_model {self.d_model}")

        spatial_pe, temporal_pe = self.forward(B, T, Ns, tokens.device)

        # pytorch's broadcasting handles [1, 1, Ns, C] + [B, T, Ns, C] correctly by expanding the singleton dimensions.
        tokens_with_pe = tokens + spatial_pe + temporal_pe
        return tokens_with_pe
