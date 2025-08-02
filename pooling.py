"""
Probabilistic Max-Pooling Implementation
========================================

This module implements the multinomial probabilistic max-pooling as described
in the original ConvDBN paper.

Key concepts:
1. Detection Layer (h): Sparse representation at full resolution
2. Pooling Layer (p): Aggregated representation
3. Multinomial sampling within each pooling region
4. No explicit unpooling - use stored sparse patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultinomialMaxPool2d(nn.Module):
    """
    Multinomial Probabilistic Max-Pooling.

    This implements the pooling strategy from the original ConvDBN paper,
    which creates a sparse output by performing multinomial sampling within
    non-overlapping regions of the input feature map. Only one "winner"
    is chosen per region.
    """

    def __init__(self, spacing=2):
        super(MultinomialMaxPool2d, self).__init__()
        self.spacing = spacing
        
    def forward(self, hidden_activations):
        """
        Performs multinomial pooling.

        Returns:
            sparse_detection: The full-resolution sparse probability map (HP in MATLAB).
            pooled_probs: The aggregated probability map for the pooled layer (HPc in MATLAB).
            winner_info: A dictionary containing information needed for unpooling.
        """
        batch, channels, height, width = hidden_activations.shape
        device = hidden_activations.device
        spacing = self.spacing

        # Calculate pooled dimensions
        pool_height = height // spacing
        pool_width = width // spacing

        # Efficient region extraction using unfold (vectorized)
        regions = F.unfold(
            hidden_activations,
            kernel_size=spacing,
            stride=spacing
        )  # [batch*channels, spacing², pool_height*pool_width]

        # Reshape to match MATLAB format
        regions = regions.view(batch, channels, spacing*spacing, pool_height*pool_width)
        regions = regions.permute(0, 1, 3, 2)  # [batch, channels, num_regions, spacing²]

        num_regions = pool_height * pool_width

        # Add a "no winner" option with zero probability, matching MATLAB's logic.
        no_winner = torch.zeros(batch, channels, num_regions, 1, device=device, dtype=regions.dtype)
        regions_with_null = torch.cat([regions, no_winner], dim=-1)

        # Apply numerically stable softmax to get probabilities for each region.
        max_vals = torch.max(regions_with_null, dim=-1, keepdim=True)[0]
        stabilized = regions_with_null - max_vals
        exp_vals = torch.exp(stabilized)
        probs = exp_vals / (torch.sum(exp_vals, dim=-1, keepdim=True) + 1e-8)

        # Perform multinomial sampling using the Gumbel-Max trick, which is an
        # efficient and differentiable way to sample from a categorical distribution.
        probs_flat = probs.view(-1, self.spacing * self.spacing + 1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs_flat) + 1e-8) + 1e-8)
        winner_indices = torch.argmax(torch.log(probs_flat + 1e-8) + gumbel_noise, dim=-1)
        winner_indices = winner_indices.view(batch, channels, pool_height, pool_width)

        # Create the full-resolution sparse probability map (HP) using the winners.
        sparse_detection = self._create_sparse_probability_map(
            probs, winner_indices, batch, channels, height, width
        )

        # Create the pooled probability map (HPc) by summing probabilities in each region.
        pooled_probs = self._create_matlab_pooled_probs(probs, pool_height, pool_width)

        # Store information needed for the unpooling operation.
        winner_info = {
            'winner_indices': winner_indices,
            'sparse_pattern': sparse_detection.detach(), # Detach to prevent gradient flow
            'original_shape': (height, width),
            'pooled_shape': (pool_height, pool_width)
        }

        return sparse_detection, pooled_probs, winner_info
    
    def _create_sparse_probability_map(self, probs, winner_indices, batch, channels, height, width):
        """
        Creates the full-resolution sparse probability map (HP in MATLAB).

        This function places the original probability values at the winner locations
        in a zero-initialized map of the original input dimensions.
        """
        device = winner_indices.device
        spacing = self.spacing
        pool_height = height // spacing
        pool_width = width // spacing

        # Initialize sparse probability map
        sparse_prob_map = torch.zeros(batch, channels, height, width, device=device, dtype=torch.float32)

        # Mask to select only valid winners (i.e., not the "no winner" option).
        valid_mask = winner_indices < (spacing * spacing)

        if valid_mask.any():
            # Get all valid indices at once
            b_idx, c_idx, ph_idx, pw_idx = torch.where(valid_mask)
            valid_winners = winner_indices[valid_mask]

            # Get the corresponding probabilities for the winners.
            region_idx = ph_idx * (width // spacing) + pw_idx
            winner_probs = probs[b_idx, c_idx, region_idx, valid_winners]

            # Convert winner's linear index within a region to 2D coordinates.
            local_h = valid_winners // spacing
            local_w = valid_winners % spacing

            # Convert pooled coordinates to original, full-resolution coordinates.
            global_h = ph_idx * spacing + local_h
            global_w = pw_idx * spacing + local_w

            # Set the probability value at the winner's global location.
            sparse_prob_map[b_idx, c_idx, global_h, global_w] = winner_probs
            
        return sparse_prob_map

    def _create_matlab_pooled_probs(self, probs, pool_height, pool_width):
        """
        Creates MATLAB-style pooled probabilities (HPc).

        This is done by summing the probabilities of all potential winners
        (excluding the "no winner" option) within each pooling region.
        """
        # Sum probabilities excluding the last one ("no winner" option).
        prob_sums = torch.sum(probs[:, :, :, :-1], dim=-1)

        # Reshape to the dimensions of the pooled map.
        pooled_probs = prob_sums.view(probs.shape[0], probs.shape[1], pool_height, pool_width)

        # Clamp the result to handle potential floating-point inaccuracies, ensuring
        # all values are strictly within the [0, 1] range for bernoulli sampling.
        return torch.clamp(pooled_probs, min=0.0, max=1.0)


class SparseUnpool2d(nn.Module):
    """
    Sparse Unpooling for MATLAB-style ConvRBM.

    Restores the sparse activation pattern using information stored
    during the forward pooling pass.
    """

    def __init__(self, spacing=2):
        super(SparseUnpool2d, self).__init__()
        self.spacing = spacing

    def forward(self, pooled_map, winner_info):
        """
        Reconstruct the sparse detection map from the pooled activation map
        and the winner indices saved during the forward pooling pass.
        """
        spacing = self.spacing
        winner_indices = winner_info['winner_indices']
        height, width = winner_info['original_shape']
        batch, channels, pool_h, pool_w = pooled_map.shape

        # Initialize empty sparse map
        sparse_map = pooled_map.new_zeros(batch, channels, height, width)

        # Determine valid winners (exclude the "no-winner" option)
        valid_mask = winner_indices < (spacing * spacing)
        if valid_mask.any():
            b_idx, c_idx, ph_idx, pw_idx = torch.where(valid_mask)
            local_idx = winner_indices[valid_mask]

            # Coordinates inside each pooling cell
            lh = local_idx // spacing
            lw = local_idx % spacing

            # Map back to original spatial coordinates
            gh = ph_idx * spacing + lh
            gw = pw_idx * spacing + lw

            sparse_map[b_idx, c_idx, gh, gw] = pooled_map[b_idx, c_idx, ph_idx, pw_idx]

        return sparse_map
