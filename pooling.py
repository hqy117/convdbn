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

    This implements the pooling strategy from the original ConvDBN paper:
    1. Divide feature map into non-overlapping regions (e.g., 2x2)
    2. For each region, create multinomial vector [a1, a2, a3, a4, 0]
    3. Apply softmax and sample exactly one winner (or none)
    4. Create sparse detection map with winners
    5. Aggregate sparse regions to create pooled map
    """

    def __init__(self, spacing=2):
        super(MultinomialMaxPool2d, self).__init__()
        self.spacing = spacing
        
    def forward(self, hidden_activations):
        """
        Multinomial pooling with exact MATLAB algorithm implementation.

        Steps:
        1. Extract regions and create multinomial vectors [a1, a2, ..., 0]
        2. Apply softmax with numerical stability
        3. Perform multinomial sampling (one winner per region or none)
        4. Create sparse detection map and MATLAB-style pooled probability map

        Returns:
            sparse_detection: H (full resolution samples)
            pooled_probs: HPc (MATLAB-style probability aggregation)
            winner_info: for unpooling
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

        # Add "no winner" option (MATLAB: poshidprobs_mult(end,:) = 0)
        no_winner = torch.zeros(batch, channels, num_regions, 1, device=device, dtype=regions.dtype)
        regions_with_null = torch.cat([regions, no_winner], dim=-1)
        # Shape: [batch, channels, num_regions, spacing²+1]

        # Numerical stability (subtract max) - MATLAB: bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult,[],1))
        # MATLAB max([],1) means max across first dimension (choices), for each region separately
        max_vals = torch.max(regions_with_null, dim=-1, keepdim=True)[0]
        stabilized = regions_with_null - max_vals

        # Apply exp (MATLAB: exp(poshidprobs_mult))
        exp_vals = torch.exp(stabilized)

        # Normalize to get probabilities (MATLAB: bsxfun(@rdivide,P,sumP))
        sum_vals = torch.sum(exp_vals, dim=-1, keepdim=True)
        probs = exp_vals / (sum_vals + 1e-8)

        # Multinomial sampling using Gumbel-Max trick
        probs_flat = probs.view(-1, spacing*spacing + 1)

        # Gumbel-Max trick for efficient multinomial sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs_flat) + 1e-8) + 1e-8)
        winner_indices = torch.argmax(torch.log(probs_flat + 1e-8) + gumbel_noise, dim=-1)

        # Reshape winner indices
        winner_indices = winner_indices.view(batch, channels, pool_height, pool_width)

        # Create sparse detection map (HP in MATLAB - probabilities, not binary samples)
        sparse_detection = self._create_sparse_probability_map(
            probs, winner_indices, batch, channels, height, width
        )

        # MATLAB-style probability aggregation (HPc = sum of probabilities excluding "no winner")
        # This is the key missing piece in the original implementation!
        pooled_probs = self._create_matlab_pooled_probs(probs, pool_height, pool_width)

        # Store winner information for unpooling
        winner_info = {
            'winner_indices': winner_indices,
            'sparse_pattern': sparse_detection.detach(),
            'original_shape': (height, width),
            'pooled_shape': (pool_height, pool_width)
        }

        return sparse_detection, pooled_probs, winner_info
    
    def _create_sparse_detection_map_optimized(self, winner_indices, batch, channels, height, width):
        """Optimized sparse detection map creation using advanced indexing."""
        device = winner_indices.device
        spacing = self.spacing

        # Initialize sparse map
        sparse_map = torch.zeros(batch, channels, height, width, device=device, dtype=torch.float32)

        # Create masks for valid winners (not "no winner" option)
        valid_mask = winner_indices < (spacing * spacing)

        if valid_mask.any():
            # Get all valid indices at once
            b_idx, c_idx, ph_idx, pw_idx = torch.where(valid_mask)
            valid_winners = winner_indices[valid_mask]

            # Convert linear indices to 2D positions (vectorized)
            local_h = valid_winners // spacing
            local_w = valid_winners % spacing

            # Convert to global positions (vectorized)
            global_h = ph_idx * spacing + local_h
            global_w = pw_idx * spacing + local_w

            # Bounds checking (vectorized)
            valid_coords = (global_h < height) & (global_w < width)

            if valid_coords.any():
                # Apply valid coordinates
                b_final = b_idx[valid_coords]
                c_final = c_idx[valid_coords]
                h_final = global_h[valid_coords]
                w_final = global_w[valid_coords]

                # Set winners to 1 (vectorized assignment)
                sparse_map[b_final, c_final, h_final, w_final] = 1.0

        return sparse_map

    def _create_sparse_probability_map(self, probs, winner_indices, batch, channels, height, width):
        """
        Create sparse probability map (HP in MATLAB - full resolution probabilities).

        This returns the actual probability values at winner locations,
        not binary 0/1 samples like _create_sparse_detection_map_optimized.

        Args:
            probs: [batch, channels, num_regions, spacing²+1] - multinomial probabilities
            winner_indices: [batch, channels, pool_height, pool_width] - winner indices
            batch, channels, height, width: dimensions

        Returns:
            sparse_prob_map: [batch, channels, height, width] - probability values
        """
        device = winner_indices.device
        spacing = self.spacing
        pool_height = height // spacing
        pool_width = width // spacing

        # Initialize sparse probability map
        sparse_prob_map = torch.zeros(batch, channels, height, width, device=device, dtype=torch.float32)

        # Create masks for valid winners (not "no winner" option)
        valid_mask = winner_indices < (spacing * spacing)

        if valid_mask.any():
            # Get all valid indices at once
            b_idx, c_idx, ph_idx, pw_idx = torch.where(valid_mask)
            valid_winners = winner_indices[valid_mask]

            # Get the corresponding probabilities for the winners
            # probs shape: [batch, channels, num_regions, spacing²+1]
            # We need to get probs[b_idx, c_idx, region_idx, winner_idx]
            region_idx = ph_idx * pool_width + pw_idx  # Convert 2D pool coords to 1D region index
            winner_probs = probs[b_idx, c_idx, region_idx, valid_winners]

            # Convert linear indices to 2D positions (vectorized)
            local_h = valid_winners // spacing
            local_w = valid_winners % spacing

            # Convert to global positions (vectorized)
            global_h = ph_idx * spacing + local_h
            global_w = pw_idx * spacing + local_w

            # Bounds checking (vectorized)
            valid_coords = (global_h < height) & (global_w < width)

            if valid_coords.any():
                # Apply valid coordinates
                b_final = b_idx[valid_coords]
                c_final = c_idx[valid_coords]
                h_final = global_h[valid_coords]
                w_final = global_w[valid_coords]
                prob_final = winner_probs[valid_coords]

                # Set winner probabilities (not 1.0, but actual probability values)
                sparse_prob_map[b_final, c_final, h_final, w_final] = prob_final

        return sparse_prob_map



    def _create_matlab_pooled_probs(self, probs, pool_height, pool_width):
        """
        Create MATLAB-style pooled probabilities (HPc).

        MATLAB code equivalent:
        Pc = sum(P(1:end-1,:));  % Sum probabilities excluding "no winner"
        HPc = reshape(Pc, [pooled_height, pooled_width, channels]);

        Args:
            probs: [batch, channels, num_regions, spacing²+1] - multinomial probabilities
            pool_height, pool_width: pooled dimensions

        Returns:
            pooled_probs: [batch, channels, pool_height, pool_width] - MATLAB HPc equivalent
        """
        batch, channels = probs.shape[:2]

        # Sum probabilities excluding the last one ("no winner" option)
        # MATLAB: Pc = sum(P(1:end-1,:))
        prob_sums = torch.sum(probs[:, :, :, :-1], dim=-1)  # [batch, channels, num_regions]

        # Reshape to pooled dimensions
        # MATLAB: HPc = reshape(Pc, [pooled_height, pooled_width, channels])
        pooled_probs = prob_sums.view(batch, channels, pool_height, pool_width)

        # Numerical stability: ensure values are in [0, 1] range for bernoulli sampling
        # This handles floating point precision issues where sums might slightly exceed 1.0
        # pooled_probs = torch.clamp(pooled_probs, min=0.0, max=1.0)

        return pooled_probs




class SparseUnpool2d(nn.Module):
    """
    Sparse Unpooling for MATLAB-style ConvRBM.

    This restores the sparse detection pattern using stored information
    from the forward pass, following the MATLAB implementation approach.
    """

    def __init__(self, spacing=2):
        super(SparseUnpool2d, self).__init__()
        self.spacing = spacing

    def forward(self, pooled_map, winner_info):
        """
        Optimized sparse detection map restoration.

        Args:
            pooled_map: [batch, channels, pool_height, pool_width]
            winner_info: dict containing sparse pattern and winner indices

        Returns:
            sparse_detection: [batch, channels, height, width] - restored sparse map
        """
        batch, channels = pooled_map.shape[:2]
        original_height, original_width = winner_info['original_shape']
        stored_sparse_pattern = winner_info['sparse_pattern']

        # Initialize output with correct dtype
        sparse_detection = torch.zeros(
            batch, channels, original_height, original_width,
            device=pooled_map.device, dtype=pooled_map.dtype
        )

        # Fast restoration using boolean indexing
        # Create mask for active pooled units
        active_mask = pooled_map > 0.5

        if active_mask.any():
            # Get indices of active pooled units
            b_idx, c_idx, ph_idx, pw_idx = torch.where(active_mask)

            # Compute region boundaries (vectorized)
            start_h = ph_idx * self.spacing
            end_h = start_h + self.spacing
            start_w = pw_idx * self.spacing
            end_w = start_w + self.spacing

            # Bounds checking (vectorized)
            end_h = torch.clamp(end_h, max=original_height)
            end_w = torch.clamp(end_w, max=original_width)

            # Ultra-fast vectorized region restoration
            # Instead of loops, use advanced indexing with broadcasting

            # Create index tensors for all active regions at once
            spacing = self.spacing
            device = pooled_map.device

            # Generate all coordinates within each active region
            region_h_coords = torch.arange(spacing, device=device).view(1, -1, 1).expand(len(b_idx), -1, spacing)
            region_w_coords = torch.arange(spacing, device=device).view(1, 1, -1).expand(len(b_idx), spacing, -1)

            # Add region offsets
            global_h_coords = start_h.view(-1, 1, 1) + region_h_coords
            global_w_coords = start_w.view(-1, 1, 1) + region_w_coords

            # Flatten coordinates
            flat_h = global_h_coords.flatten()
            flat_w = global_w_coords.flatten()

            # Repeat batch and channel indices for each coordinate
            flat_b = b_idx.view(-1, 1, 1).expand(-1, spacing, spacing).flatten()
            flat_c = c_idx.view(-1, 1, 1).expand(-1, spacing, spacing).flatten()

            # Bounds checking
            valid_mask = (flat_h < original_height) & (flat_w < original_width) & (flat_h >= 0) & (flat_w >= 0)

            if valid_mask.any():
                # Apply bounds checking
                valid_b = flat_b[valid_mask]
                valid_c = flat_c[valid_mask]
                valid_h = flat_h[valid_mask]
                valid_w = flat_w[valid_mask]

                # Vectorized assignment - copy from stored pattern
                sparse_detection[valid_b, valid_c, valid_h, valid_w] = stored_sparse_pattern[valid_b, valid_c, valid_h, valid_w]

        return sparse_detection
