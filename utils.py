"""
Utility Functions for ConvRBM Training

This module contains helper functions for contrastive divergence training,
correlation computation, and other utilities needed for ConvRBM.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_correlation(visible_data, hidden_activations, kernel_size):
    """
    Compute correlation between visible data and hidden activations using
    an explicit loop-based implementation to match MATLAB's conv2 logic exactly.
    This is then normalized by the batch size.
    """
    batch_size, in_channels, vis_height, vis_width = visible_data.shape
    _, out_channels, conv_height, conv_width = hidden_activations.shape

    if vis_height - kernel_size + 1 != conv_height or vis_width - kernel_size + 1 != conv_width:
        raise ValueError("Dimension mismatch between visible data, hidden activations, and kernel size.")

    correlation = torch.zeros(out_channels, in_channels, kernel_size, kernel_size, device=visible_data.device)

    # MATLAB's `conv2(V, rot180(H), 'valid')` is equivalent to cross-correlation.
    # The gradient w.r.t. W_uv is Sum_{i,j} H_{i,j} * V_{i+u, j+v}
    for f_idx in range(out_channels):
        for c_idx in range(in_channels):
            for u in range(kernel_size):
                for v in range(kernel_size):
                    vis_patch = visible_data[:, c_idx, u:u+conv_height, v:v+conv_width]
                    hid_map = hidden_activations[:, f_idx, :, :]
                    corr_value = torch.sum(vis_patch * hid_map)
                    correlation[f_idx, c_idx, u, v] = corr_value

    # Normalize by batch size to make the gradient independent of it
    return correlation / batch_size


def compute_bias_gradients(hidden_activations_pos, hidden_activations_neg,
                           visible_data_pos, visible_data_neg):
    """
    Computes bias gradients for both hidden and visible units. The gradients
    are normalized by both batch size and spatial dimensions.
    """
    batch_size = hidden_activations_pos.shape[0]
    hid_size = hidden_activations_pos.shape[2] * hidden_activations_pos.shape[3]
    vis_size = visible_data_pos.shape[2] * visible_data_pos.shape[3]

    # Hidden bias gradient: difference in mean activation
    pos_hid_act = torch.sum(hidden_activations_pos, dim=(0, 2, 3))
    neg_hid_act = torch.sum(hidden_activations_neg, dim=(0, 2, 3))
    # Normalize only by batch size to retain adequate gradient magnitude
    hidden_bias_grad = (pos_hid_act - neg_hid_act) / batch_size

    # Visible bias gradient: difference in mean activation
    pos_vis_act = torch.sum(visible_data_pos, dim=(0, 2, 3))
    neg_vis_act = torch.sum(visible_data_neg, dim=(0, 2, 3))
    visible_bias_grad = (pos_vis_act - neg_vis_act) / batch_size

    return hidden_bias_grad, visible_bias_grad


def update_parameters_with_momentum(weights, weight_grad, weight_momentum,
                                    bias, bias_grad, bias_momentum,
                                    learning_rate, momentum_coeff, weight_decay=0.0):
    """
    Updates parameters using momentum and optional L2 weight decay.
    Note: Updates are performed in-place.
    """
    # --- Weight updates ---
    # 1. Calculate final gradient including L2 penalty.
    #    The correct L2 penalty SUBTRACTS from the gradient.
    #    (MATLAB: dW_total = (pos-neg)/hidsize - l2reg*W)
    weight_grad_final = weight_grad - weight_decay * weights

    # 2. Update momentum term: mom = mom_coeff*mom + lr*grad_final
    weight_momentum.mul_(momentum_coeff).add_(weight_grad_final, alpha=learning_rate)
    
    # 3. Apply update: W = W + mom
    weights.add_(weight_momentum)

    # --- Bias updates (no weight decay on biases) ---
    # 1. Update momentum term: mom = mom_coeff*mom + lr*grad
    bias_momentum.mul_(momentum_coeff).add_(bias_grad, alpha=learning_rate)
    
    # 2. Apply update: bias = bias + mom
    bias.add_(bias_momentum)


def compute_reconstruction_error(original, reconstructed):
    """
    Computes the Mean Squared Error (MSE) between original and reconstructed data.
    """
    return F.mse_loss(reconstructed, original).item()


def sampling_bernoulli(probs):
    """Bernoulli sampling from probabilities."""
    return torch.bernoulli(probs)


def sampling_gaussian(probs):
    """Gaussian sampling (for compatibility)."""
    return probs + torch.randn_like(probs) * 0.1
