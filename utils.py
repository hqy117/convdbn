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
    Compute correlation between visible data and hidden activations.

    This is the key function for contrastive divergence - it computes the
    correlation that will be used for weight updates. We use the pre-pooling
    hidden activations for correlation computation.

    Args:
        visible_data: [batch, in_channels, height, width] - visible layer data
        hidden_activations: [batch, out_channels, conv_height, conv_width] - hidden activations (pre-pooling)
        kernel_size: int - size of convolutional kernel

    Returns:
        correlation: [out_channels, in_channels, kernel_size, kernel_size] - correlation matrix
    """
    batch_size, in_channels, vis_height, vis_width = visible_data.shape
    _, out_channels, conv_height, conv_width = hidden_activations.shape

    device = visible_data.device
    correlation = torch.zeros(out_channels, in_channels, kernel_size, kernel_size, device=device)

    # For each output channel (filter)
    for f in range(out_channels):
        # For each input channel
        for c in range(in_channels):
            # For each position in the kernel
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    # Extract visible patches at this kernel position
                    vis_patches = visible_data[:, c, ki:ki+conv_height, kj:kj+conv_width]
                    # Get hidden activations for this filter
                    hid_acts = hidden_activations[:, f, :, :]

                    # Compute correlation: sum over batch and spatial dimensions
                    correlation[f, c, ki, kj] = torch.sum(vis_patches * hid_acts)

    # Normalize by batch size
    correlation = correlation / batch_size

    return correlation


def compute_bias_gradients(hidden_activations_pos, hidden_activations_neg,
                          visible_data_pos, visible_data_neg):
    """
    Compute bias gradients for both hidden and visible units.

    Args:
        hidden_activations_pos: [batch, channels, height, width] - positive phase hidden
        hidden_activations_neg: [batch, channels, height, width] - negative phase hidden
        visible_data_pos: [batch, channels, height, width] - positive phase visible
        visible_data_neg: [batch, channels, height, width] - negative phase visible

    Returns:
        hidden_bias_grad: [channels] - hidden bias gradients
        visible_bias_grad: [channels] - visible bias gradients
    """
    batch_size = hidden_activations_pos.shape[0]

    # Hidden bias gradient: difference in average activation
    hidden_bias_grad = (torch.sum(hidden_activations_pos, dim=(0, 2, 3)) -
                       torch.sum(hidden_activations_neg, dim=(0, 2, 3))) / batch_size

    # Visible bias gradient: difference in average activation
    visible_bias_grad = (torch.sum(visible_data_pos, dim=(0, 2, 3)) -
                        torch.sum(visible_data_neg, dim=(0, 2, 3))) / batch_size

    return hidden_bias_grad, visible_bias_grad


def update_parameters_with_momentum(weights, weight_grad, weight_momentum,
                                   bias, bias_grad, bias_momentum,
                                   learning_rate, momentum_coeff, weight_decay=0.0):
    """
    Update parameters using momentum and optional weight decay.

    Args:
        weights: current weight tensor
        weight_grad: weight gradients
        weight_momentum: weight momentum tensor
        bias: current bias tensor
        bias_grad: bias gradients
        bias_momentum: bias momentum tensor
        learning_rate: learning rate
        momentum_coeff: momentum coefficient
        weight_decay: L2 weight decay coefficient

    Returns:
        None (updates tensors in-place)
    """
    # Weight updates with momentum and weight decay
    weight_momentum.mul_(momentum_coeff)
    if weight_decay > 0:
        weight_grad = weight_grad + weight_decay * weights
    weight_momentum.add_(weight_grad)
    weights.add_(weight_momentum, alpha=learning_rate)

    # Bias updates with momentum (no weight decay on biases)
    bias_momentum.mul_(momentum_coeff)
    bias_momentum.add_(bias_grad)
    bias.add_(bias_momentum, alpha=learning_rate)


def compute_reconstruction_error(original, reconstructed):
    """
    Compute reconstruction error between original and reconstructed data.

    Args:
        original: [batch, channels, height, width] - original data
        reconstructed: [batch, channels, height, width] - reconstructed data

    Returns:
        error: scalar - mean squared error
    """
    return F.mse_loss(reconstructed, original).item()


def check_gradient_magnitudes(weight_grad, bias_grad, threshold=1e-8):
    """
    Check if gradients are reasonable (not too small or too large).

    Args:
        weight_grad: weight gradients
        bias_grad: bias gradients
        threshold: minimum gradient magnitude threshold

    Returns:
        dict with gradient statistics
    """
    weight_grad_norm = torch.norm(weight_grad).item()
    bias_grad_norm = torch.norm(bias_grad).item()

    stats = {
        'weight_grad_norm': weight_grad_norm,
        'bias_grad_norm': bias_grad_norm,
        'weight_grad_max': torch.max(torch.abs(weight_grad)).item(),
        'bias_grad_max': torch.max(torch.abs(bias_grad)).item(),
        'gradients_too_small': weight_grad_norm < threshold and bias_grad_norm < threshold
    }

    return stats


def sampling_bernoulli(probs):
    """Bernoulli sampling from probabilities."""
    return torch.bernoulli(probs)


def sampling_gaussian(probs):
    """Gaussian sampling (for compatibility)."""
    return probs + torch.randn_like(probs) * 0.1
