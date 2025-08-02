import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np

# Import our new pooling and utility modules
from pooling import MultinomialMaxPool2d, SparseUnpool2d
from utils import compute_correlation, compute_bias_gradients, update_parameters_with_momentum, compute_reconstruction_error

class ConvRBM():
    """Convolutional Restricted Boltzmann Machine"""

    def __init__(self, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
                 use_cuda=False, input_channels=1,
                 # Architecture
                 num_filters=None, conv_kernel=None, pool_stride=None,
                 # Sparsity
                 pbias=0.0, plambda=0.0, eta_sparsity=0.0,
                 # Sigma (noise)
                 sigma=1.0, sigma_stop=None, sigma_schedule=True):

        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.input_channels = input_channels

        # Sparsity regularization parameters
        self.pbias = pbias  # Target sparsity level
        self.plambda = plambda  # Sparsity penalty weight
        self.eta_sparsity = eta_sparsity  # Sparsity learning rate

        # Noise level
        self.sigma = sigma
        self.sigma_stop = sigma_stop if sigma_stop is not None else sigma / 2
        self.sigma_schedule = sigma_schedule

        # Architecture configuration
        self.num_filters = num_filters
        self.conv_kernel = conv_kernel
        self.pool_stride = pool_stride

        # Use dataset-specific defaults if not provided
        if self.num_filters is None:
            if input_channels == 1: self.num_filters = 12  # MNIST
            elif input_channels == 3: self.num_filters = 16 # CIFAR-10
            else: self.num_filters = 16 # Generic
        
        if self.conv_kernel is None:
            if input_channels == 1: self.conv_kernel = 4
            elif input_channels == 3: self.conv_kernel = 4
            else: self.conv_kernel = 5

        if self.pool_stride is None:
            if input_channels == 1: self.pool_stride = 3
            elif input_channels == 3: self.pool_stride = 2
            else: self.pool_stride = 2

        # Sparsity tracking
        self.running_avg_prob = None 

        # Network parameters
        self.conv1_weights = nn.Parameter(torch.randn(self.num_filters, input_channels, self.conv_kernel, self.conv_kernel) * 0.1)
        self.conv1_visible_bias = nn.Parameter(torch.zeros(input_channels))
        self.conv1_hidden_bias = nn.Parameter(torch.zeros(self.num_filters))

        # Probabilistic pooling layers
        self.multinomial_pool = MultinomialMaxPool2d(spacing=self.pool_stride)
        self.sparse_unpool = SparseUnpool2d(spacing=self.pool_stride)

        # Momentum terms
        self.conv1_weights_momentum = torch.zeros_like(self.conv1_weights)
        self.conv1_visible_bias_momentum = torch.zeros_like(self.conv1_visible_bias)
        self.conv1_hidden_bias_momentum = torch.zeros_like(self.conv1_hidden_bias)

        if self.use_cuda:
            self._move_to_cuda()

    def _move_to_cuda(self):
        """Move all parameters to CUDA with optimizations"""
        device = f'cuda:{torch.cuda.current_device()}'
        self.conv1_weights = self.conv1_weights.to(device, non_blocking=True)
        self.conv1_visible_bias = self.conv1_visible_bias.to(device, non_blocking=True)
        self.conv1_hidden_bias = self.conv1_hidden_bias.to(device, non_blocking=True)
        self.conv1_weights_momentum = self.conv1_weights_momentum.to(device, non_blocking=True)
        self.conv1_visible_bias_momentum = self.conv1_visible_bias_momentum.to(device, non_blocking=True)
        self.conv1_hidden_bias_momentum = self.conv1_hidden_bias_momentum.to(device, non_blocking=True)

        # Enable GPU optimizations
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True

    def sample_hidden(self, visible_probabilities):
        """
        Samples hidden units from visible units.
        Returns the sparse, pooled, and winner info tensors.
        """
        # Convolution + bias
        conv_out = F.conv2d(visible_probabilities, weight=self.conv1_weights, bias=self.conv1_hidden_bias)

        # Apply sigma scaling, which is part of the energy function
        conv_out_scaled = conv_out / (self.sigma ** 2)
        
        # This function returns all necessary components
        sparse_detection, pooled_map, winner_info = self.multinomial_pool(conv_out_scaled)

        return sparse_detection, pooled_map, winner_info

    def sample_visible(self, hidden_probabilities, winner_info):
        """
        Samples visible units from hidden units (reconstruction).
        Requires winner_info for the unpooling operation.
        """
        # Unpool to restore the sparse, full-resolution hidden activation map
        sparse_detection = self.sparse_unpool(hidden_probabilities, winner_info)

        # Transpose convolution to reconstruct
        visible_recon = F.conv_transpose2d(sparse_detection, weight=self.conv1_weights,
                                         bias=self.conv1_visible_bias)

        # Apply sigma and sigmoid
        visible_recon = torch.sigmoid(visible_recon / (self.sigma ** 2))

        return visible_recon

    def contrastive_divergence(self, input_data):
        """
        Contrastive Divergence algorithm using explicit sampling functions.
        """
        # POSITIVE PHASE
        # =================
        # Sample hidden layer from the input data
        sparse_pos, pooled_pos, winner_info_pos = self.sample_hidden(input_data)
        
        # Get binary hidden states
        pos_hidden_states = torch.bernoulli(pooled_pos)

        # NEGATIVE PHASE (CD-k)
        # =======================
        # k steps of Gibbs sampling
        neg_hidden_states = pos_hidden_states
        current_winner_info = winner_info_pos

        for _ in range(self.k):
            # Reconstruct visible layer from hidden states
            neg_visible_probs = self.sample_visible(neg_hidden_states, current_winner_info)

            # Sample new hidden states from the reconstructed visible layer
            sparse_neg, pooled_neg, winner_info_neg = self.sample_hidden(neg_visible_probs)
            
            # Update for the next Gibbs step
            neg_hidden_states = torch.bernoulli(pooled_neg)
            current_winner_info = winner_info_neg

        # Final state of the negative chain
        final_sparse_neg = sparse_neg
        final_neg_visible_probs = neg_visible_probs

        # GRADIENT CALCULATION & PARAMETER UPDATES
        # ==========================================
        with torch.no_grad():
            pos_correlation = compute_correlation(input_data, sparse_pos, self.conv_kernel)
            neg_correlation = compute_correlation(final_neg_visible_probs, final_sparse_neg, self.conv_kernel)

            # Removed extra normalization by hidden map size; compute_correlation already averages by batch
            weight_grad = pos_correlation - neg_correlation

            hidden_bias_grad, visible_bias_grad = compute_bias_gradients(
                sparse_pos, final_sparse_neg, input_data, final_neg_visible_probs
            )
            
            if self.plambda > 0:
                batch_size = sparse_pos.size(0)
                hidden_size = sparse_pos.shape[2] * sparse_pos.shape[3]
                pos_hidden_act = torch.sum(sparse_pos, dim=(0, 2, 3)) / (batch_size * hidden_size)

                if self.running_avg_prob is None:
                    self.running_avg_prob = pos_hidden_act.clone()
                else:
                    self.running_avg_prob.mul_(self.eta_sparsity).add_(pos_hidden_act, alpha=1 - self.eta_sparsity)

                sparsity_grad = self.pbias - self.running_avg_prob
                hidden_bias_grad.add_(sparsity_grad, alpha=self.plambda)

            update_parameters_with_momentum(
                self.conv1_weights, weight_grad, self.conv1_weights_momentum,
                self.conv1_hidden_bias, hidden_bias_grad, self.conv1_hidden_bias_momentum,
                self.learning_rate, self.momentum_coefficient, self.weight_decay
            )
            self.conv1_visible_bias_momentum.mul_(self.momentum_coefficient).add_(visible_bias_grad)
            self.conv1_visible_bias.add_(self.conv1_visible_bias_momentum, alpha=self.learning_rate)

        error = compute_reconstruction_error(input_data, final_neg_visible_probs)

        if self.sigma_schedule and self.sigma > self.sigma_stop:
            self.sigma *= 0.99

        return error

    def get_pre_pooling_activations(self, visible_data):
        with torch.no_grad():
            conv_out = F.conv2d(visible_data, weight=self.conv1_weights, bias=self.conv1_hidden_bias)
            conv_out_scaled = conv_out / (self.sigma ** 2)
            return torch.sigmoid(conv_out_scaled)

    def reconstruct(self, input_data):
        """Reconstruct input data using the full forward-backward pass"""
        with torch.no_grad():
            _, pooled_hidden, winner_info = self.sample_hidden(input_data)
            reconstructed = self.sample_visible(pooled_hidden, winner_info)
            return reconstructed

    def get_hidden_features(self, input_data):
        """Extract hidden features for classification (pooled features)"""
        with torch.no_grad():
            _, pooled_features, _ = self.sample_hidden(input_data)
            return pooled_features.view(pooled_features.size(0), -1)

    def get_sparse_features(self, input_data):
        """Extract sparse detection features (full resolution)"""
        with torch.no_grad():
            sparse_detection, _, _ = self.sample_hidden(input_data)
            return sparse_detection.view(sparse_detection.size(0), -1)

    def get_sparsity_stats(self):
        """Get current sparsity statistics for monitoring"""
        stats = {
            'pbias': self.pbias,
            'plambda': self.plambda,
            'eta_sparsity': self.eta_sparsity,
            'running_avg_prob': self.running_avg_prob.cpu().numpy() if self.running_avg_prob is not None else None,
            'target_sparsity': self.pbias,
            'current_sigma': self.sigma,
            'sigma_stop': self.sigma_stop
        }
        return stats

    def get_current_sparsity(self, input_data):
        """Compute current sparsity level for a batch of data"""
        with torch.no_grad():
            sparse_detection, _, _ = self.sample_hidden(input_data)
            # Sparsity is the mean activation of the sparse feature map
            sparsity = torch.mean(sparse_detection).item()
            return sparsity
