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
    """Convolutional Restricted Boltzmann Machine with size matching fixes and backward compatibility"""

    def __init__(self, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
                 num_visible=0, num_hidden=0, batch_size=64, use_cuda=False, input_channels=1, target_features=None,
                 pbias=0.0, plambda=0.0, eta_sparsity=0.0, sigma=1.0, sigma_stop=None, sigma_schedule=True):
        # Backward compatible parameters
        self.batch_size = batch_size
        self.k = k  # CD steps
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.input_channels = input_channels

        # Sparsity regularization parameters
        self.pbias = pbias  # Target sparsity level
        self.plambda = plambda  # Sparsity penalty weight
        self.eta_sparsity = eta_sparsity  # Sparsity learning rate

        # Sigma parameters
        self.sigma = sigma  # Current sigma value
        self.sigma_stop = sigma_stop if sigma_stop is not None else sigma / 2  # Stop value for sigma decay
        self.sigma_schedule = sigma_schedule  # Whether to use sigma scheduling

        # Sparsity tracking
        self.running_avg_prob = None  # Running average of hidden unit activations

        # Adjust architecture based on target feature dimensions
        if target_features == 3072 and input_channels == 3:
            # For CIFAR10: design architecture to get close to 3072 features
            # Conv: 32 → 29 (4×4 conv), Pool: 29 → 14 (3×3, stride=2)
            # 16 × 14 × 14 = 3136 ≈ 3072
            self.num_filters = 16
            self.conv_kernel = 4
            self.pool_kernel = 3
            self.pool_stride = 2
        elif target_features == 768 and input_channels == 1:
            # For MNIST: design architecture to get close to 768 features
            # Conv: 28 → 25 (4×4 conv), Pool: 25 → 12 (2×2, stride=2)
            # 16 × 12 × 12 = 2304 (too large)
            # Better: Conv: 28 → 25 (4×4 conv), Pool: 25 → 8 (3×3, stride=3)
            # 12 × 8 × 8 = 768 ≈ 768 (perfect!)
            self.num_filters = 12
            self.conv_kernel = 4
            self.pool_kernel = 3
            self.pool_stride = 3
        else:
            # Default architecture (fallback)
            self.num_filters = 32
            self.conv_kernel = 4
            self.pool_kernel = 3
            self.pool_stride = 2

        # Network parameters
        self.conv1_weights = nn.Parameter(torch.randn(self.num_filters, input_channels, self.conv_kernel, self.conv_kernel) * 0.1)
        self.conv1_visible_bias = nn.Parameter(torch.zeros(input_channels))
        self.conv1_hidden_bias = nn.Parameter(torch.zeros(self.num_filters))

        # MATLAB-style probabilistic pooling (replacing standard max pooling)
        self.multinomial_pool = MultinomialMaxPool2d(spacing=self.pool_stride)
        self.sparse_unpool = SparseUnpool2d(spacing=self.pool_stride)

        # Storage for forward pass information (needed for unpooling)
        self.stored_winner_info = None

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
        """Sample hidden units from visible units using multinomial pooling with sigma scaling"""
        # Convolution + bias
        conv_out = F.conv2d(visible_probabilities, weight=self.conv1_weights, bias=self.conv1_hidden_bias)

        # Apply sigma scaling
        conv_out = conv_out / (self.sigma ** 2)

        # MATLAB compatibility: Pass scaled values directly to pooling (no sigmoid)
        # MATLAB's sample_multrand works in exp domain, not sigmoid domain
        sparse_detection, pooled_map, winner_info = self.multinomial_pool(conv_out)

        # Step 4: Store information for unpooling
        self.stored_winner_info = winner_info

        # Return pooled map [batch, 16, 14, 14] - this is what gets passed to next layer
        return pooled_map

    def sample_visible(self, hidden_probabilities):
        """Sample visible units from hidden units using sparse unpooling"""
        # Step 1: Sparse unpooling to restore detection map
        sparse_detection = self.sparse_unpool(hidden_probabilities, self.stored_winner_info)

        # Transpose convolution
        visible_recon = F.conv_transpose2d(sparse_detection, weight=self.conv1_weights,
                                         bias=self.conv1_visible_bias, stride=1, padding=0)

        # Apply sigma scaling
        visible_recon = visible_recon / (self.sigma ** 2)
        visible_recon = torch.sigmoid(visible_recon)

        return visible_recon

    def contrastive_divergence(self, input_data, profile=False):
        """
        Proper Contrastive Divergence algorithm with real gradients.

        This implements the correct CD algorithm following the MATLAB approach:
        1. Use pre-pooling activations for correlation computation
        2. Compute real weight gradients (no more placeholders!)
        3. Update all parameters with proper momentum

        Args:
            input_data: input batch
            profile: whether to enable detailed timing profiling

        Returns:
            reconstruction_error: scalar tensor
            profile_info: dict with timing info (if profile=True)
        """
        batch_size = input_data.size(0)
        profile_info = {} if profile else None

        if profile:
            total_start = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()

        # Positive phase
        if profile:
            pos_start = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()

        # Forward pass and store intermediate activations for correlation
        conv_pos = F.conv2d(input_data, weight=self.conv1_weights, bias=self.conv1_hidden_bias)

        # Apply sigma scaling
        conv_pos_scaled = conv_pos / (self.sigma ** 2)
        conv_pos_sigmoid = torch.sigmoid(conv_pos_scaled)

        if profile:
            conv_time = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()
            profile_info['positive_conv'] = conv_time - pos_start

        # Multinomial pooling
        sparse_pos, pooled_pos, winner_info_pos = self.multinomial_pool(conv_pos_sigmoid)

        if profile:
            pool_time = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()
            profile_info['positive_pooling'] = pool_time - conv_time

        # Sample hidden states from pooled probabilities
        pos_hidden_states = torch.bernoulli(pooled_pos)  # [batch, 16, 14, 14]

        # Negative phase (CD-k)
        if profile:
            neg_start = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()
            # Initialize timing accumulators for CD-K breakdown
            profile_info['negative_visible_reconstruction'] = 0.0
            profile_info['negative_conv'] = 0.0
            profile_info['negative_pooling'] = 0.0
            profile_info['negative_sampling'] = 0.0

        neg_hidden_states = pos_hidden_states
        for k_step in range(self.k):
            # Reconstruct visible
            if profile:
                vis_recon_start = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()

            self.stored_winner_info = winner_info_pos  # Use positive phase pattern
            neg_visible_probs = self.sample_visible(neg_hidden_states)

            if profile:
                vis_recon_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['negative_visible_reconstruction'] += vis_recon_time - vis_recon_start

            # Re-encode hidden - convolution
            if profile:
                neg_conv_start = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()

            conv_neg = F.conv2d(neg_visible_probs, weight=self.conv1_weights, bias=self.conv1_hidden_bias)
            conv_neg_scaled = conv_neg / (self.sigma ** 2)
            conv_neg_sigmoid = torch.sigmoid(conv_neg_scaled)

            if profile:
                neg_conv_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['negative_conv'] += neg_conv_time - neg_conv_start

            # Re-encode hidden - pooling
            if profile:
                neg_pool_start = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()

            sparse_neg, pooled_neg, winner_info_neg = self.multinomial_pool(conv_neg_sigmoid)

            if profile:
                neg_pool_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['negative_pooling'] += neg_pool_time - neg_pool_start

            # Sampling
            if profile:
                neg_samp_start = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()

            neg_hidden_states = torch.bernoulli(pooled_neg)

            if profile:
                neg_samp_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['negative_sampling'] += neg_samp_time - neg_samp_start

        if profile:
            cd_time = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()
            profile_info['negative_cd_k_total'] = cd_time - neg_start

        # Final negative phase activations
        neg_visible_probs = self.sample_visible(neg_hidden_states)
        conv_neg = F.conv2d(neg_visible_probs, weight=self.conv1_weights, bias=self.conv1_hidden_bias)
        conv_neg_scaled = conv_neg / (self.sigma ** 2)
        conv_neg_sigmoid = torch.sigmoid(conv_neg_scaled)

        # Gradient computation
        if profile:
            grad_start = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()

        with torch.no_grad():
            # Compute correlations using PRE-POOLING activations (this is crucial!)
            pos_correlation = compute_correlation(input_data, conv_pos_sigmoid, self.conv_kernel)
            neg_correlation = compute_correlation(neg_visible_probs, conv_neg_sigmoid, self.conv_kernel)

            # CD weight gradient = positive correlation - negative correlation
            weight_grad = pos_correlation - neg_correlation

            # Compute bias gradients
            hidden_bias_grad, visible_bias_grad = compute_bias_gradients(
                conv_pos_sigmoid, conv_neg_sigmoid, input_data, neg_visible_probs
            )

            if profile:
                corr_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['gradient_computation'] = corr_time - grad_start

            # Sparsity regularization
            if profile:
                sparsity_start = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()

            if self.plambda > 0:  # Only apply if sparsity penalty is enabled
                # Compute positive hidden activations (sum over spatial dimensions)
                pos_hidden_act = torch.sum(conv_pos_sigmoid, dim=(0, 2, 3))  # [num_filters]
                hidden_size = conv_pos_sigmoid.shape[2] * conv_pos_sigmoid.shape[3]  # spatial size
                pos_hidden_act = pos_hidden_act / (batch_size * hidden_size)  # normalize

                # Update running average of hidden unit probabilities
                if self.running_avg_prob is None:
                    self.running_avg_prob = pos_hidden_act.clone()
                else:
                    # running_avg = eta_sparsity * running_avg + (1 - eta_sparsity) * current
                    self.running_avg_prob = (self.eta_sparsity * self.running_avg_prob +
                                           (1 - self.eta_sparsity) * pos_hidden_act)

                # Compute sparsity gradient: pbias - running_avg_prob
                sparsity_grad = self.pbias - self.running_avg_prob

                # Add sparsity penalty to hidden bias gradient
                hidden_bias_grad = hidden_bias_grad + self.plambda * sparsity_grad

            if profile:
                sparsity_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['sparsity_regularization'] = sparsity_time - sparsity_start

            # Parameter updates
            if profile:
                update_start = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()

            # Update weights with proper CD gradient (NO MORE PLACEHOLDERS!)
            update_parameters_with_momentum(
                self.conv1_weights, weight_grad, self.conv1_weights_momentum,
                self.conv1_hidden_bias, hidden_bias_grad, self.conv1_hidden_bias_momentum,
                self.learning_rate, self.momentum_coefficient, self.weight_decay
            )

            # Update visible bias
            self.conv1_visible_bias_momentum.mul_(self.momentum_coefficient)
            self.conv1_visible_bias_momentum.add_(visible_bias_grad)
            self.conv1_visible_bias.add_(self.conv1_visible_bias_momentum, alpha=self.learning_rate)

            if profile:
                update_time = time.time()
                if self.use_cuda:
                    torch.cuda.synchronize()
                profile_info['parameter_updates'] = update_time - update_start

        # Calculate reconstruction error
        error = compute_reconstruction_error(input_data, neg_visible_probs)

        # Sigma scheduling
        if self.sigma_schedule and self.sigma > self.sigma_stop:
            # Decay sigma by 0.99 each step
            self.sigma = self.sigma * 0.99

        if profile:
            total_time = time.time()
            if self.use_cuda:
                torch.cuda.synchronize()
            profile_info['total_time'] = total_time - total_start
            return error, profile_info
        else:
            return error

    def get_pre_pooling_activations(self, visible_data):
        """
        Get pre-pooling activations for analysis or feature extraction.

        Args:
            visible_data: input data

        Returns:
            conv_activations: [batch, channels, conv_height, conv_width] - pre-pooling activations
        """
        with torch.no_grad():
            conv_out = F.conv2d(visible_data, weight=self.conv1_weights, bias=self.conv1_hidden_bias)
            conv_out = conv_out / (self.sigma ** 2)  # Apply sigma scaling
            conv_out = torch.sigmoid(conv_out)
            return conv_out

    def _random_probabilities(self, num):
        """Generate random probabilities"""
        random_probabilities = torch.rand(num)
        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()
        return random_probabilities

    def reconstruct(self, input_data):
        """Reconstruct input data using the full forward-backward pass"""
        with torch.no_grad():
            hidden = self.sample_hidden(input_data)
            reconstructed = self.sample_visible(hidden)
            return reconstructed

    def get_hidden_features(self, input_data):
        """Extract hidden features for classification (pooled features)"""
        with torch.no_grad():
            hidden_features = self.sample_hidden(input_data)
            return hidden_features.view(hidden_features.size(0), -1)

    def get_sparse_features(self, input_data):
        """Extract sparse detection features (full resolution)"""
        with torch.no_grad():
            conv_out = F.conv2d(input_data, weight=self.conv1_weights, bias=self.conv1_hidden_bias)
            conv_out = conv_out / (self.sigma ** 2)  # Apply sigma scaling
            conv_out = torch.sigmoid(conv_out)
            sparse_detection, _, _ = self.multinomial_pool(conv_out)
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
            conv_out = F.conv2d(input_data, weight=self.conv1_weights, bias=self.conv1_hidden_bias)
            conv_out = conv_out / (self.sigma ** 2)
            conv_out = torch.sigmoid(conv_out)

            # Compute average activation across batch and spatial dimensions
            sparsity = torch.mean(conv_out).item()
            return sparsity


