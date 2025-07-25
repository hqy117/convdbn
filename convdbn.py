import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pooling import MultinomialMaxPool2d, SparseUnpool2d
from utils import compute_correlation, compute_bias_gradients, update_parameters_with_momentum

class ConvRBMLayer(nn.Module):
    """Single layer Convolutional RBM with MultinomialMaxPool2d"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 k=2, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=False,
                 spacing=3, pbias=0.002, plambda=5.0, eta_sparsity=0.0,
                 sigma=0.2, sigma_stop=0.1, sigma_schedule=True):
        super(ConvRBMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        # Sparsity and sigma parameters
        self.spacing = spacing
        self.pbias = pbias
        self.plambda = plambda
        self.eta_sparsity = eta_sparsity
        self.sigma = sigma
        self.sigma_stop = sigma_stop
        self.sigma_schedule = sigma_schedule
        self.running_avg_prob = None

        # Convolutional weights and biases
        self.conv_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(in_channels))
        self.hidden_bias = nn.Parameter(torch.zeros(out_channels))

        # Momentum terms
        self.conv_weights_momentum = torch.zeros_like(self.conv_weights)
        self.visible_bias_momentum = torch.zeros_like(self.visible_bias)
        self.hidden_bias_momentum = torch.zeros_like(self.hidden_bias)

        # Pooling layers
        self.multinomial_pool = MultinomialMaxPool2d(spacing=spacing)
        self.sparse_unpool = SparseUnpool2d(spacing=spacing)
        self.stored_winner_info = None

        if self.use_cuda:
            self._move_to_cuda()

    def _move_to_cuda(self):
        """Move all parameters to CUDA with optimizations"""
        # Use PyTorch's built-in method to move the module to GPU
        self.cuda()

        # Move momentum tensors manually (they're not nn.Parameters)
        device = f'cuda:{torch.cuda.current_device()}'
        self.conv_weights_momentum = self.conv_weights_momentum.to(device, non_blocking=True)
        self.visible_bias_momentum = self.visible_bias_momentum.to(device, non_blocking=True)
        self.hidden_bias_momentum = self.hidden_bias_momentum.to(device, non_blocking=True)

        # Enable GPU optimizations
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True

    def sample_hidden(self, visible):
        """Sample hidden units from visible units with MultinomialMaxPool2d"""
        # Convolution
        hidden_pre = F.conv2d(visible, self.conv_weights, bias=self.hidden_bias,
                             stride=self.stride, padding=self.padding)

        # Apply sigma scaling
        hidden_pre_scaled = hidden_pre / (self.sigma ** 2)
        hidden_prob = torch.sigmoid(hidden_pre_scaled)

        # MultinomialMaxPool2d
        sparse_detection, pooled_map, winner_info = self.multinomial_pool(hidden_prob)

        # Store winner info for reconstruction
        self.stored_winner_info = winner_info

        # Return pooled map
        return pooled_map

    def sample_visible(self, hidden, output_size=None):
        """Sample visible units from hidden units using sparse unpooling"""
        # Sparse unpooling to restore detection map
        sparse_detection = self.sparse_unpool(hidden, self.stored_winner_info)

        # Transpose convolution
        visible_recon = F.conv_transpose2d(sparse_detection, weight=self.conv_weights,
                                         bias=self.visible_bias, stride=self.stride, padding=self.padding)

        # Apply sigma scaling
        visible_recon = visible_recon / (self.sigma ** 2)
        visible_recon = torch.sigmoid(visible_recon)

        return visible_recon

    def sample_visible_original(self, hidden):
        """Reconstruct visible units from hidden units"""
        visible_pre = F.conv_transpose2d(hidden, self.conv_weights, bias=self.visible_bias,
                                        stride=self.stride, padding=self.padding)
        visible_prob = torch.sigmoid(visible_pre)

        # Adjust to target size if specified
        if output_size is not None:
            target_h, target_w = output_size
            current_h, current_w = visible_prob.shape[-2:]

            if current_h != target_h or current_w != target_w:
                visible_prob = F.interpolate(visible_prob, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return visible_prob

    def contrastive_divergence(self, visible_data):
        """
        Proper Contrastive Divergence algorithm with MultinomialMaxPool2d.

        This implements the correct CD algorithm following the MATLAB approach:
        1. Use pre-pooling activations for correlation computation
        2. Include sparsity regularization and sigma scheduling
        3. Use proper MultinomialMaxPool2d throughout
        """
        batch_size = visible_data.size(0)

        # === POSITIVE PHASE ===
        # Forward pass and store intermediate activations for correlation
        conv_pos = F.conv2d(visible_data, weight=self.conv_weights, bias=self.hidden_bias,
                           stride=self.stride, padding=self.padding)

        # Apply sigma scaling (following MATLAB implementation)
        conv_pos_scaled = conv_pos / (self.sigma ** 2)
        conv_pos_sigmoid = torch.sigmoid(conv_pos_scaled)  # PRE-POOLING activations

        # Multinomial pooling
        sparse_pos, pooled_pos, winner_info_pos = self.multinomial_pool(conv_pos_sigmoid)

        # Sample hidden states from pooled probabilities
        pos_hidden_states = torch.bernoulli(pooled_pos)

        # === NEGATIVE PHASE (CD-k) ===
        neg_hidden_states = pos_hidden_states
        for k_step in range(self.k):
            # Reconstruct visible
            self.stored_winner_info = winner_info_pos  # Use positive phase pattern
            neg_visible_probs = self.sample_visible(neg_hidden_states)

            # Re-encode hidden
            conv_neg = F.conv2d(neg_visible_probs, weight=self.conv_weights, bias=self.hidden_bias,
                               stride=self.stride, padding=self.padding)
            conv_neg_scaled = conv_neg / (self.sigma ** 2)
            conv_neg_sigmoid = torch.sigmoid(conv_neg_scaled)
            sparse_neg, pooled_neg, winner_info_neg = self.multinomial_pool(conv_neg_sigmoid)
            neg_hidden_states = torch.bernoulli(pooled_neg)

        # Final negative phase activations
        neg_visible_probs = self.sample_visible(neg_hidden_states)
        conv_neg = F.conv2d(neg_visible_probs, weight=self.conv_weights, bias=self.hidden_bias,
                           stride=self.stride, padding=self.padding)
        conv_neg_scaled = conv_neg / (self.sigma ** 2)
        conv_neg_sigmoid = torch.sigmoid(conv_neg_scaled)

        # === GRADIENT COMPUTATION ===
        with torch.no_grad():
            # Compute correlations using PRE-POOLING activations (this is crucial!)
            pos_correlation = compute_correlation(visible_data, conv_pos_sigmoid, self.kernel_size)
            neg_correlation = compute_correlation(neg_visible_probs, conv_neg_sigmoid, self.kernel_size)

            # CD weight gradient = positive correlation - negative correlation
            weight_grad = pos_correlation - neg_correlation

            # Compute bias gradients
            hidden_bias_grad, visible_bias_grad = compute_bias_gradients(
                conv_pos_sigmoid, conv_neg_sigmoid, visible_data, neg_visible_probs
            )

            # === SPARSITY REGULARIZATION (following MATLAB fobj_sparsity.m) ===
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

            # === PARAMETER UPDATES ===
            # Update weights with proper CD gradient
            update_parameters_with_momentum(
                self.conv_weights, weight_grad, self.conv_weights_momentum,
                self.hidden_bias, hidden_bias_grad, self.hidden_bias_momentum,
                self.learning_rate, self.momentum_coefficient, self.weight_decay
            )

            # Update visible bias
            self.visible_bias_momentum.mul_(self.momentum_coefficient)
            self.visible_bias_momentum.add_(visible_bias_grad)
            self.visible_bias.add_(self.visible_bias_momentum, alpha=self.learning_rate)

        # Calculate reconstruction error
        from utils import compute_reconstruction_error
        error = compute_reconstruction_error(visible_data, neg_visible_probs)

        # === SIGMA SCHEDULING (following MATLAB crbm_train.m) ===
        if self.sigma_schedule and self.sigma > self.sigma_stop:
            # Decay sigma by 0.99 each step (following MATLAB: params.sigma = params.sigma*0.99)
            self.sigma = self.sigma * 0.99

        return error

    def get_sparsity_stats(self):
        """Get current sparsity statistics"""
        return {
            'pbias': self.pbias,
            'plambda': self.plambda,
            'eta_sparsity': self.eta_sparsity,
            'running_avg_prob': self.running_avg_prob.cpu().numpy() if self.running_avg_prob is not None else None,
            'target_sparsity': self.pbias,
            'current_sigma': self.sigma,
            'sigma_stop': self.sigma_stop
        }

    def get_sparse_features(self, input_data):
        """Extract sparse detection features (full resolution)"""
        with torch.no_grad():
            conv_out = F.conv2d(input_data, weight=self.conv_weights, bias=self.hidden_bias,
                               stride=self.stride, padding=self.padding)
            conv_out = conv_out / (self.sigma ** 2)  # Apply sigma scaling
            conv_out = torch.sigmoid(conv_out)
            sparse_detection, _, _ = self.multinomial_pool(conv_out)
            return sparse_detection.view(sparse_detection.size(0), -1)

    def get_pooled_features(self, input_data):
        """Extract pooled features (reduced resolution)"""
        with torch.no_grad():
            return self.sample_hidden(input_data).view(input_data.size(0), -1)


class ConvDBN(nn.Module):
    """Multi-layer Convolutional Deep Belief Network, supports 2-5 layers, target feature dimensions close to 3072"""

    def __init__(self, k=2, learning_rate=1e-3, momentum_coefficient=0.5,
                 weight_decay=1e-4, use_cuda=False, input_channels=1, input_dim=28,
                 dataset='mnist', num_layers=3, spacing=3, pbias=0.002, plambda=5.0,
                 eta_sparsity=0.0, sigma=0.2, sigma_stop=0.1, sigma_schedule=True):
        super(ConvDBN, self).__init__()

        self.use_cuda = use_cuda
        self.k = k
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.dataset = dataset
        self.num_layers = num_layers

        # Sparsity and sigma parameters (following MATLAB demo_cdbn.m)
        self.spacing = spacing
        self.pbias = pbias
        self.plambda = plambda
        self.eta_sparsity = eta_sparsity
        self.sigma = sigma
        self.sigma_stop = sigma_stop
        self.sigma_schedule = sigma_schedule

        # Target feature dimensions: MNIST ~768, CIFAR10 ~3072
        target_features = 768 if dataset == 'mnist' else 3072

        # Configure network architecture based on dataset and number of layers
        self.layers = nn.ModuleList()

        if dataset == 'mnist':
            self._build_mnist_architecture(k, learning_rate, momentum_coefficient, weight_decay, use_cuda, target_features)
        else:  # CIFAR10
            self._build_cifar10_architecture(k, learning_rate, momentum_coefficient, weight_decay, use_cuda, target_features)

    def _create_layer(self, in_channels, out_channels, kernel_size, k, learning_rate, momentum_coefficient, weight_decay, use_cuda):
        """Helper function to create ConvRBMLayer with all parameters"""
        return ConvRBMLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=1, padding=0, k=k, learning_rate=learning_rate,
            momentum_coefficient=momentum_coefficient, weight_decay=weight_decay, use_cuda=use_cuda,
            spacing=self.spacing, pbias=self.pbias, plambda=self.plambda,
            eta_sparsity=self.eta_sparsity, sigma=self.sigma, sigma_stop=self.sigma_stop,
            sigma_schedule=self.sigma_schedule
        )

    def _build_mnist_architecture(self, k, learning_rate, momentum_coefficient, weight_decay, use_cuda, target_features):
        """Build MNIST architecture, target feature dimensions close to 768 (MNIST baseline: 28×28=784), using stride=1"""
        if self.num_layers == 2:
            # 2-layer: [1, 28, 28] -> [48, 24, 24] -> MaxPool(2x2) -> [48, 12, 12] -> [12, 8, 8] = 768 features ≈ 768*1.0
            self.layers.append(self._create_layer(1, 48, 5, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))
            self.layers.append(self._create_layer(48, 12, 5, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))

        elif self.num_layers == 3:
            # 3-layer: [1, 28, 28] -> [16, 24, 24] -> [8, 20, 20] -> [3, 16, 16] = 768 features ≈ 768*1.0
            self.layers.append(self._create_layer(1, 16, 5, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))
            self.layers.append(self._create_layer(16, 8, 5, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))
            self.layers.append(self._create_layer(8, 3, 5, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))

        elif self.num_layers == 4:
            # 4-layer: [1, 28, 28] -> [32, 24, 24] -> [20, 20, 20] -> [12, 16, 16] -> [6, 12, 12] = 864 features ≈ 768*1.125
            self.layers.append(ConvRBMLayer(
                in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda
            ))
            self.layers.append(ConvRBMLayer(
                in_channels=32, out_channels=20, kernel_size=5, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda
            ))
            self.layers.append(ConvRBMLayer(
                in_channels=20, out_channels=12, kernel_size=5, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda
            ))
            self.layers.append(ConvRBMLayer(
                in_channels=12, out_channels=6, kernel_size=5, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda
            ))

        elif self.num_layers == 5:
            # 5-layer: [1, 28, 28] -> [40, 24, 24] -> [28, 20, 20] -> [20, 16, 16] -> [14, 12, 12] -> [9, 8, 8] = 576 features ≈ 768*0.75
            self.layers.append(ConvRBMLayer(
                in_channels=1, out_channels=40, kernel_size=5, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda
            ))
            for i, out_ch in enumerate([28, 20, 14, 9]):
                in_ch = [40, 28, 20, 14][i]
                self.layers.append(ConvRBMLayer(
                    in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=1, padding=0,
                    k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                    weight_decay=weight_decay, use_cuda=use_cuda
                ))

    def _build_cifar10_architecture(self, k, learning_rate, momentum_coefficient, weight_decay, use_cuda, target_features):
        """Build CIFAR10 architecture, target feature dimensions close to 3072, using stride=1"""
        if self.num_layers == 2:
            # 2-layer: [3, 32, 32] -> conv5x5 -> [64, 28, 28] -> pool(2x2) -> [64, 14, 14] -> conv3x3 -> [32, 12, 12] = 4608 features
            self.layers.append(self._create_layer(3, 64, 5, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))
            self.layers.append(self._create_layer(64, 32, 3, k, learning_rate, momentum_coefficient, weight_decay, use_cuda))

        elif self.num_layers == 3:
            # 3-layer with spacing=2 to avoid dimension collapse
            # [3, 32, 32] -> conv5x5 -> [48, 28, 28] -> pool(2x2) -> [48, 14, 14] -> conv3x3 -> [24, 12, 12] -> pool(2x2) -> [24, 6, 6] -> conv3x3 -> [12, 4, 4] = 192 features
            # Use smaller spacing for CIFAR-10 to prevent dimension collapse
            layer1 = ConvRBMLayer(
                in_channels=3, out_channels=48, kernel_size=5, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda, spacing=2,  # Use spacing=2 for CIFAR-10
                pbias=self.pbias, plambda=self.plambda, eta_sparsity=self.eta_sparsity,
                sigma=self.sigma, sigma_stop=self.sigma_stop, sigma_schedule=self.sigma_schedule
            )
            layer2 = ConvRBMLayer(
                in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda, spacing=2,  # Use spacing=2 for CIFAR-10
                pbias=self.pbias, plambda=self.plambda, eta_sparsity=self.eta_sparsity,
                sigma=self.sigma, sigma_stop=self.sigma_stop, sigma_schedule=self.sigma_schedule
            )
            layer3 = ConvRBMLayer(
                in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=0,
                k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                weight_decay=weight_decay, use_cuda=use_cuda, spacing=2,  # Use spacing=2 for CIFAR-10
                pbias=self.pbias, plambda=self.plambda, eta_sparsity=self.eta_sparsity,
                sigma=self.sigma, sigma_stop=self.sigma_stop, sigma_schedule=self.sigma_schedule
            )
            self.layers.extend([layer1, layer2, layer3])

        elif self.num_layers == 4:
            # 4-layer with spacing=2 for CIFAR-10
            configs = [(3, 32, 5), (32, 24, 3), (24, 16, 3), (16, 8, 3)]
            for in_ch, out_ch, kernel in configs:
                layer = ConvRBMLayer(
                    in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=1, padding=0,
                    k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                    weight_decay=weight_decay, use_cuda=use_cuda, spacing=2,
                    pbias=self.pbias, plambda=self.plambda, eta_sparsity=self.eta_sparsity,
                    sigma=self.sigma, sigma_stop=self.sigma_stop, sigma_schedule=self.sigma_schedule
                )
                self.layers.append(layer)

        elif self.num_layers == 5:
            # 5-layer with spacing=2 for CIFAR-10
            configs = [(3, 32, 5), (32, 24, 3), (24, 16, 3), (16, 12, 3), (12, 8, 3)]
            for in_ch, out_ch, kernel in configs:
                layer = ConvRBMLayer(
                    in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=1, padding=0,
                    k=k, learning_rate=learning_rate, momentum_coefficient=momentum_coefficient,
                    weight_decay=weight_decay, use_cuda=use_cuda, spacing=2,
                    pbias=self.pbias, plambda=self.plambda, eta_sparsity=self.eta_sparsity,
                    sigma=self.sigma, sigma_stop=self.sigma_stop, sigma_schedule=self.sigma_schedule
                )
                self.layers.append(layer)
        
    def forward_layer(self, x, layer_idx):
        """Forward propagation for specified layer"""
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].sample_hidden(x)
        else:
            raise ValueError(f"Layer {layer_idx} does not exist. Model has {len(self.layers)} layers.")

    def forward_all_layers(self, x):
        """Forward propagation through all layers"""
        activations = [x]  # Store activations for each layer
        sparse_patterns = []  # Store sparse detection patterns
        current = x

        for i, layer in enumerate(self.layers):
            current = layer.sample_hidden(current)

            # For 2-layer configuration first layer, apply multinomial pooling
            if self.num_layers == 2 and i == 0 and len(self.multinomial_pools) > 0:
                sparse_detection, current, winner_info = self.multinomial_pools[0](current)
                sparse_patterns.append(winner_info)

            activations.append(current)

        return activations, sparse_patterns  # Return activations and sparse patterns for all layers
    
    def get_features(self, x):
        """Extract final features"""
        with torch.no_grad():
            # Through all layers
            activations, sparse_patterns = self.forward_all_layers(x)
            final_features = activations[-1]  # Output of the last layer

            # Return flattened features
            return final_features.view(final_features.size(0), -1)
    
    def reconstruct(self, x):
        """Complete reconstruction"""
        with torch.no_grad():
            # Forward propagation to get activations of all layers
            activations, sparse_patterns = self.forward_all_layers(x)

            # Backward reconstruction: start from the last layer
            current = activations[-1]  # Output of the last layer

            # Layer-by-layer backward reconstruction
            for i in range(len(self.layers) - 1, 0, -1):
                target_shape = activations[i].shape[-2:]  # Spatial dimensions of target layer
                current = self.layers[i].sample_visible(current, target_shape)

                # For 2-layer configuration, need reverse pooling after first layer
                if self.num_layers == 2 and i == 1 and len(sparse_patterns) > 0:
                    current = self.sparse_unpools[0](current, sparse_patterns[0])

            # Final reconstruction to input
            x_recon = self.layers[0].sample_visible(current, x.shape[-2:])

            return x_recon
    
    def train_layer(self, layer_idx, data_loader, epochs=5):
        """Train specified layer"""
        if layer_idx < 1 or layer_idx > self.num_layers:
            raise ValueError(f"Layer index must be between 1 and {self.num_layers}")

        return self._train_layer_generic(layer_idx - 1, data_loader, epochs)  # Convert to 0-based index
    
    def _train_layer_generic(self, layer_idx, data_loader, epochs):
        """Generic layer training method"""
        total_error = 0
        batch_count = 0

        for epoch in range(epochs):
            epoch_error = 0
            for batch, _ in data_loader:
                if batch.size(0) != data_loader.batch_size:
                    continue

                batch = batch.view(batch.size(0), self.input_channels, self.input_dim, self.input_dim)
                if self.use_cuda:
                    batch = batch.cuda()

                # Get input for current layer
                if layer_idx == 0:
                    # First layer uses original input directly
                    layer_input = batch
                else:
                    # Other layers need to go through previous layers
                    with torch.no_grad():
                        current = batch
                        for i in range(layer_idx):
                            current = self.layers[i].sample_hidden(current)
                            # For 2-layer configuration first layer, apply multinomial pooling
                            if self.num_layers == 2 and i == 0 and len(self.multinomial_pools) > 0:
                                _, current, _ = self.multinomial_pools[0](current)
                        layer_input = current

                # Train current layer
                error = self.layers[layer_idx].contrastive_divergence(layer_input)
                epoch_error += error.item()
                batch_count += 1

            avg_error = epoch_error / max(1, batch_count)
            print(f"Layer {layer_idx + 1}, Epoch {epoch+1}/{epochs}: Error = {avg_error:.4f}")
            total_error += avg_error

        return total_error / epochs