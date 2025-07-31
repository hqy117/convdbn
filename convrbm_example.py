import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
from convrbm import ConvRBM
from tqdm import tqdm
import argparse
import time

########## COMMAND LINE ARGUMENTS ##########
parser = argparse.ArgumentParser(description='ConvRBM Training')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                    help='Dataset to use: mnist or cifar10 (default: mnist)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs (default: 50)')
parser.add_argument('--subset-size', type=int, default=None,
                    help='Use only a subset of the dataset for quick testing (default: use full dataset)')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Batch size for training (default: 100 for CPU, 200 for GPU)')
parser.add_argument('--eval-train', action='store_true',
                    help='Also evaluate accuracy on training set to check for overfitting')
parser.add_argument('--profile', action='store_true',
                    help='Enable detailed performance profiling of training stages')
args = parser.parse_args()

########## CONFIGURATION ##########
# Dataset selection from command line
DATASET = args.dataset

# Dataset-specific configurations
DATASET_CONFIG = {
    'mnist': {
        'input_dim': 28,
        'input_channels': 1,
        'data_folder': 'data/mnist',
        'dataset_class': torchvision.datasets.MNIST,
        'num_classes': 10
    },
    'cifar10': {
        'input_dim': 32,
        'input_channels': 3,
        'data_folder': 'data/cifar10',
        'dataset_class': torchvision.datasets.CIFAR10,
        'num_classes': 10
    }
}

# Get current dataset configuration
config = DATASET_CONFIG[DATASET]
INPUT_DIM = config['input_dim']
INPUT_CHANNELS = config['input_channels']
DATA_FOLDER = config['data_folder']
DATASET_CLASS = config['dataset_class']
NUM_CLASSES = config['num_classes']

# GPU acceleration - temporarily disabled due to probability range issues
CUDA = False  # torch.cuda.is_available()  # Temporarily disabled
CUDA_DEVICE = 0

print(f"GPU Available: {CUDA}")
if CUDA:
    print(f"GPU Device: {torch.cuda.get_device_name(CUDA_DEVICE)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / 1e9:.1f} GB")

# Training parameters
# Adaptive batch size based on GPU availability
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
elif CUDA:
    BATCH_SIZE = 200  # Larger batch size for GPU
else:
    BATCH_SIZE = 100  # Standard batch size for CPU
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = args.epochs

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

########## LOADING DATASET ##########
print(f'Loading {DATASET.upper()} dataset...')

train_dataset = DATASET_CLASS(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = DATASET_CLASS(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)

# Apply subset if specified
if args.subset_size is not None:
    print(f'Using subset of {args.subset_size} samples for quick testing...')
    train_indices = torch.randperm(len(train_dataset))[:args.subset_size]
    test_indices = torch.randperm(len(test_dataset))[:min(args.subset_size//2, len(test_dataset))]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Use corrected ConvRBM with multi-channel input support
# For CIFAR10, use 3072 feature dimensions; for MNIST, use 768 features
if DATASET == 'cifar10':
    TARGET_FEATURES = 3072  # Close to raw CIFAR10 pixels (32×32×3=3072)
elif DATASET == 'mnist':
    TARGET_FEATURES = 768   # Close to raw MNIST pixels (28×28=784)
else:
    TARGET_FEATURES = None
convrbm = ConvRBM(k=CD_K, use_cuda=CUDA, batch_size=BATCH_SIZE, learning_rate=1e-3,
                  input_channels=INPUT_CHANNELS, target_features=TARGET_FEATURES,
                  pbias=0.002, plambda=5.0, eta_sparsity=0.0, sigma=0.2, sigma_stop=0.1, sigma_schedule=True)

########## NETWORK ARCHITECTURE INFO ##########
print('=' * 60)
print(f'ConvRBM Network Architecture ({DATASET.upper()}):')
print('=' * 60)

# Calculate output dimensions using actual ConvRBM parameters
conv_out_dim = INPUT_DIM - convrbm.conv_kernel + 1
pool_out_dim = (conv_out_dim - convrbm.pool_kernel) // convrbm.pool_stride + 1
hidden_features = convrbm.num_filters * pool_out_dim * pool_out_dim

print(f'Input: [batch_size, {INPUT_CHANNELS}, {INPUT_DIM}, {INPUT_DIM}] ({DATASET.upper()} images)')
print('  ↓')
print(f'Conv2d: {convrbm.num_filters} filters, {convrbm.conv_kernel}×{convrbm.conv_kernel} kernel, stride=1, padding=0')
print(f'  → Output: [batch_size, {convrbm.num_filters}, {conv_out_dim}, {conv_out_dim}]')
print('  ↓')
print('Sigmoid activation')
print('  ↓')
print(f'MultinomialMaxPool2d: {convrbm.pool_kernel}×{convrbm.pool_kernel} regions, multinomial sampling')
print(f'  → Detection map: [batch_size, {convrbm.num_filters}, {conv_out_dim}, {conv_out_dim}] (sparse)')
print(f'  → Pooled features: [batch_size, {convrbm.num_filters}, {pool_out_dim}, {pool_out_dim}] (aggregated)')
print('  ↓')
print(f'Hidden Features: {hidden_features} pooled + {convrbm.num_filters * conv_out_dim * conv_out_dim} sparse dimensions')
print('  |')
print(f'SparseUnpool2d: Pattern restoration using stored sparse detection maps')
print(f'  -> Output: [batch_size, {convrbm.num_filters}, {conv_out_dim}, {conv_out_dim}]')
print('  |')
print(f'ConvTranspose2d: {convrbm.num_filters}->{INPUT_CHANNELS} channels, {convrbm.conv_kernel}x{convrbm.conv_kernel} kernel, stride=1, padding=0')
print(f'  → Output: [batch_size, {INPUT_CHANNELS}, {INPUT_DIM}, {INPUT_DIM}]')
print('  ↓')
print('Sigmoid activation')
print('  ↓')
print(f'Reconstructed: [batch_size, {INPUT_CHANNELS}, {INPUT_DIM}, {INPUT_DIM}]')
print('=' * 60)
print(f'Training Parameters:')
print(f'  - Epochs: {EPOCHS}')
print(f'  - Batch Size: {BATCH_SIZE}')
print(f'  - CD Steps: {CD_K}')
print(f'  - Learning Rate: 1e-3')
print(f'  - Device: {"CUDA" if CUDA else "CPU"}')
print('=' * 60)
print(f'Sparsity & Sigma Parameters (demo_cdbn.m style):')
stats = convrbm.get_sparsity_stats()
print(f'  - Target Sparsity (pbias): {stats["pbias"]:.4f}')
print(f'  - Sparsity Weight (plambda): {stats["plambda"]:.1f}')
print(f'  - Sparsity Learning Rate (eta_sparsity): {stats["eta_sparsity"]:.3f}')
print(f'  - Initial Sigma: {stats["current_sigma"]:.3f}')
print(f'  - Sigma Stop: {stats["sigma_stop"]:.3f}')
print(f'  - Sigma Scheduling: {"Enabled" if convrbm.sigma_schedule else "Disabled"}')
print('=' * 60)

########## TRAINING Convolutional RBM ##########
print('Training Convolutional RBM...')

# ConvRBM initialized above

def evaluate_model(convrbm, train_loader, test_loader, max_train_samples=60000, max_test_samples=10000, eval_train=False):
    """Evaluate model performance"""
    
    sample_batch = torch.randn(1, INPUT_CHANNELS, INPUT_DIM, INPUT_DIM)
    if CUDA:
        sample_batch = sample_batch.cuda()
    sample_features = convrbm.get_hidden_features(sample_batch)
    feature_dim = sample_features.shape[1]
    
    # Extract training features
    train_features = []
    train_labels = []
    train_count = 0
    
    for batch, labels in train_loader:
        if train_count >= max_train_samples:
            break
        if batch.size(0) != BATCH_SIZE:
            continue
            
        batch = batch.view(batch.size(0), INPUT_CHANNELS, INPUT_DIM, INPUT_DIM)
        if CUDA:
            batch = batch.cuda()
            
        try:
            features = convrbm.get_hidden_features(batch)
            if CUDA:
                features = features.cpu()
            
            train_features.append(features.detach().numpy())
            train_labels.append(labels.numpy())
            train_count += BATCH_SIZE
            
        except Exception as e:
            continue
    
    # Extract test features
    test_features = []
    test_labels = []
    test_count = 0
    
    for batch, labels in test_loader:
        if test_count >= max_test_samples:
            break
        if batch.size(0) != BATCH_SIZE:
            continue
            
        batch = batch.view(batch.size(0), INPUT_CHANNELS, INPUT_DIM, INPUT_DIM)
        if CUDA:
            batch = batch.cuda()
            
        try:
            features = convrbm.get_hidden_features(batch)
            if CUDA:
                features = features.cpu()
            
            test_features.append(features.detach().numpy())
            test_labels.append(labels.numpy())
            test_count += BATCH_SIZE
            
        except Exception as e:
            continue
    
    if len(train_features) > 0 and len(test_features) > 0:
        # Combine features
        train_features = np.vstack(train_features)
        train_labels = np.hstack(train_labels)
        test_features = np.vstack(test_features)
        test_labels = np.hstack(test_labels)

        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features, train_labels)

        # Test set accuracy
        test_predictions = clf.predict(test_features)
        test_accuracy = np.mean(test_predictions == test_labels)

        if eval_train:
            # Training set accuracy
            train_predictions = clf.predict(train_features)
            train_accuracy = np.mean(train_predictions == train_labels)
            return test_accuracy, len(test_features), train_accuracy, len(train_features)
        else:
            return test_accuracy, len(test_features)
    else:
        if eval_train:
            return 0.0, 0, 0.0, 0
        else:
            return 0.0, 0

# Epoch 0 (Initial performance)
print('\n' + '='*50)
print('EPOCH 0 (Before Training)')
print('='*50)
if args.eval_train:
    test_accuracy, n_test_samples, train_accuracy, n_train_samples = evaluate_model(convrbm, train_loader, test_loader, eval_train=True)
    print(f'Epoch 0 - Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*n_test_samples)}/{n_test_samples}) on {n_test_samples} test samples')
    print(f'Epoch 0 - Train Accuracy: {train_accuracy:.4f} ({int(train_accuracy*n_train_samples)}/{n_train_samples}) on {n_train_samples} train samples')
else:
    test_accuracy, n_test_samples = evaluate_model(convrbm, train_loader, test_loader)
    print(f'Epoch 0 - Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*n_test_samples)}/{n_test_samples}) on {n_test_samples} test samples')

for epoch in range(EPOCHS):
    print(f'\n' + '='*50)
    print(f'EPOCH {epoch + 1}/{EPOCHS}')
    print('='*50)
    
    epoch_error = 0.0
    count = 0
    
    # Use tqdm to show training progress
    train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
    for batch, _ in train_pbar:
        # Ensure consistent batch size
        if batch.size(0) != BATCH_SIZE:
            continue
            
        # Ensure correct input format [batch_size, channels, height, width]
        batch = batch.view(batch.size(0), INPUT_CHANNELS, INPUT_DIM, INPUT_DIM)
        count += 1

        if CUDA:
            batch = batch.cuda()

        try:
            # Enable profiling for first few batches if requested
            enable_profile = args.profile and (count <= 3 or count % 10 == 0)

            if enable_profile:
                batch_error, profile_info = convrbm.contrastive_divergence(batch, profile=True)

                # Print detailed timing information
                total_time = profile_info['total_time']
                print(f"\nBATCH {count} PERFORMANCE PROFILE")
                print(f"Total Time: {total_time:.3f}s")
                print(f"  Positive Phase:")
                print(f"    - Convolution: {profile_info['positive_conv']:.3f}s ({profile_info['positive_conv']/total_time*100:.1f}%)")
                print(f"    - Pooling: {profile_info['positive_pooling']:.3f}s ({profile_info['positive_pooling']/total_time*100:.1f}%)")
                print(f"  Negative Phase (CD-{convrbm.k}): {profile_info['negative_cd_k_total']:.3f}s ({profile_info['negative_cd_k_total']/total_time*100:.1f}%)")
                print(f"    - Visible Reconstruction: {profile_info['negative_visible_reconstruction']:.3f}s ({profile_info['negative_visible_reconstruction']/total_time*100:.1f}%)")
                print(f"    - Convolution: {profile_info['negative_conv']:.3f}s ({profile_info['negative_conv']/total_time*100:.1f}%)")
                print(f"    - Pooling: {profile_info['negative_pooling']:.3f}s ({profile_info['negative_pooling']/total_time*100:.1f}%)")
                print(f"    - Sampling: {profile_info['negative_sampling']:.3f}s ({profile_info['negative_sampling']/total_time*100:.1f}%)")
                print(f"  Gradient Computation: {profile_info['gradient_computation']:.3f}s ({profile_info['gradient_computation']/total_time*100:.1f}%)")
                print(f"  Sparsity Regularization: {profile_info['sparsity_regularization']:.3f}s ({profile_info['sparsity_regularization']/total_time*100:.1f}%)")
                print(f"  Parameter Updates: {profile_info['parameter_updates']:.3f}s ({profile_info['parameter_updates']/total_time*100:.1f}%)")
                print(f"  Batch Size: {batch.shape[0]}, Error: {batch_error:.4f}")
                print("-" * 50)
            else:
                batch_error = convrbm.contrastive_divergence(batch)

            epoch_error += batch_error

            # Clean GPU memory
            if CUDA:
                torch.cuda.empty_cache()

            # Update progress bar info
            avg_error = epoch_error / count
            train_pbar.set_postfix({
                'batch': f'{count}',
                'avg_error': f'{avg_error:.4f}'
            })
                
        except Exception as e:
            continue
    
    train_pbar.close()
            
    avg_epoch_error = epoch_error / max(count, 1)
    print(f'Epoch {epoch + 1} Training Error: {avg_epoch_error:.4f}')

    # Evaluate performance after each epoch
    if args.eval_train:
        test_accuracy, n_test_samples, train_accuracy, n_train_samples = evaluate_model(convrbm, train_loader, test_loader, eval_train=True)
        print(f'Epoch {epoch + 1} - Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*n_test_samples)}/{n_test_samples}) on {n_test_samples} test samples')
        print(f'Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.4f} ({int(train_accuracy*n_train_samples)}/{n_train_samples}) on {n_train_samples} train samples')
        print(f'Epoch {epoch + 1} - Overfitting Gap: {train_accuracy - test_accuracy:.4f} (Train - Test)')
    else:
        test_accuracy, n_test_samples = evaluate_model(convrbm, train_loader, test_loader)
        print(f'Epoch {epoch + 1} - Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*n_test_samples)}/{n_test_samples}) on {n_test_samples} test samples')

########## FINAL EVALUATION ##########
print('\n' + '='*50)
print('FINAL EVALUATION')
print('='*50)
if args.eval_train:
    test_accuracy, n_test_samples, train_accuracy, n_train_samples = evaluate_model(convrbm, train_loader, test_loader, eval_train=True)
    print(f'Final Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*n_test_samples)}/{n_test_samples}) on {n_test_samples} test samples')
    print(f'Final Train Accuracy: {train_accuracy:.4f} ({int(train_accuracy*n_train_samples)}/{n_train_samples}) on {n_train_samples} train samples')
    print(f'Final Overfitting Gap: {train_accuracy - test_accuracy:.4f} (Train - Test)')
    if train_accuracy - test_accuracy > 0.1:
        print('WARNING: Significant overfitting detected (gap > 0.1)')
    elif train_accuracy - test_accuracy > 0.05:
        print('CAUTION: Moderate overfitting detected (gap > 0.05)')
    else:
        print('Good generalization (gap <= 0.05)')
else:
    test_accuracy, n_test_samples = evaluate_model(convrbm, train_loader, test_loader)
    print(f'Final Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*n_test_samples)}/{n_test_samples}) on {n_test_samples} test samples')

print('\nTraining completed successfully!')

