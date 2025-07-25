import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.transforms
from convdbn import ConvDBN
from tqdm import tqdm
import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='ConvDBN Training')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                    help='Dataset to use: mnist or cifar10 (default: mnist)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs (default: 50)')
parser.add_argument('--subset-size', type=int, default=None,
                    help='Use only a subset of the dataset for quick testing (default: use full dataset)')
parser.add_argument('--layers', type=int, default=3, choices=[2, 3, 4, 5],
                    help='Number of ConvDBN layers: 2-5 (default: 3)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--eval-train', action='store_true',
                    help='Also evaluate accuracy on training set to check for overfitting')
args = parser.parse_args()

# Random seed setup
import random
SEED = args.seed
print(f'Setting random seed to: {SEED}')

# Set all random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
# Dataset selection from command line
DATASET = args.dataset
NUM_LAYERS = args.layers

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

BATCH_SIZE = 64
CD_K = 2
LR = 1e-3  
TOTAL_EPOCHS = args.epochs  # from command line arguments
TOTAL_LAYERS = NUM_LAYERS


CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

print(f"GPU Available: {CUDA}")
if CUDA:
    print(f"GPU Device: {torch.cuda.get_device_name(CUDA_DEVICE)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / 1e9:.1f} GB")
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

# Network architecture info
print(f'\nConvDBN ({NUM_LAYERS}-Layer Convolutional Deep Belief Network) Training ({DATASET.upper()})')
print('-' * 60)
print('Training Strategy:')
print(f'  - Each epoch trains all {NUM_LAYERS} layers sequentially')
layer_sequence = ' -> '.join([f'Layer {i+1}' for i in range(NUM_LAYERS)])
print(f'  - {layer_sequence} = 1 complete epoch')
print('  - Classification accuracy tested after each complete epoch')
print('-' * 60)
target_baseline = 768 if DATASET == 'mnist' else 3072
print(f'Architecture (Target: ~{target_baseline} features for fair comparison):')
print(f'  Input: [batch, {INPUT_CHANNELS}, {INPUT_DIM}, {INPUT_DIM}] ({DATASET.upper()} images)')

print('Architecture Overview:')
print(f'  - {NUM_LAYERS} ConvRBM layers with stride=1 convolutions')
print(f'  - Kernel sizes: 5x5 for most layers, 3x3 for some')
if DATASET == 'mnist':
    if NUM_LAYERS == 2:
        print('  - MaxPool(2x2) after first layer for dimensionality reduction')
else:  # CIFAR10
    if NUM_LAYERS == 2:
        print('  - MaxPool(2x2) after first layer for dimensionality reduction')
print('  - Target: feature dimensions close to baseline for fair comparison')
print('  - Actual feature dimensions will be shown after model creation')
print('-' * 60)

print('Parameter Analysis:')
print('  Note: Actual feature dimensions and parameter count will be calculated after model creation')
print('-' * 60)

# Training ConvDBN
print('\nCreating ConvDBN...')
convdbn = ConvDBN(k=CD_K, use_cuda=CUDA, learning_rate=LR,
                  input_channels=INPUT_CHANNELS, input_dim=INPUT_DIM,
                  dataset=DATASET, num_layers=NUM_LAYERS,
                  pbias=0.002, plambda=5.0, eta_sparsity=0.0, sigma=0.2, sigma_stop=0.1, sigma_schedule=True)

# Get actual feature dimensions and architecture details from the model
print('\nAnalyzing actual model architecture...')
with torch.no_grad():
    test_input = torch.randn(1, INPUT_CHANNELS, INPUT_DIM, INPUT_DIM)
    if CUDA:
        test_input = test_input.cuda()

    # Get activations through all layers to show actual dimensions
    activations, sparse_patterns = convdbn.forward_all_layers(test_input)

    print('Actual Layer Dimensions:')
    print(f'  Input: {list(activations[0].shape)} ({DATASET.upper()} images)')

    for i, layer in enumerate(convdbn.layers):
        layer_input_shape = list(activations[i].shape)
        layer_output_shape = list(activations[i+1].shape)

        # Get layer parameters
        in_ch = layer.in_channels
        out_ch = layer.out_channels
        kernel_size = layer.kernel_size

        layer_name = f'Layer {i+1}'

        # Check if this layer has multinomial pooling
        has_pooling = (NUM_LAYERS == 2 and i == 0 and len(convdbn.multinomial_pools) > 0)

        if has_pooling:
            # Show pre-pool and post-pool dimensions
            pre_pool_shape = layer_output_shape.copy()
            if len(sparse_patterns) > 0:
                # Calculate pre-pool dimensions (before multinomial pooling)
                pool_spacing = convdbn.multinomial_pools[0].spacing
                pre_pool_h = layer_output_shape[2] * pool_spacing
                pre_pool_w = layer_output_shape[3] * pool_spacing
                print(f'  {layer_name}: Conv2d({in_ch}→{out_ch}, {kernel_size}×{kernel_size}) → {layer_input_shape} → [{out_ch}, {pre_pool_h}, {pre_pool_w}]')
                print(f'    → MultinomialPool({pool_spacing}×{pool_spacing}) → {layer_output_shape}')
            else:
                print(f'  {layer_name}: Conv2d({in_ch}→{out_ch}, {kernel_size}×{kernel_size}) → {layer_input_shape} → {layer_output_shape}')
        else:
            print(f'  {layer_name}: Conv2d({in_ch}→{out_ch}, {kernel_size}×{kernel_size}) → {layer_input_shape} → {layer_output_shape}')

    actual_features = convdbn.get_features(test_input)
    actual_feature_dims = actual_features.shape[1]

print(f'\nFinal Feature Analysis:')
print(f'  ACTUAL Final Features: {actual_feature_dims} dimensions')
baseline = 768 if DATASET == 'mnist' else 3072
ratio = actual_feature_dims / baseline
print(f'  Ratio to baseline ({baseline}): {ratio:.3f}')
status = 'GOOD' if 0.85 <= ratio <= 1.15 else 'ACCEPTABLE' if 0.7 <= ratio <= 1.3 else 'OUT OF RANGE'
print(f'  Status: {status}')
print('-' * 60)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(convdbn)
print(f'ConvDBN Total Parameters: {total_params:,}')

def evaluate_classification(convdbn, train_loader, test_loader, eval_train=False):
    """TODO: Add English docstring"""
    
    train_features = []
    train_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in train_loader:
            if CUDA:
                batch_images = batch_images.cuda()
            features = convdbn.get_features(batch_images)
            train_features.append(features.cpu().numpy())
            train_labels.append(batch_labels.numpy())

    if len(train_features) == 0:
        if eval_train:
            return 0.0, 0, 0.0, 0
        else:
            return 0.0

    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)

    # Extract test features
    test_features = []
    test_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            if CUDA:
                batch_images = batch_images.cuda()
            features = convdbn.get_features(batch_images)
            test_features.append(features.cpu().numpy())
            test_labels.append(batch_labels.numpy())

    test_features = np.vstack(test_features)
    test_labels = np.hstack(test_labels)

    # Train classifier
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_features, train_labels)

    # Test set accuracy
    test_accuracy = classifier.score(test_features, test_labels)

    if eval_train:
        # Training set accuracy
        train_accuracy = classifier.score(train_features, train_labels)
        return test_accuracy, len(test_features), train_accuracy, len(train_features)
    else:
        return test_accuracy

# Training
print('\nEPOCH-BASED TRAINING')
print('-' * 40)


print(f"Using full {DATASET.upper()} training set: {len(train_dataset)} samples")
print(f"Using full {DATASET.upper()} test set: {len(test_dataset)} samples")

# Epoch 0: Initial performance
print("Epoch 0 (Before Training):")
if args.eval_train:
    test_accuracy, n_test_samples, train_accuracy, n_train_samples = evaluate_classification(convdbn, train_loader, test_loader, eval_train=True)
    print(f"  Test Accuracy: {test_accuracy:.4f} ({n_test_samples} test samples)")
    print(f"  Train Accuracy: {train_accuracy:.4f} ({n_train_samples} train samples)")
    print(f"  Overfitting Gap: {train_accuracy - test_accuracy:.4f} (Train - Test)\n")
else:
    test_accuracy = evaluate_classification(convdbn, train_loader, test_loader)
    print(f"  Test Accuracy: {test_accuracy:.4f}\n")


for epoch in range(1, TOTAL_EPOCHS + 1):
    print(f"Epoch {epoch}/{TOTAL_EPOCHS}:")

    
    epoch_errors = []

    for layer_idx in range(TOTAL_LAYERS):
        layer_name = f"Layer {layer_idx + 1}"

        
        with tqdm(desc=f"  {layer_name}", leave=False) as pbar:
            avg_error = convdbn.train_layer(layer_idx + 1, train_loader, 1)  
            epoch_errors.append(avg_error)
            pbar.set_postfix({"Error": f"{avg_error:.4f}"})

    
    avg_epoch_error = np.mean(epoch_errors)

    if args.eval_train:
        test_accuracy, n_test_samples, train_accuracy, n_train_samples = evaluate_classification(convdbn, train_loader, test_loader, eval_train=True)
        
        error_str = ", ".join([f"{err:.4f}" for err in epoch_errors])
        print(f"  Layers Error: {error_str}")
        print(f"  Average Error: {avg_epoch_error:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({n_test_samples} test samples)")
        print(f"  Train Accuracy: {train_accuracy:.4f} ({n_train_samples} train samples)")
        print(f"  Overfitting Gap: {train_accuracy - test_accuracy:.4f} (Train - Test)\n")
    else:
        test_accuracy = evaluate_classification(convdbn, train_loader, test_loader)
        
        error_str = ", ".join([f"{err:.4f}" for err in epoch_errors])
        print(f"  Layers Error: {error_str}")
        print(f"  Average Error: {avg_epoch_error:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}\n")

print('-' * 40)
print('TRAINING COMPLETED!')
print('-' * 40)


total_params = count_parameters(convdbn)
baseline = 768 if DATASET == 'mnist' else 3072

if args.eval_train:
    test_accuracy, n_test_samples, train_accuracy, n_train_samples = evaluate_classification(convdbn, train_loader, test_loader, eval_train=True)
    print(f'Final Test Accuracy: {test_accuracy:.4f} ({n_test_samples} test samples)')
    print(f'Final Train Accuracy: {train_accuracy:.4f} ({n_train_samples} train samples)')
    print(f'Final Overfitting Gap: {train_accuracy - test_accuracy:.4f} (Train - Test)')
    if train_accuracy - test_accuracy > 0.1:
        print('WARNING: Significant overfitting detected (gap > 0.1)')
    elif train_accuracy - test_accuracy > 0.05:
        print('CAUTION: Moderate overfitting detected (gap > 0.05)')
    else:
        print('Good generalization (gap <= 0.05)')
else:
    test_accuracy = evaluate_classification(convdbn, train_loader, test_loader)
    print(f'Final Test Accuracy: {test_accuracy:.4f}')

print(f'Total Parameters: {total_params:,}')
print(f'Feature Dimensions: {actual_feature_dims}')
print(f'Feature Ratio to Baseline ({baseline}): {actual_feature_dims/baseline:.3f}')
print(f'Architecture: {NUM_LAYERS}-layer ConvDBN on {DATASET.upper()}')
print('-' * 40)