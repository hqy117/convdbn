import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.transforms
from tqdm import tqdm
import argparse

########## COMMAND LINE ARGUMENTS ##########
parser = argparse.ArgumentParser(description='Baseline Logistic Regression')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                    help='Dataset to use: mnist or cifar10 (default: mnist)')
args = parser.parse_args()

########## CONFIGURATION ##########
# Dataset selection from command line
DATASET = args.dataset

# Dataset-specific configurations
DATASET_CONFIG = {
    'mnist': {
        'data_folder': 'data/mnist',
        'dataset_class': torchvision.datasets.MNIST,
        'input_dims': 784,  # 28*28*1
        'num_classes': 10
    },
    'cifar10': {
        'data_folder': 'data/cifar10',
        'dataset_class': torchvision.datasets.CIFAR10,
        'input_dims': 3072,  # 32*32*3
        'num_classes': 10
    }
}

# Get current dataset configuration
config = DATASET_CONFIG[DATASET]
BATCH_SIZE = 64
DATA_FOLDER = config['data_folder']
DATASET_CLASS = config['dataset_class']
INPUT_DIMS = config['input_dims']
NUM_CLASSES = config['num_classes']

########## LOADING DATASET ##########
print(f'Loading {DATASET.upper()} dataset...')

train_dataset = DATASET_CLASS(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = DATASET_CLASS(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('='*70)
print('BASELINE EXPERIMENT: Raw Pixels + Logistic Regression')
print('='*70)
print(f'Dataset: {DATASET.upper()}')
print(f'Approach: {DATASET.upper()} raw pixels ({INPUT_DIMS} dims) → LogisticRegression')
print(f'Training samples: {len(train_dataset)}')
print(f'Test samples: {len(test_dataset)}')
print('No ConvRBM/ConvDBN feature extraction!')
print('='*70)

########## EXTRACT RAW PIXEL FEATURES ##########
print('\nExtracting raw pixel features from training data...')
train_features = []
train_labels = []

for batch_images, batch_labels in tqdm(train_loader, desc="Train Features"):
    # Flatten to INPUT_DIMS dimensions
    flattened = batch_images.view(batch_images.size(0), -1).numpy()
    train_features.append(flattened)
    train_labels.append(batch_labels.numpy())

train_features = np.vstack(train_features)
train_labels = np.hstack(train_labels)

print(f'Training features shape: {train_features.shape}')  # Should be (N, INPUT_DIMS)

print('\nExtracting raw pixel features from test data...')
test_features = []
test_labels = []

for batch_images, batch_labels in tqdm(test_loader, desc="Test Features"):
    # Flatten to INPUT_DIMS dimensions
    flattened = batch_images.view(batch_images.size(0), -1).numpy()
    test_features.append(flattened)
    test_labels.append(batch_labels.numpy())

test_features = np.vstack(test_features)
test_labels = np.hstack(test_labels)

print(f'Test features shape: {test_features.shape}')  # Should be (N, INPUT_DIMS)

########## TRAIN LOGISTIC REGRESSION ##########
print('\nTraining Logistic Regression on raw pixels...')
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(train_features, train_labels)

########## EVALUATE ##########
print('\nEvaluating...')
train_accuracy = classifier.score(train_features, train_labels)
test_accuracy = classifier.score(test_features, test_labels)

print('='*70)
print('BASELINE RESULTS')
print('='*70)
print(f'Dataset: {DATASET.upper()}')
print(f'Feature dimensions: {INPUT_DIMS} (raw pixels)')
print(f'Logistic Regression parameters: {INPUT_DIMS * NUM_CLASSES + NUM_CLASSES}')
print(f'Train accuracy: {train_accuracy:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')
print('='*70)
print('Now compare this with ConvRBM/ConvDBN results!')
print('If they are similar, then ConvRBM/ConvDBN may not be helpful.')
print('='*70)