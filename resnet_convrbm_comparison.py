"""
ResNet vs ConvRBM-ResNet Comparison Script
=========================================

This script compares:
1. Vanilla ResNet (small, <2M parameters, <=10 layers)
2. ResNet with ConvRBM as first layer

Both models are trained and evaluated on CIFAR-10 for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import argparse
from tqdm import tqdm

# Import ConvRBM components
from convdbn import ConvRBMLayer
from pooling import MultinomialMaxPool2d, SparseUnpool2d
from utils import compute_correlation, compute_bias_gradients, update_parameters_with_momentum

class BasicBlock(nn.Module):
    """Basic ResNet block for small ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """Small ResNet with <2M parameters and <=10 layers for CIFAR-10"""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(SmallResNet, self).__init__()
        self.in_planes = 32
        
        # First conv layer (layer 1)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResNet layers (layers 2-9)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # layers 2-3
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # layers 4-5
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2) # layers 6-7
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2) # layers 8-9
        
        # Final classifier (layer 10)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # First conv + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling + classifier
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ConvRBMFirstLayer(nn.Module):
    """ConvRBM layer adapted for use as first layer in ResNet"""
    
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, use_cuda=False):
        super(ConvRBMFirstLayer, self).__init__()
        
        # Create ConvRBM layer but adapt it for supervised learning
        # Note: ConvRBM works best without padding, so we use padding=0
        self.convrbm = ConvRBMLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,  # ConvRBM designed for no padding
            k=2,  # CD-k steps
            learning_rate=1e-3,
            momentum_coefficient=0.5,
            weight_decay=1e-4,
            use_cuda=use_cuda,
            spacing=2,  # smaller spacing for CIFAR-10
            pbias=0.002,
            plambda=0.0,  # Disable sparsity for supervised learning
            eta_sparsity=0.0,
            sigma=1.0,  # Start with sigma=1 and disable scheduling
            sigma_stop=1.0,
            sigma_schedule=False
        )
        
        # Batch norm for stability (like regular ResNet)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Flag to enable/disable RBM training
        self.rbm_training = False
        
    def forward(self, x):
        if self.training and self.rbm_training:
            # During RBM pre-training phase
            return self.convrbm.sample_hidden(x)
        else:
            # During supervised training - use as regular conv layer
            with torch.no_grad():
                # Forward through conv weights but without pooling
                # Use padding=0 since ConvRBM doesn't use padding
                hidden_pre = F.conv2d(x, self.convrbm.conv_weights, bias=self.convrbm.hidden_bias,
                                    stride=self.convrbm.stride, padding=0)
                # Apply sigma scaling and sigmoid
                hidden_pre_scaled = hidden_pre / (self.convrbm.sigma ** 2)
                hidden_activations = torch.sigmoid(hidden_pre_scaled)
            
            # Apply batch norm and return
            return F.relu(self.bn(hidden_activations))
    
    def pretrain_rbm(self, data_loader, epochs=5):
        """Pre-train the ConvRBM layer"""
        self.rbm_training = True
        self.train()
        
        print(f"Pre-training ConvRBM layer for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_error = 0.0
            count = 0
            
            pbar = tqdm(data_loader, desc=f"ConvRBM Epoch {epoch+1}/{epochs}")
            for batch, _ in pbar:
                if batch.size(0) != data_loader.batch_size:
                    continue
                    
                if self.convrbm.use_cuda:
                    batch = batch.cuda()
                
                # RBM contrastive divergence training
                error = self.convrbm.contrastive_divergence(batch)
                epoch_error += error
                count += 1
                
                pbar.set_postfix({'Error': f'{error:.4f}'})
            
            avg_error = epoch_error / count if count > 0 else 0
            print(f"ConvRBM Epoch {epoch+1}: Average Error = {avg_error:.4f}")
        
        self.rbm_training = False
        print("ConvRBM pre-training completed!")


class ConvRBMResNet(nn.Module):
    """ResNet with ConvRBM as first layer"""
    
    def __init__(self, block, num_blocks, num_classes=10, use_cuda=False):
        super(ConvRBMResNet, self).__init__()
        self.in_planes = 32
        
        # ConvRBM first layer instead of regular conv
        self.conv_rbm = ConvRBMFirstLayer(in_channels=3, out_channels=32, kernel_size=3, 
                                        stride=1, padding=1, use_cuda=use_cuda)
        
        # Rest is same as regular ResNet
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # ConvRBM first layer
        out = self.conv_rbm(x)
        
        # Rest of ResNet
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def pretrain_convrbm(self, data_loader, epochs=5):
        """Pre-train the ConvRBM first layer"""
        self.conv_rbm.pretrain_rbm(data_loader, epochs)


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_small_resnet():
    """Create small ResNet with <2M parameters"""
    # ResNet with [1,1,1,1] blocks = 10 layers total
    return SmallResNet(BasicBlock, [1, 1, 1, 1], num_classes=10)


def create_convrbm_resnet(use_cuda=False):
    """Create ResNet with ConvRBM first layer"""
    return ConvRBMResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, use_cuda=use_cuda)


def train_model(model, train_loader, test_loader, epochs=100, lr=0.1, model_name="Model"):
    """Train a model with standard supervised learning"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nTraining {model_name}...")
    print(f"Parameters: {count_parameters(model):,}")
    
    best_acc = 0
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Testing phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        scheduler.step()
        
        print(f'{model_name} Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%')
    
    return train_accuracies, test_accuracies, best_acc


def main():
    parser = argparse.ArgumentParser(description='ResNet vs ConvRBM-ResNet Comparison')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--rbm-pretrain-epochs', type=int, default=5, help='ConvRBM pre-training epochs')
    parser.add_argument('--no-pretrain', action='store_true', help='Skip ConvRBM pre-training')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    print("Preparing CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # For RBM pre-training, we need unnormalized data
    transform_pretrain = transforms.Compose([transforms.ToTensor()])
    pretrain_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_pretrain)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print("="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)

    # Create models
    vanilla_resnet = create_small_resnet()
    convrbm_resnet = create_convrbm_resnet(use_cuda=torch.cuda.is_available())

    print(f"Vanilla ResNet parameters: {count_parameters(vanilla_resnet):,}")
    print(f"ConvRBM ResNet parameters: {count_parameters(convrbm_resnet):,}")
    
    if count_parameters(vanilla_resnet) >= 2_000_000:
        print("WARNING: Model has >=2M parameters!")
    
    print("\nModel Architecture (10 layers total):")
    print("1. First conv layer (3->32 channels)")  
    print("2-3. ResNet block 1 (32 channels)")
    print("4-5. ResNet block 2 (32->64 channels, stride=2)")
    print("6-7. ResNet block 3 (64->128 channels, stride=2)") 
    print("8-9. ResNet block 4 (128->256 channels, stride=2)")
    print("10. Linear classifier (256->10)")

    print("="*80)

    # ConvRBM pre-training if enabled
    if not args.no_pretrain:
        print("\nPRE-TRAINING CONVRBM LAYER")
        print("="*40)
        convrbm_resnet.pretrain_convrbm(pretrain_loader, epochs=args.rbm_pretrain_epochs)

    # Train vanilla ResNet
    print("\n" + "="*80)
    print("TRAINING VANILLA RESNET")
    print("="*80)
    vanilla_train_acc, vanilla_test_acc, vanilla_best = train_model(
        vanilla_resnet, train_loader, test_loader, 
        epochs=args.epochs, lr=args.lr, model_name="Vanilla ResNet"
    )

    # Train ConvRBM ResNet
    print("\n" + "="*80)
    print("TRAINING CONVRBM RESNET")
    print("="*80)
    convrbm_train_acc, convrbm_test_acc, convrbm_best = train_model(
        convrbm_resnet, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr, model_name="ConvRBM ResNet"
    )

    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"Vanilla ResNet:")
    print(f"  - Parameters: {count_parameters(vanilla_resnet):,}")
    print(f"  - Best Test Accuracy: {vanilla_best:.2f}%")
    print(f"  - Final Train Accuracy: {vanilla_train_acc[-1]:.2f}%")
    print(f"  - Final Test Accuracy: {vanilla_test_acc[-1]:.2f}%")

    print(f"\nConvRBM ResNet:")
    print(f"  - Parameters: {count_parameters(convrbm_resnet):,}")
    print(f"  - Best Test Accuracy: {convrbm_best:.2f}%")  
    print(f"  - Final Train Accuracy: {convrbm_train_acc[-1]:.2f}%")
    print(f"  - Final Test Accuracy: {convrbm_test_acc[-1]:.2f}%")

    improvement = convrbm_best - vanilla_best
    print(f"\nAccuracy Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✓ ConvRBM ResNet outperforms Vanilla ResNet")
    elif improvement < -0.5:
        print("✗ ConvRBM ResNet underperforms Vanilla ResNet") 
    else:
        print("≈ Both models perform similarly")

    print("="*80)

if __name__ == '__main__':
    main()