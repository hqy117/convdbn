#!/usr/bin/env python3
"""
Script to create enhanced plots for ConvRBM training results.
Shows Test Accuracy and optionally Training Error in one figure with dual y-axes.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def extract_data(log_file):
    """Extract training errors, test accuracies, and train accuracies from log file."""
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found!")
        return [], [], [], [], [], []

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract training errors (epochs 1-50)
    error_pattern = r'Epoch (\d+) Training Error: ([\d.]+)'
    error_matches = re.findall(error_pattern, content)
    error_epochs = [int(e) for e, _ in error_matches]
    errors = [float(err) for _, err in error_matches]

    # Extract test accuracies (epochs 0-50)
    test_acc_pattern = r'Epoch (\d+) - Test Accuracy: ([\d.]+)'
    test_acc_matches = re.findall(test_acc_pattern, content)
    test_acc_epochs = [int(e) for e, _ in test_acc_matches]
    test_accuracies = [float(acc) * 100 for _, acc in test_acc_matches]  # Convert to percentage

    # Extract train accuracies (epochs 0-50)
    train_acc_pattern = r'Epoch (\d+) - Train Accuracy: ([\d.]+)'
    train_acc_matches = re.findall(train_acc_pattern, content)
    train_acc_epochs = [int(e) for e, _ in train_acc_matches]
    train_accuracies = [float(acc) * 100 for _, acc in train_acc_matches]  # Convert to percentage

    return error_epochs, errors, test_acc_epochs, test_accuracies, train_acc_epochs, train_accuracies

def extract_convdbn_data(log_file):
    """Extract training errors, test accuracies, and train accuracies from ConvDBN log file."""
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found!")
        return [], [], [], [], [], []

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract average errors and epoch numbers for ConvDBN
    # Pattern: "Average Error: X.XXXX" with preceding "Epoch X/50:"
    epoch_pattern = r'Epoch (\d+)/50:'
    error_pattern = r'Average Error: ([\d.]+)'

    # Find all epoch numbers and errors
    epoch_matches = re.findall(epoch_pattern, content)
    error_matches = re.findall(error_pattern, content)

    # Convert to appropriate format - match epochs with errors
    # ConvDBN has epochs 1-50, so we use those for both epochs and errors
    if len(epoch_matches) == len(error_matches):
        error_epochs = [int(e) for e in epoch_matches]  # Use all epochs 1-50
        errors = [float(err) for err in error_matches]
    else:
        # Fallback: create epoch list based on number of errors
        error_epochs = list(range(1, len(error_matches) + 1))
        errors = [float(err) for err in error_matches]

    # Extract test accuracies (epochs 0-50)
    test_acc_pattern = r'Test Accuracy: ([\d.]+)'
    test_acc_matches = re.findall(test_acc_pattern, content)
    test_acc_epochs = list(range(len(test_acc_matches)))  # 0, 1, 2, ..., 50
    test_accuracies = [float(acc) * 100 for acc in test_acc_matches]  # Convert to percentage

    # Extract train accuracies (epochs 0-50)
    train_acc_pattern = r'Train Accuracy: ([\d.]+)'
    train_acc_matches = re.findall(train_acc_pattern, content)
    train_acc_epochs = list(range(len(train_acc_matches)))  # 0, 1, 2, ..., 50
    train_accuracies = [float(acc) * 100 for acc in train_acc_matches]  # Convert to percentage

    return error_epochs, errors, test_acc_epochs, test_accuracies, train_acc_epochs, train_accuracies

def create_dual_axis_plot(error_epochs, errors, acc_epochs, accuracies, title, color_scheme, filename):
    """Create a dual-axis plot for one dataset."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot training error on left y-axis
    if errors:
        line1 = ax1.plot(error_epochs, errors, color=color_scheme['error'], linewidth=2.5, 
                        marker='o', markersize=4, label='Training Error')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Error', fontsize=12, fontweight='bold', color=color_scheme['error'])
        ax1.tick_params(axis='y', labelcolor=color_scheme['error'])
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis limit based on actual data
        max_epoch = max(error_epochs) if error_epochs else 50
        if accuracies and acc_epochs:
            max_epoch = max(max_epoch, max(acc_epochs))
        ax1.set_xlim(0, max_epoch)
        
        # Add error reduction annotation
        error_reduction = errors[0] - errors[-1]
        ax1.text(0.02, 0.98, f'Error Reduction: {error_reduction:.0f}\n({errors[0]:.0f} → {errors[-1]:.0f})', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color_scheme['error'], alpha=0.1),
                fontsize=10, fontweight='bold')
    
    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    if accuracies:
        line2 = ax2.plot(acc_epochs, accuracies, color=color_scheme['accuracy'], linewidth=2.5,
                        marker='s', markersize=4, label='Test Accuracy')
        ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold', color=color_scheme['accuracy'])
        ax2.tick_params(axis='y', labelcolor=color_scheme['accuracy'])

        # Add accuracy stats annotation
        max_acc = max(accuracies)
        final_acc = accuracies[-1]
        initial_acc = accuracies[0]
        ax2.text(0.98, 0.98, f'Max Accuracy: {max_acc:.2f}%\nFinal: {final_acc:.2f}%\nInitial: {initial_acc:.2f}%', 
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color_scheme['accuracy'], alpha=0.1),
                fontsize=10, fontweight='bold')
    
    # Set title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels() if errors else ([], [])
    lines2, labels2 = ax2.get_legend_handles_labels() if accuracies else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=11)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    return fig

def create_enhanced_dual_axis_plot(error_epochs, errors, test_acc_epochs, test_accuracies, train_acc_epochs, train_accuracies, title, color_scheme, filename):
    """Create a dual-axis plot with training error and both training & test accuracies."""
    if not error_epochs and not test_acc_epochs and not train_acc_epochs:
        print(f"No data found for {title}")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training error on left y-axis
    if errors:
        line1 = ax1.plot(error_epochs, errors, color=color_scheme['error'], linewidth=2.5,
                        marker='o', markersize=4, label='Training Error')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Error', fontsize=12, fontweight='bold', color=color_scheme['error'])
        ax1.tick_params(axis='y', labelcolor=color_scheme['error'])
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis limit based on actual data
        max_epoch = max(error_epochs) if error_epochs else 50
        if test_accuracies and test_acc_epochs:
            max_epoch = max(max_epoch, max(test_acc_epochs))
        if train_accuracies and train_acc_epochs:
            max_epoch = max(max_epoch, max(train_acc_epochs))
        ax1.set_xlim(0, max_epoch)

        # Add error reduction annotation
        error_reduction = errors[0] - errors[-1]
        ax1.text(0.02, 0.98, f'Error Reduction: {error_reduction:.0f}\n({errors[0]:.0f} → {errors[-1]:.0f})',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color_scheme['error'], alpha=0.1),
                fontsize=10, fontweight='bold')

    # Create second y-axis for accuracies
    ax2 = ax1.twinx()

    # Plot both test and train accuracies
    lines2 = []
    labels2 = []

    if test_accuracies:
        line2 = ax2.plot(test_acc_epochs, test_accuracies, color=color_scheme['test_accuracy'], linewidth=2.5,
                        marker='s', markersize=4, label='Test Accuracy')
        lines2.extend(line2)
        labels2.append('Test Accuracy')

    if train_accuracies:
        line3 = ax2.plot(train_acc_epochs, train_accuracies, color=color_scheme['train_accuracy'], linewidth=2.5,
                        marker='^', markersize=4, label='Train Accuracy')
        lines2.extend(line3)
        labels2.append('Train Accuracy')

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add accuracy stats annotation
    if test_accuracies and train_accuracies:
        test_max = max(test_accuracies)
        train_max = max(train_accuracies)
        overfitting_gap = train_accuracies[-1] - test_accuracies[-1]

        stats_text = f'Test: {test_accuracies[0]:.1f}% → {test_accuracies[-1]:.1f}% (Max: {test_max:.1f}%)\n'
        stats_text += f'Train: {train_accuracies[0]:.1f}% → {train_accuracies[-1]:.1f}% (Max: {train_max:.1f}%)\n'
        stats_text += f'Gap: {overfitting_gap:.1f}%'

        ax2.text(0.98, 0.98, stats_text,
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
                fontsize=9, fontweight='bold')

    # Set title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels() if errors else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=11)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")

    return fig

def create_test_accuracy_plot(error_epochs, errors, test_acc_epochs, test_accuracies, title, color_scheme, filename, include_train_error=False):
    """Create a plot with Test Accuracy on left axis and optionally Training Error on right axis."""
    if not test_acc_epochs:
        print(f"No test accuracy data found for {title}")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot test accuracy on left y-axis
    line1 = ax1.plot(test_acc_epochs, test_accuracies, color=color_scheme['test_accuracy'], linewidth=2.5,
                    marker='s', markersize=4, label='Test Accuracy')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold', color=color_scheme['test_accuracy'])
    ax1.tick_params(axis='y', labelcolor=color_scheme['test_accuracy'])
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis limit based on actual data
    max_epoch = max(test_acc_epochs) if test_acc_epochs else 50
    if include_train_error and error_epochs:
        max_epoch = max(max_epoch, max(error_epochs))
    ax1.set_xlim(0, max_epoch)
    
    # Add test accuracy stats annotation
    max_acc = max(test_accuracies)
    final_acc = test_accuracies[-1]
    initial_acc = test_accuracies[0]
    ax1.text(0.02, 0.98, f'Max Accuracy: {max_acc:.2f}%\nFinal: {final_acc:.2f}%\nInitial: {initial_acc:.2f}%', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_scheme['test_accuracy'], alpha=0.1),
            fontsize=10, fontweight='bold')

    lines = line1
    labels = ['Test Accuracy']

    # Optionally plot training error on right y-axis
    if include_train_error and errors and error_epochs:
        ax2 = ax1.twinx()
        line2 = ax2.plot(error_epochs, errors, color=color_scheme['error'], linewidth=2.5,
                        marker='o', markersize=4, label='Training Error')
        ax2.set_ylabel('Training Error', fontsize=12, fontweight='bold', color=color_scheme['error'])
        ax2.tick_params(axis='y', labelcolor=color_scheme['error'])
        
        # Add error reduction annotation
        error_reduction = errors[0] - errors[-1]
        ax2.text(0.98, 0.98, f'Error Reduction: {error_reduction:.0f}\n({errors[0]:.0f} → {errors[-1]:.0f})', 
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color_scheme['error'], alpha=0.1),
                fontsize=10, fontweight='bold')
        
        lines = line1 + line2
        labels = ['Test Accuracy', 'Training Error']

    # Set title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    ax1.legend(lines, labels, loc='lower left', fontsize=11)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    return fig

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create plots for ConvRBM/ConvDBN training results')
    parser.add_argument('--train-error', '-t', action='store_true', 
                       help='Include training error in the plots (default: False)')
    args = parser.parse_args()
    
    include_train_error = args.train_error
    
    # Load data from specific log files
    print("Loading MNIST data...")
    mnist_log_file = 'logs/convrbm/convrbm_mnist_0802.log'
    mnist_err_epochs, mnist_errors, mnist_test_acc_epochs, mnist_test_accs, mnist_train_acc_epochs, mnist_train_accs = extract_data(mnist_log_file)

    print("Loading CIFAR10 ConvRBM data...")
    cifar10_log_file = 'logs/convrbm/convrbm_cifar10_0802.log'
    cifar_err_epochs, cifar_errors, cifar_test_acc_epochs, cifar_test_accs, cifar_train_acc_epochs, cifar_train_accs = extract_data(cifar10_log_file)

    # Load ConvDBN data
    print("Loading ConvDBN data...")
    convdbn_files = [
        'logs/convdbn/cifar10_2layer_eval_train.log',
        'logs/convdbn/cifar10_3layer_eval_train.log',
        'logs/convdbn/cifar10_4layer_eval_train.log',
        'logs/convdbn/cifar10_5layer_eval_train.log'
    ]

    # Color schemes for each dataset (consistent colors as requested)
    # Red for test accuracy, Green for training error, Blue for train accuracy
    mnist_colors = {'error': '#2ca02c', 'test_accuracy': '#d62728', 'train_accuracy': '#1f77b4'}  # Green, Red, Blue
    cifar_colors = {'error': '#2ca02c', 'test_accuracy': '#d62728', 'train_accuracy': '#1f77b4'}  # Green, Red, Blue
    convdbn_colors = {'error': '#2ca02c', 'test_accuracy': '#d62728', 'train_accuracy': '#1f77b4'}  # Green, Red, Blue

    # Create MNIST plot (if data available)
    if mnist_test_accs:
        print("Creating MNIST plot...")
        import os
        log_dir = os.path.dirname(mnist_log_file)
        log_name = os.path.splitext(os.path.basename(mnist_log_file))[0]
        output_file = os.path.join(log_dir, f'{log_name}.png')
        
        fig1 = create_test_accuracy_plot(
            mnist_err_epochs, mnist_errors,
            mnist_test_acc_epochs, mnist_test_accs,
            'ConvRBM Training Results - MNIST Dataset',
            mnist_colors,
            output_file,
            include_train_error=include_train_error
        )
    # Create CIFAR10 ConvRBM plot (if data available) - put in same directory as log file
    if cifar_test_accs:
        print("Creating CIFAR10 ConvRBM plot...")
        # Generate output filename based on log file location and name
        import os
        log_dir = os.path.dirname(cifar10_log_file)
        log_name = os.path.splitext(os.path.basename(cifar10_log_file))[0]
        output_file = os.path.join(log_dir, f'{log_name}.png')

        fig2 = create_test_accuracy_plot(
            cifar_err_epochs, cifar_errors,
            cifar_test_acc_epochs, cifar_test_accs,
            'ConvRBM Training Results - CIFAR10 Dataset',
            cifar_colors,
            output_file,
            include_train_error=include_train_error
        )

    # Create ConvDBN plots
    for convdbn_file in convdbn_files:
        if os.path.exists(convdbn_file):
            print(f"Processing {os.path.basename(convdbn_file)}...")

            # Extract data using ConvDBN-specific function
            dbn_err_epochs, dbn_errors, dbn_test_acc_epochs, dbn_test_accs, dbn_train_acc_epochs, dbn_train_accs = extract_convdbn_data(convdbn_file)

            if dbn_test_accs:
                # Generate output filename
                log_dir = os.path.dirname(convdbn_file)
                log_name = os.path.splitext(os.path.basename(convdbn_file))[0]
                output_file = os.path.join(log_dir, f'{log_name}.png')

                # Extract layer info from filename for title
                if '2layer' in log_name:
                    title = 'ConvDBN Training Results - CIFAR10 (2-Layer)'
                elif '3layer' in log_name:
                    title = 'ConvDBN Training Results - CIFAR10 (3-Layer)'
                elif '4layer' in log_name:
                    title = 'ConvDBN Training Results - CIFAR10 (4-Layer)'
                elif '5layer' in log_name:
                    title = 'ConvDBN Training Results - CIFAR10 (5-Layer)'
                else:
                    title = f'ConvDBN Training Results - {log_name}'

                print(f"Creating {title} plot...")
                fig_dbn = create_test_accuracy_plot(
                    dbn_err_epochs, dbn_errors,
                    dbn_test_acc_epochs, dbn_test_accs,
                    title,
                    convdbn_colors,
                    output_file,
                    include_train_error=include_train_error
                )
    
    # Print summary
    print("\n" + "="*80)
    if include_train_error:
        print("DUAL-AXIS PLOTS GENERATED - Test Accuracy + Training Error")
    else:
        print("PLOTS GENERATED - Test Accuracy")
    print("="*80)

    if mnist_test_accs:
        print(f"MNIST ConvRBM:")
        print(f"  📊 Plot: logs/convrbm/convrbm_mnist_0802.png")
        print(f"  📈 Test Accuracy: {mnist_test_accs[0]:.2f}% → {mnist_test_accs[-1]:.2f}% (Max: {max(mnist_test_accs):.2f}%)")
        if include_train_error and mnist_errors:
            print(f"  📉 Training Error: {mnist_errors[0]:.0f} → {mnist_errors[-1]:.0f}")

    if cifar_test_accs:
        print(f"\nCIFAR10 ConvRBM:")
        print(f"  📊 Plot: {output_file}")
        print(f"  📈 Test Accuracy: {cifar_test_accs[0]:.2f}% → {cifar_test_accs[-1]:.2f}% (Max: {max(cifar_test_accs):.2f}%)")
        if include_train_error and cifar_errors:
            print(f"  📉 Training Error: {cifar_errors[0]:.0f} → {cifar_errors[-1]:.0f}")

    # Print ConvDBN summaries
    convdbn_count = 0
    for convdbn_file in convdbn_files:
        if os.path.exists(convdbn_file):
            dbn_err_epochs, dbn_errors, dbn_test_acc_epochs, dbn_test_accs, dbn_train_acc_epochs, dbn_train_accs = extract_convdbn_data(convdbn_file)
            if dbn_test_accs:
                convdbn_count += 1
                log_name = os.path.splitext(os.path.basename(convdbn_file))[0]
                layer_info = log_name.replace('cifar10_', '').replace('_eval_train', '')

                print(f"\nCIFAR10 ConvDBN ({layer_info}):")
                print(f"  📊 Plot: logs/convdbn/{log_name}.png")
                print(f"  📈 Test Accuracy: {dbn_test_accs[0]:.2f}% → {dbn_test_accs[-1]:.2f}% (Max: {max(dbn_test_accs):.2f}%)")
                if include_train_error and dbn_errors:
                    print(f"  📉 Training Error: {dbn_errors[0]:.4f} → {dbn_errors[-1]:.4f}")

    print(f"\n✅ All plots saved successfully! ({(1 if mnist_test_accs else 0) + (1 if cifar_test_accs else 0) + convdbn_count} plots generated)")

if __name__ == "__main__":
    main()
