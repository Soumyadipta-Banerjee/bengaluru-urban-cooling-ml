"""
Evaluation and visualization module for:
1. SVR Kernel & Hyperparameter Comparison
2. Neural-Network-KKT Fuzzy Regression Intervals & Uncertainty Bounds
3. Confusion Matrix Heatmaps (Low, Moderate, High)
4. Stratified Class Distribution
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_k_vs_validation_score(tuning_df, best_k, output_path="outputs/plots/k_vs_validation_score.png"):
    """
    Plots the validation accuracy and balanced accuracy across k neighbours to identify optimal k.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    plt.plot(tuning_df['k'], tuning_df['train_accuracy'], marker='o', linestyle='--', color='#2b5c8f', label='Training Accuracy', alpha=0.7)
    plt.plot(tuning_df['k'], tuning_df['val_accuracy'], marker='s', color='#2ca02c', linewidth=2, label='Validation Overall Accuracy')
    plt.plot(tuning_df['k'], tuning_df['val_balanced_accuracy'], marker='^', color='#d62728', linewidth=2, label='Validation Balanced Accuracy')
    
    best_row = tuning_df[tuning_df['k'] == best_k].iloc[0]
    plt.axvline(x=best_k, color='#e377c2', linestyle=':', linewidth=2, label=f'Selected k = {best_k}')
    plt.scatter([best_k], [best_row['val_balanced_accuracy']], color='#d62728', s=120, zorder=5, edgecolor='black')
    
    plt.title("Assignment 4: k-NN Hyperparameter Tuning (k vs Validation Score)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Nearest Neighbours (k)", fontsize=11, labelpad=8)
    plt.ylabel("Accuracy Score", fontsize=11, labelpad=8)
    plt.xticks(tuning_df['k'])
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_svr_kernel_comparison(results_df, output_path="outputs/plots/svr_kernel_comparison.png"):
    """
    Plots the validation performance across SVR Kernels (RBF, Poly, Linear) and C parameters.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    # Group by kernel and C, taking max validation balanced accuracy
    grouped = results_df.groupby(['kernel', 'C'])['val_bal_acc'].max().unstack()
    
    ax = grouped.plot(kind='bar', figsize=(9, 5.5), colormap='viridis', edgecolor='black', width=0.8)
    plt.title("Support Vector Regression: Kernel & Regularization (C) Comparison", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("SVR Kernel Type", fontsize=11, labelpad=8)
    plt.ylabel("Validation Balanced Accuracy", fontsize=11, labelpad=8)
    plt.xticks(rotation=0)
    plt.legend(title='Regularization C', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_fuzzy_kkt_intervals(y_true, center_preds, lower_bounds, upper_bounds, 
                             n_samples=50, output_path="outputs/plots/fuzzy_kkt_bounds.png"):
    """
    Plots the Neural-Network-KKT Fuzzy Regression predictions showing the fuzzy interval [y_lower, y_upper]
    and demonstrating the Karush-Kuhn-Tucker inclusion constraints.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(11, 6), dpi=300)
    
    # Sort by true values for clean presentation
    idx = np.argsort(y_true)[:n_samples]
    x_axis = np.arange(len(idx))
    
    y_t = y_true[idx]
    y_c = center_preds[idx]
    y_l = lower_bounds[idx]
    y_u = upper_bounds[idx]
    
    plt.fill_between(x_axis, y_l, y_u, color='#74c476', alpha=0.35, label='KKT Fuzzy Interval [y_lower, y_upper]')
    plt.plot(x_axis, y_c, color='#238b45', linewidth=2, label='NN-KKT Center Prediction (ŷ)')
    plt.scatter(x_axis, y_t, color='#d94801', s=35, zorder=5, label='True Cooling Intensity (ΔT °C)')
    
    plt.title("Neural-Network-KKT Fuzzy Regression: Uncertainty Spreads & Inclusion Bounds", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Test Sample Index (Sorted by Cooling Intensity)", fontsize=11, labelpad=8)
    plt.ylabel("Cooling Intensity (ΔT in °C)", fontsize=11, labelpad=8)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", output_path="outputs/plots/confusion_matrix.png"):
    if class_names is None:
        class_names = ["Low", "Moderate", "High"]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=300)
    
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"
            
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 11, 'weight': 'bold'})
                
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Cooling Potential Class", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Cooling Potential Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_class_distribution(y_train, y_val, y_test, class_names=None, output_path="outputs/plots/class_distribution.png"):
    if class_names is None:
        class_names = ["Low", "Moderate", "High"]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_counts = [np.sum(y_train == i) for i in [1, 2, 3]]
    val_counts = [np.sum(y_val == i) for i in [1, 2, 3]]
    test_counts = [np.sum(y_test == i) for i in [1, 2, 3]]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.bar(x - width, train_counts, width, label='Training Set (70%)', color='#4a90e2')
    plt.bar(x, val_counts, width, label='Validation Set (15%)', color='#50e3c2')
    plt.bar(x + width, test_counts, width, label='Testing Set (15%)', color='#f5a623')
    
    plt.title("Stratified Cooling Potential Class Distribution Across Splits", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Cooling Potential Class", fontsize=11, labelpad=8)
    plt.ylabel("Number of Samples", fontsize=11, labelpad=8)
    plt.xticks(x, class_names)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
