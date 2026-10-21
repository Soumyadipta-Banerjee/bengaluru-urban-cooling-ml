"""
Algorithm Implementation:
1. Support Vector Regression (SVR) with Kernels (RBF, Polynomial, Linear)
2. Neural-Network-KKT Fuzzy Regression with Karush-Kuhn-Tucker inclusion constraints
Both models predict continuous cooling intensity (ΔT in °C) and map to tertile classes (Low, Moderate, High).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =========================================================================
# 1. Continuous to Tertile Class Conversion
# =========================================================================
def continuous_to_classes(y_pred_cont, tertiles):
    """
    Converts continuous cooling intensity predictions into 3 discrete tertile classes:
    1: Low, 2: Moderate, 3: High
    """
    t_low, t_high = tertiles
    classes = np.zeros(len(y_pred_cont), dtype=np.uint8)
    classes[y_pred_cont <= t_low] = 1
    classes[(y_pred_cont > t_low) & (y_pred_cont <= t_high)] = 2
    classes[y_pred_cont > t_high] = 3
    return classes

# =========================================================================
# 2. Support Vector Regression (SVR) with Kernels
# =========================================================================
class SVRClassifierWrapper:
    """
    Wraps SVR to act as both a continuous regressor and a 3-class tertile classifier.
    """
    def __init__(self, kernel='rbf', C=10.0, epsilon=0.1, gamma='scale', degree=2, max_iter=3000):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=degree, max_iter=max_iter)
        self.tertiles = None

    def fit(self, X, y_cont, tertiles):
        self.tertiles = tertiles
        self.model.fit(X, y_cont)
        return self

    def predict_continuous(self, X):
        return self.model.predict(X)

    def predict(self, X):
        y_cont = self.predict_continuous(X)
        return continuous_to_classes(y_cont, self.tertiles)

def tune_svr_kernels(X_train, y_cont_train, y_train_class, 
                     X_val, y_cont_val, y_val_class, tertiles):
    """
    Evaluates key SVR kernels (RBF, Polynomial, Linear) and hyperparameters on validation set.
    """
    print("\n--- Tuning Support Vector Regression (SVR) with Kernels ---")
    param_configs = [
        {'kernel': 'rbf', 'C': 50.0, 'epsilon': 0.01, 'gamma': 0.015, 'degree': 2},
        {'kernel': 'rbf', 'C': 20.0, 'epsilon': 0.02, 'gamma': 0.02, 'degree': 2},
        {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.05, 'gamma': 'scale', 'degree': 2},
        {'kernel': 'rbf', 'C': 5.0, 'epsilon': 0.05, 'gamma': 0.02, 'degree': 2},
        {'kernel': 'poly', 'C': 10.0, 'epsilon': 0.05, 'gamma': 'scale', 'degree': 2},
        {'kernel': 'linear', 'C': 10.0, 'epsilon': 0.05, 'gamma': 'scale', 'degree': 2},
    ]

    best_val_score = -1.0
    best_config = None
    best_svr = None
    results = []

    for cfg in param_configs:
        print(f"  Testing SVR [{cfg['kernel'].upper()} kernel, C={cfg['C']}, gamma={cfg['gamma']}]...")
        wrapper = SVRClassifierWrapper(
            kernel=cfg['kernel'], C=cfg['C'], epsilon=cfg['epsilon'],
            gamma=cfg['gamma'], degree=cfg['degree'], max_iter=3000
        )
        wrapper.fit(X_train, y_cont_train, tertiles)
        
        val_cont = wrapper.predict_continuous(X_val)
        val_preds = wrapper.predict(X_val)
        
        val_r2 = r2_score(y_cont_val, val_cont)
        val_rmse = np.sqrt(mean_squared_error(y_cont_val, val_cont))
        val_bal_acc = balanced_accuracy_score(y_val_class, val_preds)
        val_f1 = f1_score(y_val_class, val_preds, average='macro', zero_division=0)
        
        results.append({
            'kernel': cfg['kernel'], 'C': cfg['C'], 'epsilon': cfg['epsilon'], 'gamma': str(cfg['gamma']),
            'val_r2': val_r2, 'val_rmse': val_rmse,
            'val_bal_acc': val_bal_acc, 'val_macro_f1': val_f1
        })
        
        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_config = cfg
            best_svr = wrapper

    results_df = pd.DataFrame(results)
    print(f"\nOptimal SVR Identified: Kernel={best_config['kernel'].upper()}, C={best_config['C']}, "
          f"eps={best_config['epsilon']}, gamma={best_config['gamma']} -> Val Bal Acc: {best_val_score:.4f}")
    return best_svr, best_config, results_df


# =========================================================================
# 3. Neural-Network-KKT Fuzzy Regression
# =========================================================================
class FuzzyKKTNetwork(nn.Module):
    """
    Multi-layer neural network for Fuzzy Regression.
    Dual output:
    1. center_head: predicts center cooling intensity y_hat
    2. spread_head: predicts non-negative fuzzy spread s >= 0 via Softplus
    Fuzzy interval is [y_hat - s, y_hat + s].
    """
    def __init__(self, input_dim=64, hidden_dim=128):
        super(FuzzyKKTNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        # Center head
        self.center_head = nn.Linear(hidden_dim // 2, 1)
        # Fuzzy spread head (must be non-negative)
        self.spread_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(beta=1.0)
        )

    def forward(self, x):
        features = self.shared(x)
        center = self.center_head(features)
        spread = self.spread_head(features) + 1e-4 # epsilon for numerical stability
        return center, spread

class FuzzyKKTLoss(nn.Module):
    """
    Karush-Kuhn-Tucker (KKT) Constrained Fuzzy Regression Loss.
    
    L = MSE(center, y) + alpha * spread + beta * KKT_violation_penalty
    where KKT violations enforce fuzzy inclusion:
      y_lower <= y <= y_upper
      g1 = y - (center + spread) <= 0
      g2 = (center - spread) - y <= 0
    """
    def __init__(self, alpha=0.05, beta=10.0):
        super(FuzzyKKTLoss, self).__init__()
        self.alpha = alpha # Weight for minimizing fuzziness (spread)
        self.beta = beta   # Penalty for violating KKT inclusion constraints

    def forward(self, center, spread, target):
        center = center.squeeze(-1)
        spread = spread.squeeze(-1)
        
        # 1. Predictive center accuracy (MSE)
        mse_loss = torch.mean((center - target) ** 2)
        
        # 2. Minimizing total vagueness / fuzzy spread
        spread_loss = torch.mean(spread)
        
        # 3. KKT inequality constraint penalties (Relu violation squared)
        upper_bound = center + spread
        lower_bound = center - spread
        
        violation_upper = torch.relu(target - upper_bound)
        violation_lower = torch.relu(lower_bound - target)
        kkt_penalty = torch.mean(violation_upper ** 2 + violation_lower ** 2)
        
        total_loss = mse_loss + self.alpha * spread_loss + self.beta * kkt_penalty
        return total_loss, mse_loss.item(), kkt_penalty.item()

class FuzzyKKTRegressionModel:
    """
    Scikit-learn compatible wrapper for the Neural-Network-KKT Fuzzy Regression model.
    """
    def __init__(self, input_dim=64, hidden_dim=128, alpha=0.05, beta=10.0, 
                 lr=0.002, epochs=80, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.net = None
        self.tertiles = None

    def fit(self, X_train, y_cont_train, tertiles, X_val=None, y_cont_val=None):
        self.tertiles = tertiles
        self.net = FuzzyKKTNetwork(input_dim=self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
        criterion = FuzzyKKTLoss(alpha=self.alpha, beta=self.beta)
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_cont_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"\n--- Training Neural-Network-KKT Fuzzy Regression ({self.epochs} epochs) ---")
        self.net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                center, spread = self.net(batch_x)
                loss, _, _ = criterion(center, spread, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            scheduler.step()

            if epoch % 20 == 0 or epoch == self.epochs:
                avg_loss = total_loss / len(dataset)
                print(f"  Epoch [{epoch:02d}/{self.epochs}] - Total KKT Loss: {avg_loss:.4f}")
        return self

    def predict_fuzzy(self, X):
        """
        Returns (center, lower_bound, upper_bound, spread) as numpy arrays.
        """
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            center, spread = self.net(x_tensor)
            center = center.squeeze(-1).numpy()
            spread = spread.squeeze(-1).numpy()
            lower = center - spread
            upper = center + spread
        return center, lower, upper, spread

    def predict_continuous(self, X):
        center, _, _, _ = self.predict_fuzzy(X)
        return center

    def predict(self, X):
        center = self.predict_continuous(X)
        return continuous_to_classes(center, self.tertiles)

# =========================================================================
# 4. Comprehensive Evaluator
# =========================================================================
def evaluate_model_pipeline(model, X_test, y_cont_test, y_test_class, model_name="Model"):
    """
    Evaluates both continuous regression metrics (R2, RMSE, MAE) 
    and tertile classification metrics (Accuracy, Balanced Accuracy, Macro F1, Confusion Matrix).
    """
    preds_cont = model.predict_continuous(X_test)
    preds_class = model.predict(X_test)

    r2 = r2_score(y_cont_test, preds_cont)
    rmse = np.sqrt(mean_squared_error(y_cont_test, preds_cont))
    mae = mean_absolute_error(y_cont_test, preds_cont)

    acc = accuracy_score(y_test_class, preds_class)
    bal_acc = balanced_accuracy_score(y_test_class, preds_class)
    macro_prec = precision_score(y_test_class, preds_class, average='macro', zero_division=0)
    macro_rec = recall_score(y_test_class, preds_class, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test_class, preds_class, average='macro', zero_division=0)

    cm = confusion_matrix(y_test_class, preds_class, labels=[1, 2, 3])
    class_recalls = cm.diagonal() / cm.sum(axis=1)

    print(f"\n=======================================================")
    print(f" Evaluation Report: {model_name}")
    print(f"=======================================================")
    print(f" Continuous Regression Metrics:")
    print(f"   R² Score:            {r2:.4f}")
    print(f"   RMSE:                {rmse:.4f} °C")
    print(f"   MAE:                 {mae:.4f} °C")
    print(f" Classification (Tertile) Metrics:")
    print(f"   Overall Accuracy:    {acc*100:.2f}%")
    print(f"   Balanced Accuracy:   {bal_acc*100:.2f}%")
    print(f"   Macro Precision:     {macro_prec*100:.2f}%")
    print(f"   Macro Recall:        {macro_rec*100:.2f}%")
    print(f"   Macro F1-Score:      {macro_f1*100:.2f}%")
    print(f" Class-wise Recall:")
    print(f"   Low Cooling:         {class_recalls[0]*100:.2f}%")
    print(f"   Moderate Cooling:    {class_recalls[1]*100:.2f}%")
    print(f"   High Cooling:        {class_recalls[2]*100:.2f}%")
    print(f" Confusion Matrix:\n{cm}")

    return {
        'model_name': model_name,
        'r2': r2, 'rmse': rmse, 'mae': mae,
        'overall_accuracy': acc, 'balanced_accuracy': bal_acc,
        'macro_precision': macro_prec, 'macro_recall': macro_rec, 'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'class_recalls': class_recalls,
        'preds_cont': preds_cont,
        'preds_class': preds_class
    }
