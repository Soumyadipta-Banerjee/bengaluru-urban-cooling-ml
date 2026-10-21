"""
Model training and hyperparameter tuning module for k-Nearest Neighbours.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def tune_knn(X_train, y_train, X_val, y_val, k_values=None):
    """
    Evaluates k-NN across a range of k values to find the optimal k on the validation set.
    Returns the tuning dataframe and the best k.
    """
    if k_values is None:
        k_values = list(range(1, 36, 2))
        
    records = []
    best_val_score = -1.0
    best_k = k_values[0]
    
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_bal_acc = balanced_accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
        
        records.append({
            'k': k,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_balanced_accuracy': val_bal_acc,
            'val_macro_f1': val_f1
        })
        
        # Primary selection metric: Balanced Accuracy (robust to slight class skew)
        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_k = k
            
    tuning_df = pd.DataFrame(records)
    print(f"Optimal k identified: k={best_k} with Validation Balanced Accuracy = {best_val_score:.4f}")
    return tuning_df, best_k

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Computes all required assignment metrics:
    Overall Accuracy, Balanced Accuracy, Macro Precision/Recall/F1, Class-wise Recall, Confusion Matrix.
    """
    if class_names is None:
        class_names = ["Low", "Moderate", "High"]
        
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    macro_prec = precision_score(y_test, preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test, preds, labels=[1, 2, 3])
    # Class-wise recall: diagonal / row sums
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'overall_accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'class_recalls': {name: r for name, r in zip(class_names, class_recalls)},
        'classification_report': classification_report(y_test, preds, target_names=class_names)
    }
    
    return metrics, preds

def save_model(model, scaler, tertiles, output_path="outputs/models/knn_model.joblib"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'tertiles': tertiles
    }
    joblib.dump(bundle, output_path)
    print(f"Trained model bundle saved to: {output_path}")
