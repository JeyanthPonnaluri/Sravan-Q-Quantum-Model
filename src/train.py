"""
End-to-end training script for quantum-classical fusion fraud detection.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve, brier_score_loss
)
import xgboost as xgb
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import load_and_preprocess_data
from qkernel import QuantumKernel
from save_load import save_models, save


def compute_metrics(y_true, y_pred_proba, y_pred=None):
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    # ROC-AUC
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Average Precision (AUPRC)
    metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
    
    # Brier Score (lower is better)
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    
    # Recall at FPR = 0.01
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    idx_001 = np.where(fpr <= 0.01)[0]
    if len(idx_001) > 0:
        metrics['recall_at_fpr_001'] = tpr[idx_001[-1]]
    else:
        metrics['recall_at_fpr_001'] = 0.0
    
    # Expected Calibration Error (ECE)
    metrics['ece'] = compute_ece(y_true, y_pred_proba)
    
    return metrics


def compute_ece(y_true, y_pred_proba, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def generate_logical_scores(df, method='simple_rules'):
    """
    Generate logical scores using simple rules or LLM (placeholder).
    
    Args:
        df: DataFrame with transaction features
        method: 'simple_rules' or 'llm_placeholder'
        
    Returns:
        Array of logical scores [0, 1]
    """
    if method == 'simple_rules':
        # Simple rule-based logical scoring
        scores = np.zeros(len(df))
        
        # High amount transactions
        if 'amount (INR)' in df.columns:
            high_amount = df['amount (INR)'] > df['amount (INR)'].quantile(0.95)
            scores += 0.3 * high_amount
        
        # Late night transactions
        if 'hour_of_day' in df.columns:
            late_night = (df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)
            scores += 0.2 * late_night
        
        # Weekend transactions
        if 'is_weekend' in df.columns:
            scores += 0.1 * df['is_weekend']
        
        # ATM transactions
        if 'device_type' in df.columns:
            atm_transactions = df['device_type'] == 'atm'
            scores += 0.15 * atm_transactions
        
        # Normalize to [0, 1]
        scores = np.clip(scores, 0, 1)
        
    elif method == 'llm_placeholder':
        # Placeholder for LLM-based scoring
        # In practice, this would call an LLM API with transaction context
        scores = np.random.beta(2, 8, len(df))  # Skewed towards low scores
    
    else:
        # Default: zeros (no logical component)
        scores = np.zeros(len(df))
    
    return scores


def train_models(data_path, feature_columns=None, n_qubits=4, 
                jitter_std=0.001, jitter_repeats=20, save_dir="models", max_samples=5000):
    """
    Main training function for quantum-classical fusion model.
    """
    print("=== Quantum-Classical Fusion Fraud Detection Training ===")
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data = load_and_preprocess_data(data_path, feature_columns)
    
    # Subsample for demonstration if dataset is too large
    if data['X_train'].shape[0] > max_samples:
        print(f"   Subsampling to {max_samples} training samples for demonstration...")
        # Stratified subsample to maintain fraud ratio
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(data['X_train']))
        train_indices, _ = train_test_split(
            indices, train_size=max_samples, stratify=data['y_train'], random_state=42
        )
        
        # Update training data
        data['X_train'] = data['X_train'][train_indices]
        data['angles_train'] = data['angles_train'][train_indices]
        data['y_train'] = data['y_train'][train_indices]
        data['train_df'] = data['train_df'].iloc[train_indices]
    
    X_train, X_cal, X_test = data['X_train'], data['X_cal'], data['X_test']
    angles_train, angles_cal, angles_test = data['angles_train'], data['angles_cal'], data['angles_test']
    y_train, y_cal, y_test = data['y_train'], data['y_cal'], data['y_test']
    train_df, cal_df, test_df = data['train_df'], data['cal_df'], data['test_df']
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Calibration set: {X_cal.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize quantum kernel
    print(f"\n2. Initializing quantum kernel with {n_qubits} qubits...")
    qkernel = QuantumKernel(n_qubits=n_qubits)
    
    # Compute quantum kernel matrices
    print("3. Computing quantum kernel matrices...")
    print("   Computing K_train...")
    K_train = qkernel.kernel_matrix_efficient(angles_train)
    print("   Computing K_cal...")
    K_cal = qkernel.kernel_matrix_efficient(angles_cal, angles_train)
    print("   Computing K_test...")
    K_test = qkernel.kernel_matrix_efficient(angles_test, angles_train)
    
    print(f"   Kernel matrices computed. Train: {K_train.shape}, Cal: {K_cal.shape}, Test: {K_test.shape}")
    
    # Train quantum SVM
    print("\n4. Training quantum SVM...")
    quantum_svm = SVC(kernel='precomputed', class_weight='balanced', probability=True, random_state=42)
    quantum_svm.fit(K_train, y_train)
    
    # Calibrate quantum model
    print("5. Calibrating quantum model...")
    quantum_model = CalibratedClassifierCV(
        quantum_svm, method='isotonic', cv='prefit'
    )
    quantum_model.fit(K_cal, y_cal)
    
    # Train classical XGBoost
    print("\n6. Training classical XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Generate logical scores
    print("\n7. Generating logical scores...")
    p_logic_cal = generate_logical_scores(cal_df, method='simple_rules')
    p_logic_test = generate_logical_scores(test_df, method='simple_rules')
    
    # Get predictions on calibration set for fusion training
    print("\n8. Getting calibration predictions...")
    p_quantum_cal = quantum_model.predict_proba(K_cal)[:, 1]
    p_classical_cal = xgb_model.predict_proba(X_cal)[:, 1]
    
    # Train fusion meta-model
    print("9. Training fusion meta-model...")
    fusion_features_cal = np.column_stack([p_quantum_cal, p_classical_cal, p_logic_cal])
    fusion_model = LogisticRegression(random_state=42, class_weight='balanced')
    fusion_model.fit(fusion_features_cal, y_cal)
    
    # Evaluate on test set
    print("\n10. Evaluating on test set...")
    
    # Get test predictions
    p_quantum_test = quantum_model.predict_proba(K_test)[:, 1]
    p_classical_test = xgb_model.predict_proba(X_test)[:, 1]
    
    # Fusion prediction
    fusion_features_test = np.column_stack([p_quantum_test, p_classical_test, p_logic_test])
    p_fusion_test = fusion_model.predict_proba(fusion_features_test)[:, 1]
    
    # Compute metrics for all models
    print("\n=== EVALUATION RESULTS ===")
    
    models_to_evaluate = {
        'Quantum': p_quantum_test,
        'Classical (XGBoost)': p_classical_test,
        'Fusion': p_fusion_test
    }
    
    for model_name, predictions in models_to_evaluate.items():
        metrics = compute_metrics(y_test, predictions)
        print(f"\n{model_name} Model:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  Recall@FPR=0.01: {metrics['recall_at_fpr_001']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
    
    # Compute uncertainty for quantum model
    print(f"\n11. Computing uncertainty estimates (n_repeats={jitter_repeats})...")
    uncertainties = []
    
    # Sample a few test examples for uncertainty computation
    n_uncertainty_samples = min(50, len(angles_test))
    uncertainty_indices = np.random.choice(len(angles_test), n_uncertainty_samples, replace=False)
    
    for idx in uncertainty_indices:
        mean_pred, std_pred = qkernel.compute_uncertainty(
            angles_test[idx], angles_train, quantum_model, 
            n_repeats=jitter_repeats, jitter_std=jitter_std
        )
        uncertainties.append(std_pred)
    
    avg_uncertainty = np.mean(uncertainties)
    print(f"Average quantum uncertainty: {avg_uncertainty:.4f}")
    
    # Save all models and artifacts
    print(f"\n12. Saving models to {save_dir}/...")
    os.makedirs(save_dir, exist_ok=True)
    
    models_to_save = {
        'preprocessor': data['preprocessor'],
        'quantum_model': quantum_model,
        'xgboost_model': xgb_model,
        'fusion_model': fusion_model,
        'angles_train': angles_train,
        'qkernel_config': {
            'n_qubits': n_qubits,
            'jitter_std': jitter_std,
            'jitter_repeats': jitter_repeats
        }
    }
    
    save_models(models_to_save, save_dir)
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    
    return {
        'models': models_to_save,
        'metrics': {name: compute_metrics(y_test, pred) for name, pred in models_to_evaluate.items()},
        'test_predictions': models_to_evaluate,
        'uncertainty': avg_uncertainty
    }


def main():
    parser = argparse.ArgumentParser(description='Train quantum-classical fusion fraud detection model')
    parser.add_argument('--data', default='upi_transactions_2024.csv', 
                       help='Path to dataset CSV file')
    parser.add_argument('--features', nargs='+', default=None,
                       help='Feature columns to use (default: auto-select)')
    parser.add_argument('--n_qubits', type=int, default=4,
                       help='Number of qubits for quantum kernel (max 5)')
    parser.add_argument('--jitter', type=float, default=0.001,
                       help='Jitter standard deviation for uncertainty')
    parser.add_argument('--jitter_repeats', type=int, default=20,
                       help='Number of jittered repeats for uncertainty')
    parser.add_argument('--save_dir', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum training samples for quantum kernel (default: 5000)')
    
    args = parser.parse_args()
    
    # Validate arguments
    args.n_qubits = min(args.n_qubits, 5)
    
    # Run training
    results = train_models(
        data_path=args.data,
        feature_columns=args.features,
        n_qubits=args.n_qubits,
        jitter_std=args.jitter,
        jitter_repeats=args.jitter_repeats,
        save_dir=args.save_dir,
        max_samples=args.max_samples
    )
    
    return results


if __name__ == "__main__":
    main()