"""
Cost Analysis and Threshold Optimization for Fraud Detection.
Tunes classification thresholds to minimize merchant losses from False Positives (friction) and False Negatives (fraud).
"""
import numpy as np
from typing import Dict, Any, List, Tuple

class CostOptimizer:
    def __init__(self, default_friction_cost: float = 1000.0, default_fraud_loss: float = 5000.0):
        """
        Initialize the Cost Optimizer.
        
        Args:
            default_friction_cost: Financial loss associated with falsely flagging a good transaction (friction).
            default_fraud_loss: Financial loss associated with missing a fraudulent transaction (chargeback + fee).
        """
        self.friction_cost = default_friction_cost
        self.fraud_loss = default_fraud_loss

    def calculate_cost_matrix(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              friction_cost: float = None, fraud_loss: float = None) -> Dict[str, Any]:
        """
        Calculate total financial costs across a range of thresholds.
        
        Args:
            y_true: True labels (0 for clean, 1 for fraud)
            y_pred_proba: Predicted fraud probabilities [0, 1]
            friction_cost: Cost of a False Positive (defaults to self.friction_cost)
            fraud_loss: Cost of a False Negative (defaults to self.fraud_loss)
            
        Returns:
            Dictionary containing optimal threshold, metrics at optimal threshold, and cost curve values.
        """
        c_fp = friction_cost if friction_cost is not None else self.friction_cost
        c_fn = fraud_loss if fraud_loss is not None else self.fraud_loss
        
        thresholds = np.linspace(0.0, 1.0, 101)
        total_costs = []
        fp_costs = []
        fn_costs = []
        recalls = []
        precisions = []
        f1_scores = []
        
        n_samples = len(y_true)
        if n_samples == 0:
            return {
                "optimal_threshold": 0.5,
                "min_cost": 0.0,
                "baseline_cost": 0.0,
                "savings": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        # Calculate costs for baseline "Approve All" and "Block All"
        # Approve all: FP = 0, FN = sum(y_true)
        fn_approve_all = np.sum(y_true == 1)
        cost_approve_all = fn_approve_all * c_fn
        
        # Block all: FP = sum(y_true == 0), FN = 0
        fp_block_all = np.sum(y_true == 0)
        cost_block_all = fp_block_all * c_fp
        
        no_model_baseline_cost = min(cost_approve_all, cost_block_all)
        
        for th in thresholds:
            # Predict labels based on threshold
            y_pred = (y_pred_proba >= th).astype(int)
            
            # Confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            # Costs
            fp_cost = fp * c_fp
            fn_cost = fn * c_fn
            total_cost = fp_cost + fn_cost
            
            total_costs.append(float(total_cost))
            fp_costs.append(float(fp_cost))
            fn_costs.append(float(fn_cost))
            
            # Classic metrics
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)
            
        # Find index of minimum cost
        min_idx = np.argmin(total_costs)
        optimal_threshold = float(thresholds[min_idx])
        min_cost = total_costs[min_idx]
        
        # Baseline cost at standard 0.5 threshold
        idx_05 = 50  # index corresponding to threshold = 0.5
        baseline_cost = total_costs[idx_05]
        
        savings_vs_baseline = max(0.0, baseline_cost - min_cost)
        savings_vs_no_model = max(0.0, no_model_baseline_cost - min_cost)
        
        return {
            "optimal_threshold": optimal_threshold,
            "min_cost": min_cost,
            "baseline_cost": baseline_cost,
            "no_model_cost": no_model_baseline_cost,
            "savings_vs_baseline": savings_vs_baseline,
            "savings_vs_no_model": savings_vs_no_model,
            "precision": precisions[min_idx],
            "recall": recalls[min_idx],
            "f1": f1_scores[min_idx],
            "fp_count": int(np.sum((y_true == 0) & ((y_pred_proba >= optimal_threshold).astype(int) == 1))),
            "fn_count": int(np.sum((y_true == 1) & ((y_pred_proba >= optimal_threshold).astype(int) == 0))),
            "curve": {
                "thresholds": thresholds.tolist(),
                "total_costs": total_costs,
                "fp_costs": fp_costs,
                "fn_costs": fn_costs,
                "precisions": precisions,
                "recalls": recalls
            }
        }
        
    def find_action_for_score(self, score: float, optimal_threshold: float) -> Tuple[str, str]:
        """
        Determine recommended action and risk level dynamically based on the optimal threshold.
        
        Args:
            score: Unified fraud score (0-100)
            optimal_threshold: The cost-optimized threshold (0.0 to 1.0)
            
        Returns:
            Tuple of (risk_level, recommended_action)
        """
        # Convert threshold to percentage scale
        th_pct = optimal_threshold * 100
        
        # Define dynamic bands around optimal threshold
        if score >= max(th_pct, 75.0):
            return "CRITICAL", "BLOCK TRANSACTION - High confidence fraud detection"
        elif score >= th_pct:
            return "HIGH", "MANUAL REVIEW REQUIRED - Exceeds cost-optimized risk boundary"
        elif score >= th_pct * 0.7:
            return "MEDIUM", "ADDITIONAL VERIFICATION - Elevate security checks"
        elif score >= th_pct * 0.4:
            return "LOW", "MONITOR TRANSACTION - Elevated background checks"
        else:
            return "MINIMAL", "APPROVE - Low suspicion profile"
