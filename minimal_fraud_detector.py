"""
Minimal Quantum-Classical Fraud Detection Demo
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import pennylane as qml
import pickle
import os

def create_demo_data(n_samples=200):
    """Create demo fraud data"""
    np.random.seed(42)
    
    # Generate features
    amounts = np.random.lognormal(6, 1.5, n_samples)
    hours = np.random.randint(0, 24, n_samples)
    weekends = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Create fraud labels with clear patterns
    fraud_prob = (
        0.05 +  # Base rate
        0.4 * (amounts > np.percentile(amounts, 85)) +  # High amounts
        0.3 * (hours < 6) +  # Late night
        0.2 * weekends  # Weekend
    )
    
    fraud_labels = np.random.binomial(1, np.clip(fraud_prob, 0, 0.8), n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'amount': amounts,
        'hour_of_day': hours,
        'is_weekend': weekends,
        'fraud_flag': fraud_labels
    })
    
    print(f"Created {n_samples} samples with {fraud_labels.mean():.1%} fraud rate")
    return data

def quantum_kernel_simple(X1, X2=None):
    """Simple quantum kernel using PennyLane"""
    if X2 is None:
        X2 = X1
    
    # Simple quantum device
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def circuit(x):
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            # Use first 2 features for quantum circuit
            x1 = X1[i][:2]
            x2 = X2[j][:2]
            
            state1 = circuit(x1)
            state2 = circuit(x2)
            
            # Fidelity as kernel
            K[i, j] = abs(np.vdot(state1, state2))**2
    
    return K

def train_fraud_models():
    """Train quantum and classical fraud detection models"""
    print("=== Minimal Fraud Detection Training ===\n")
    
    # Create or load data
    try:
        df = pd.read_csv('data/upi_transactions_2024.csv')
        # Sample for demo
        df = df.sample(n=200, random_state=42)
        
        # Use available columns
        X = df[['amount (INR)', 'hour_of_day', 'is_weekend']].fillna(0)
        y = df['fraud_flag']
        
        print(f"Loaded real data: {len(df)} samples, fraud rate: {y.mean():.1%}")
        
    except:
        print("Using synthetic data...")
        df = create_demo_data(200)
        X = df[['amount', 'hour_of_day', 'is_weekend']]
        y = df['fraud_flag']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to quantum angles (first 2 features)
    angles_train = (X_train_scaled[:, :2] / 3.0) * np.pi
    angles_test = (X_test_scaled[:, :2] / 3.0) * np.pi
    
    print("Training models...")
    
    # Train quantum SVM
    print("  Quantum SVM...")
    K_train = quantum_kernel_simple(angles_train)
    K_test = quantum_kernel_simple(angles_test, angles_train)
    
    quantum_svm = SVC(kernel='precomputed', probability=True, random_state=42)
    quantum_svm.fit(K_train, y_train)
    
    # Train classical XGBoost
    print("  XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=20, max_depth=3, random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Get predictions
    quantum_pred = quantum_svm.predict_proba(K_test)[:, 1]
    classical_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Simple fusion
    fusion_pred = (quantum_pred + classical_pred) / 2
    
    # Evaluate
    print("\n=== RESULTS ===")
    try:
        print(f"Quantum AUC: {roc_auc_score(y_test, quantum_pred):.3f}")
        print(f"Classical AUC: {roc_auc_score(y_test, classical_pred):.3f}")
        print(f"Fusion AUC: {roc_auc_score(y_test, fusion_pred):.3f}")
    except:
        print("AUC calculation failed (likely single class in test set)")
        print(f"Test set fraud rate: {y_test.mean():.1%}")
    
    # Save models (without quantum circuit)
    models = {
        'scaler': scaler,
        'quantum_svm': quantum_svm,
        'xgb_model': xgb_model,
        'angles_train': angles_train
    }
    
    os.makedirs('demo_models', exist_ok=True)
    with open('demo_models/fraud_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Models saved to demo_models/fraud_models.pkl")
    
    # Test prediction
    print("\n=== Test Prediction ===")
    test_case = np.array([[15000, 2, 1]])  # High amount, late night, weekend
    test_scaled = scaler.transform(test_case)
    test_angles = (test_scaled[:, :2] / 3.0) * np.pi
    
    # Quantum prediction
    K_test_single = quantum_kernel_simple(test_angles, angles_train)
    q_score = quantum_svm.predict_proba(K_test_single)[0, 1] * 100
    
    # Classical prediction
    c_score = xgb_model.predict_proba(test_scaled)[0, 1] * 100
    
    # Fusion
    f_score = (q_score + c_score) / 2
    
    print(f"Test transaction [15000 INR, 2 AM, Weekend]:")
    print(f"  Quantum Score: {q_score:.1f}%")
    print(f"  Classical Score: {c_score:.1f}%")
    print(f"  Fusion Score: {f_score:.1f}%")
    
    print("\n✅ Training completed successfully!")
    return models

if __name__ == "__main__":
    train_fraud_models()