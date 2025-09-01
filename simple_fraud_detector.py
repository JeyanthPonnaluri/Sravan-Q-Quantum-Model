"""
Simple Quantum-Classical Fraud Detection System
A streamlined implementation for demonstration purposes.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import pennylane as qml
import joblib
import os

# Simple quantum kernel implementation
class SimpleQuantumKernel:
    def __init__(self, n_qubits=3):
        self.n_qubits = n_qubits
        self.device = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(self.device)
        def quantum_circuit(x):
            """Simple quantum feature map"""
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            return qml.state()
        
        self.quantum_circuit = quantum_circuit
    
    def compute_kernel(self, X1, X2=None):
        """Compute quantum kernel matrix"""
        if X2 is None:
            X2 = X1
        
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                state1 = self.quantum_circuit(X1[i])
                state2 = self.quantum_circuit(X2[j])
                # Fidelity as kernel
                K[i, j] = abs(np.vdot(state1, state2))**2
        
        return K

def load_and_prepare_data(file_path, sample_size=1000):
    """Load and prepare fraud detection data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
    except:
        print("Creating synthetic data...")
        df = create_synthetic_data(sample_size * 2)
    
    # Sample for demo
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Select key features
    features = ['amount (INR)', 'hour_of_day', 'is_weekend']
    
    # Handle missing columns
    if 'amount (INR)' not in df.columns and 'amount' in df.columns:
        df['amount (INR)'] = df['amount']
    
    # Create features if missing
    if 'hour_of_day' not in df.columns:
        df['hour_of_day'] = np.random.randint(0, 24, len(df))
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = np.random.choice([0, 1], len(df))
    
    # Prepare features
    X = df[features].fillna(0)
    y = df['fraud_flag'] if 'fraud_flag' in df.columns else np.random.choice([0, 1], len(df), p=[0.95, 0.05])
    
    return X, y, df

def create_synthetic_data(n_samples=1000):
    """Create synthetic fraud data"""
    np.random.seed(42)
    
    data = {
        'amount (INR)': np.random.lognormal(6, 1.5, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'merchant_category': np.random.choice(['grocery', 'fuel', 'online'], n_samples)
    }
    
    # Create fraud labels with higher fraud rate for demo
    fraud_prob = (
        0.1 +  # Higher base rate for demo
        0.2 * (data['amount (INR)'] > np.percentile(data['amount (INR)'], 80)) +
        0.15 * (np.array(data['hour_of_day']) < 6) +
        0.1 * np.array(data['is_weekend'])
    )
    data['fraud_flag'] = np.random.binomial(1, np.clip(fraud_prob, 0, 1), n_samples)
    
    return pd.DataFrame(data)

def train_models(X, y):
    """Train quantum and classical models"""
    print("Training models...")
    
    # Split data
    # Check if we have enough samples for stratification
    if np.sum(y) >= 2 and np.sum(1-y) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to quantum angles
    angles_train = (X_train_scaled / 3.0) * np.pi
    angles_test = (X_test_scaled / 3.0) * np.pi
    
    # Train quantum model
    print("Training quantum SVM...")
    qkernel = SimpleQuantumKernel(n_qubits=3)
    K_train = qkernel.compute_kernel(angles_train)
    K_test = qkernel.compute_kernel(angles_test, angles_train)
    
    quantum_svm = SVC(kernel='precomputed', probability=True, random_state=42)
    quantum_svm.fit(K_train, y_train)
    
    # Train classical model
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Get predictions
    quantum_pred = quantum_svm.predict_proba(K_test)[:, 1]
    classical_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Simple fusion (average)
    fusion_pred = (quantum_pred + classical_pred) / 2
    
    # Evaluate
    print("\n=== RESULTS ===")
    print(f"Quantum AUC: {roc_auc_score(y_test, quantum_pred):.3f}")
    print(f"Classical AUC: {roc_auc_score(y_test, classical_pred):.3f}")
    print(f"Fusion AUC: {roc_auc_score(y_test, fusion_pred):.3f}")
    
    # Save models
    models = {
        'scaler': scaler,
        'quantum_svm': quantum_svm,
        'xgb_model': xgb_model,
        'qkernel': qkernel,
        'angles_train': angles_train
    }
    
    os.makedirs('simple_models', exist_ok=True)
    joblib.dump(models, 'simple_models/fraud_models.pkl')
    print("Models saved to simple_models/fraud_models.pkl")
    
    return models

def predict_fraud(transaction_data, models):
    """Predict fraud for a single transaction"""
    # Prepare input
    X_input = np.array([[
        transaction_data['amount'],
        transaction_data['hour_of_day'],
        transaction_data['is_weekend']
    ]])
    
    # Scale and convert to angles
    X_scaled = models['scaler'].transform(X_input)
    angles_input = (X_scaled / 3.0) * np.pi
    
    # Quantum prediction
    K_input = models['qkernel'].compute_kernel(angles_input, models['angles_train'])
    quantum_score = models['quantum_svm'].predict_proba(K_input)[0, 1] * 100
    
    # Classical prediction
    classical_score = models['xgb_model'].predict_proba(X_scaled)[0, 1] * 100
    
    # Fusion score
    fusion_score = (quantum_score + classical_score) / 2
    
    return {
        'quantum_score': round(quantum_score, 2),
        'classical_score': round(classical_score, 2),
        'fusion_score': round(fusion_score, 2)
    }

def main():
    """Main execution"""
    print("=== Simple Quantum-Classical Fraud Detection ===\n")
    
    # Load data
    X, y, df = load_and_prepare_data('data/upi_transactions_2024.csv', sample_size=500)
    print(f"Using {len(X)} samples with {len(X.columns)} features")
    print(f"Fraud rate: {y.mean():.1%}")
    
    # Train models
    models = train_models(X, y)
    
    # Test prediction
    print("\n=== Testing Prediction ===")
    test_transaction = {
        'amount': 15000,
        'hour_of_day': 2,
        'is_weekend': 1
    }
    
    result = predict_fraud(test_transaction, models)
    print(f"Test transaction: {test_transaction}")
    print(f"Quantum Score: {result['quantum_score']}%")
    print(f"Classical Score: {result['classical_score']}%")
    print(f"Fusion Score: {result['fusion_score']}%")
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()