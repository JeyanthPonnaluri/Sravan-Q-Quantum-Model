"""
Enhanced Quantum-Classical Fraud Detection with All Features
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
import pickle
import os

def load_and_prepare_data(file_path, sample_size=300):
    """Load and prepare fraud detection data with all features"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from dataset")
        
        # Sample for demo
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data...")
        df = create_synthetic_data(sample_size)
    
    # Select ALL available features for fraud detection
    feature_columns = [
        'amount (INR)', 'hour_of_day', 'is_weekend', 'day_of_week',
        'sender_age_group', 'receiver_age_group', 'sender_state', 
        'sender_bank', 'receiver_bank', 'merchant_category', 
        'device_type', 'transaction type', 'network_type', 'transaction_status'
    ]
    
    # Handle missing columns and create encoders
    encoders = {}
    processed_features = []
    feature_names = []
    
    # Define numeric and categorical features
    numeric_features = ['amount (INR)', 'hour_of_day', 'is_weekend']
    
    for col in feature_columns:
        if col in df.columns:
            if col in numeric_features:
                # Numeric features
                processed_features.append(df[col].fillna(0).values.reshape(-1, 1))
                feature_names.append(col)
            else:
                # Categorical features - encode them
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna('unknown').astype(str))
                encoders[col] = le
                processed_features.append(encoded.reshape(-1, 1))
                feature_names.append(col)
    
    # Combine all features
    X = np.hstack(processed_features)
    y = df['fraud_flag'].values
    
    print(f"Using {len(feature_names)} features: {feature_names}")
    print(f"Dataset: {len(df)} samples, fraud rate: {y.mean():.1%}")
    
    return X, y, df, encoders, feature_names

def create_synthetic_data(n_samples=300):
    """Create synthetic fraud data with all features"""
    np.random.seed(42)
    
    data = {
        'amount (INR)': np.random.lognormal(6, 1.5, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples),
        'sender_age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55'], n_samples),
        'receiver_age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55'], n_samples),
        'sender_state': np.random.choice(['Delhi', 'Maharashtra', 'Karnataka', 'Uttar Pradesh', 'Tamil Nadu'], n_samples),
        'sender_bank': np.random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB', 'Kotak'], n_samples),
        'receiver_bank': np.random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB', 'Kotak'], n_samples),
        'merchant_category': np.random.choice(['Grocery', 'Fuel', 'Entertainment', 'Food', 'Shopping'], n_samples),
        'device_type': np.random.choice(['Android', 'iOS'], n_samples, p=[0.7, 0.3]),
        'transaction type': np.random.choice(['P2P', 'P2M'], n_samples, p=[0.6, 0.4]),
        'network_type': np.random.choice(['4G', '5G', 'WiFi', '3G'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'transaction_status': np.random.choice(['SUCCESS', 'FAILED'], n_samples, p=[0.95, 0.05])
    }
    
    # Create fraud labels with realistic patterns
    fraud_prob = (
        0.05 +  # Base rate
        0.3 * (data['amount (INR)'] > np.percentile(data['amount (INR)'], 85)) +  # High amounts
        0.25 * (np.array(data['hour_of_day']) < 6) +  # Late night
        0.15 * np.array(data['is_weekend']) +  # Weekend
        0.1 * (np.array(data['sender_age_group']) == '18-25') +  # Young users
        0.1 * (np.array(data['device_type']) == 'iOS')  # iOS devices (example pattern)
    )
    
    data['fraud_flag'] = np.random.binomial(1, np.clip(fraud_prob, 0, 0.7), n_samples)
    
    return pd.DataFrame(data)

def quantum_kernel_enhanced(X1, X2=None, n_qubits=4):
    """Enhanced quantum kernel using more qubits"""
    if X2 is None:
        X2 = X1
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(x):
        # Use up to n_qubits features
        for i in range(min(len(x), n_qubits)):
            qml.RY(x[i], wires=i)
        
        # Entangling layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        return qml.state()
    
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            # Use first n_qubits features for quantum circuit
            x1 = X1[i][:n_qubits]
            x2 = X2[j][:n_qubits]
            
            state1 = circuit(x1)
            state2 = circuit(x2)
            
            # Fidelity as kernel
            K[i, j] = abs(np.vdot(state1, state2))**2
    
    return K

def train_enhanced_models():
    """Train enhanced quantum and classical fraud detection models"""
    print("=== Enhanced Quantum-Classical Fraud Detection Training ===\n")
    
    # Load data with all features
    X, y, df, encoders, feature_names = load_and_prepare_data('data/upi_transactions_2024.csv', sample_size=250)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to quantum angles (use first 4 features for quantum)
    n_qubits = 4
    angles_train = (X_train_scaled[:, :n_qubits] / 3.0) * np.pi
    angles_test = (X_test_scaled[:, :n_qubits] / 3.0) * np.pi
    
    print("Training models...")
    
    # Train quantum SVM
    print("  Quantum SVM (4 qubits)...")
    K_train = quantum_kernel_enhanced(angles_train, n_qubits=n_qubits)
    K_test = quantum_kernel_enhanced(angles_test, angles_train, n_qubits=n_qubits)
    
    quantum_svm = SVC(kernel='precomputed', probability=True, random_state=42, class_weight='balanced')
    quantum_svm.fit(K_train, y_train)
    
    # Train classical XGBoost with all features
    print("  XGBoost (all features)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, random_state=42, 
        eval_metric='logloss', scale_pos_weight=10  # Handle class imbalance
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Train fusion model
    print("  Fusion model...")
    quantum_pred_train = quantum_svm.predict_proba(K_train)[:, 1]
    classical_pred_train = xgb_model.predict_proba(X_train_scaled)[:, 1]
    
    # Simple rule-based logical score
    logical_scores_train = compute_logical_scores(X_train, feature_names)
    
    # Fusion features
    fusion_features_train = np.column_stack([
        quantum_pred_train, classical_pred_train, logical_scores_train
    ])
    
    fusion_model = LogisticRegression(random_state=42, class_weight='balanced')
    fusion_model.fit(fusion_features_train, y_train)
    
    # Get test predictions
    quantum_pred = quantum_svm.predict_proba(K_test)[:, 1]
    classical_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
    logical_scores_test = compute_logical_scores(X_test, feature_names)
    
    fusion_features_test = np.column_stack([
        quantum_pred, classical_pred, logical_scores_test
    ])
    fusion_pred = fusion_model.predict_proba(fusion_features_test)[:, 1]
    
    # Evaluate
    print("\n=== RESULTS ===")
    try:
        print(f"Quantum AUC: {roc_auc_score(y_test, quantum_pred):.3f}")
        print(f"Classical AUC: {roc_auc_score(y_test, classical_pred):.3f}")
        print(f"Fusion AUC: {roc_auc_score(y_test, fusion_pred):.3f}")
    except:
        print("AUC calculation failed (likely single class in test set)")
        print(f"Test set fraud rate: {y_test.mean():.1%}")
        print(f"Prediction ranges - Q: [{quantum_pred.min():.3f}, {quantum_pred.max():.3f}]")
        print(f"                    C: [{classical_pred.min():.3f}, {classical_pred.max():.3f}]")
        print(f"                    F: [{fusion_pred.min():.3f}, {fusion_pred.max():.3f}]")
    
    # Save models
    models = {
        'scaler': scaler,
        'quantum_svm': quantum_svm,
        'xgb_model': xgb_model,
        'fusion_model': fusion_model,
        'angles_train': angles_train,
        'encoders': encoders,
        'feature_names': feature_names,
        'n_qubits': n_qubits
    }
    
    os.makedirs('enhanced_models', exist_ok=True)
    with open('enhanced_models/fraud_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Models saved to enhanced_models/fraud_models.pkl")
    
    # Test prediction with all features
    print("\n=== Test Prediction ===")
    test_case = create_test_transaction()
    result = predict_with_all_features(test_case, models)
    
    print(f"Test transaction: {test_case}")
    print(f"  Quantum Score: {result['quantum_score']:.1f}%")
    print(f"  Classical Score: {result['classical_score']:.1f}%")
    print(f"  Logical Score: {result['logical_score']:.1f}%")
    print(f"  Fusion Score: {result['fusion_score']:.1f}%")
    
    print("\n✅ Enhanced training completed successfully!")
    return models

def compute_logical_scores(X, feature_names):
    """Compute rule-based logical scores"""
    scores = np.zeros(len(X))
    
    # Find feature indices
    feature_idx = {name: i for i, name in enumerate(feature_names)}
    
    # Rule 1: High amounts
    if 'amount (INR)' in feature_idx:
        amount_idx = feature_idx['amount (INR)']
        high_amounts = X[:, amount_idx] > np.percentile(X[:, amount_idx], 90)
        scores += 0.3 * high_amounts
    
    # Rule 2: Late night transactions
    if 'hour_of_day' in feature_idx:
        hour_idx = feature_idx['hour_of_day']
        late_night = (X[:, hour_idx] < 6) | (X[:, hour_idx] > 22)
        scores += 0.2 * late_night
    
    # Rule 3: Weekend transactions
    if 'is_weekend' in feature_idx:
        weekend_idx = feature_idx['is_weekend']
        scores += 0.15 * X[:, weekend_idx]
    
    # Rule 4: Young age group (encoded as 0 typically)
    if 'sender_age_group' in feature_idx:
        age_idx = feature_idx['sender_age_group']
        young_users = X[:, age_idx] == 0  # Assuming 18-25 is encoded as 0
        scores += 0.1 * young_users
    
    return np.clip(scores, 0, 1)

def create_test_transaction():
    """Create a test transaction with all features"""
    return {
        'amount': 25000,
        'hour_of_day': 3,
        'is_weekend': 1,
        'day_of_week': 'Saturday',
        'sender_age_group': '18-25',
        'receiver_age_group': '26-35',
        'sender_state': 'Delhi',
        'sender_bank': 'HDFC',
        'receiver_bank': 'SBI',
        'merchant_category': 'Entertainment',
        'device_type': 'Android',
        'transaction_type': 'P2P',
        'network_type': '4G',
        'transaction_status': 'SUCCESS'
    }

def predict_with_all_features(transaction, models):
    """Predict fraud using all features"""
    # Prepare input vector
    X_input = []
    
    # Numeric features
    X_input.extend([
        transaction.get('amount', 0),
        transaction.get('hour_of_day', 12),
        transaction.get('is_weekend', 0)
    ])
    
    # Categorical features (all non-numeric features)
    categorical_features = [
        'day_of_week', 'sender_age_group', 'receiver_age_group', 'sender_state',
        'sender_bank', 'receiver_bank', 'merchant_category', 'device_type',
        'transaction_type', 'network_type', 'transaction_status'
    ]
    
    for feature in categorical_features:
        if feature in models['encoders']:
            value = transaction.get(feature, 'unknown')
            encoder = models['encoders'][feature]
            
            # Handle unseen categories
            if value in encoder.classes_:
                encoded = encoder.transform([value])[0]
            else:
                encoded = 0  # Default for unseen categories
            
            X_input.append(encoded)
        else:
            X_input.append(0)
    
    X_input = np.array(X_input).reshape(1, -1)
    
    # Scale features
    X_scaled = models['scaler'].transform(X_input)
    
    # Quantum prediction
    angles_input = (X_scaled[:, :models['n_qubits']] / 3.0) * np.pi
    K_input = quantum_kernel_enhanced(angles_input, models['angles_train'], models['n_qubits'])
    quantum_score = models['quantum_svm'].predict_proba(K_input)[0, 1] * 100
    
    # Classical prediction
    classical_score = models['xgb_model'].predict_proba(X_scaled)[0, 1] * 100
    
    # Logical score
    logical_score = compute_logical_scores(X_input, models['feature_names'])[0] * 100
    
    # Fusion prediction
    fusion_features = np.array([[quantum_score/100, classical_score/100, logical_score/100]])
    fusion_score = models['fusion_model'].predict_proba(fusion_features)[0, 1] * 100
    
    return {
        'quantum_score': quantum_score,
        'classical_score': classical_score,
        'logical_score': logical_score,
        'fusion_score': fusion_score
    }

if __name__ == "__main__":
    train_enhanced_models()