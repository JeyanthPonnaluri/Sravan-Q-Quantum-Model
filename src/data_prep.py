"""
Data preprocessing module for fraud detection.
Handles loading, feature engineering, encoding, and scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class FraudDataPreprocessor:
    def __init__(self, feature_columns=None):
        """
        Initialize preprocessor with configurable feature set.
        
        Args:
            feature_columns: List of column names to use as features.
                           If None, uses default set.
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = feature_columns or [
            'amount (INR)', 'sender_age_group', 'merchant_category', 
            'hour_of_day', 'is_weekend', 'device_type'
        ]
        self.numeric_features = ['amount (INR)', 'hour_of_day']
        self.categorical_features = [
            col for col in self.feature_columns 
            if col not in self.numeric_features
        ]
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data with fallback to synthetic generation."""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Generating synthetic data...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic fraud detection dataset."""
        np.random.seed(42)
        
        data = {
            'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
            'amount': np.random.lognormal(mean=6, sigma=1.5, size=n_samples),
            'sender_age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], n_samples),
            'merchant_category': np.random.choice(['grocery', 'fuel', 'restaurant', 'online', 'atm'], n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'device_type': np.random.choice(['mobile', 'web', 'atm'], n_samples, p=[0.7, 0.2, 0.1]),
        }
        
        # Create fraud labels with realistic patterns
        fraud_prob = (
            0.02 +  # base rate
            0.05 * (data['amount'] > np.percentile(data['amount'], 95)) +  # high amounts
            0.03 * (np.array(data['hour_of_day']) < 6) +  # late night
            0.02 * (np.array(data['device_type']) == 'atm')  # ATM transactions
        )
        
        data['fraud_flag'] = np.random.binomial(1, fraud_prob, n_samples)
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} synthetic records")
        return df
    
    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features: handle missing values, encode categoricals, scale numerics.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders/scalers (True for training data)
            
        Returns:
            Tuple of (processed_features, quantum_angles)
        """
        # Select and copy features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
        
        # Encode categorical features
        encoded_features = []
        
        for col in self.categorical_features:
            if col in X.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_categories = set(self.label_encoders[col].classes_)
                        X_col_mapped = X[col].astype(str).apply(
                            lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                        )
                        encoded = self.label_encoders[col].transform(X_col_mapped)
                    else:
                        encoded = np.zeros(len(X))
                
                encoded_features.append(encoded.reshape(-1, 1))
        
        # Add numeric features
        numeric_data = []
        for col in self.numeric_features:
            if col in X.columns:
                numeric_data.append(X[col].values.reshape(-1, 1))
        
        # Combine all features
        if encoded_features and numeric_data:
            all_features = np.hstack(encoded_features + numeric_data)
        elif encoded_features:
            all_features = np.hstack(encoded_features)
        elif numeric_data:
            all_features = np.hstack(numeric_data)
        else:
            raise ValueError("No valid features found")
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(all_features)
        else:
            X_scaled = self.scaler.transform(all_features)
        
        # Clip to [-3, 3] range
        X_scaled = np.clip(X_scaled, -3, 3)
        
        # Convert to quantum angles: angle = (scaled/3) * π
        angles = (X_scaled / 3.0) * np.pi
        
        return X_scaled, angles
    
    def time_ordered_split(self, df: pd.DataFrame, 
                          train_size: float = 0.6, 
                          cal_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-ordered stratified split to avoid data leakage.
        
        Args:
            df: Input dataframe (should have timestamp or be pre-sorted)
            train_size: Fraction for training set
            cal_size: Fraction for calibration set (remainder goes to test)
            
        Returns:
            Tuple of (train_df, cal_df, test_df)
        """
        n_samples = len(df)
        train_end = int(n_samples * train_size)
        cal_end = int(n_samples * (train_size + cal_size))
        
        # Sort by index to maintain time order (assuming data is chronologically ordered)
        df_sorted = df.sort_index()
        
        train_df = df_sorted.iloc[:train_end]
        cal_df = df_sorted.iloc[train_end:cal_end]
        test_df = df_sorted.iloc[cal_end:]
        
        print(f"Split sizes - Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}")
        print(f"Fraud rates - Train: {train_df['fraud_flag'].mean():.3f}, "
              f"Cal: {cal_df['fraud_flag'].mean():.3f}, Test: {test_df['fraud_flag'].mean():.3f}")
        
        return train_df, cal_df, test_df


def load_and_preprocess_data(filepath: str, feature_columns=None):
    """Convenience function to load and preprocess data."""
    preprocessor = FraudDataPreprocessor(feature_columns)
    df = preprocessor.load_data(filepath)
    
    # Create splits
    train_df, cal_df, test_df = preprocessor.time_ordered_split(df)
    
    # Preprocess features
    X_train, angles_train = preprocessor.preprocess_features(train_df, fit=True)
    X_cal, angles_cal = preprocessor.preprocess_features(cal_df, fit=False)
    X_test, angles_test = preprocessor.preprocess_features(test_df, fit=False)
    
    # Extract labels
    y_train = train_df['fraud_flag'].values
    y_cal = cal_df['fraud_flag'].values
    y_test = test_df['fraud_flag'].values
    
    return {
        'preprocessor': preprocessor,
        'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test,
        'angles_train': angles_train, 'angles_cal': angles_cal, 'angles_test': angles_test,
        'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test,
        'train_df': train_df, 'cal_df': cal_df, 'test_df': test_df
    }


if __name__ == "__main__":
    # Test the preprocessing
    data = load_and_preprocess_data("data/your_dataset.csv")
    print("Preprocessing completed successfully!")
    print(f"Feature shape: {data['X_train'].shape}")
    print(f"Angles shape: {data['angles_train'].shape}")