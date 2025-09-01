"""
Integration tests for FastAPI fraud detection API.
"""
import pytest
import json
import sys
import os
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the model loading to avoid dependency on trained models
import unittest.mock as mock


class TestFraudDetectionAPI:
    """Test suite for fraud detection API."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client with mocked models."""
        
        # Mock the model loading
        with mock.patch('api_main.load_all_models') as mock_load:
            # Import after patching
            from api_main import app
            
            # Setup mock models and data
            mock_preprocessor = mock.MagicMock()
            mock_preprocessor.feature_columns = ['amount', 'sender_age_group', 'merchant_category', 
                                               'hour_of_day', 'is_weekend', 'device_type']
            mock_preprocessor.preprocess_features.return_value = (
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],  # X_processed
                [[0.1, 0.2, 0.3, 0.4]]  # angles
            )
            
            mock_quantum_model = mock.MagicMock()
            mock_quantum_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% fraud probability
            
            mock_xgb_model = mock.MagicMock()
            mock_xgb_model.predict_proba.return_value = [[0.4, 0.6]]  # 60% fraud probability
            
            mock_fusion_model = mock.MagicMock()
            mock_fusion_model.predict_proba.return_value = [[0.35, 0.65]]  # 65% fraud probability
            
            mock_qkernel = mock.MagicMock()
            mock_qkernel.kernel_matrix.return_value = [[0.8, 0.6], [0.6, 0.9]]
            mock_qkernel.compute_uncertainty.return_value = (0.65, 0.05)  # mean, std
            mock_qkernel.n_qubits = 4
            
            # Set up the mocked global variables
            with mock.patch.dict('api_main.models', {
                'preprocessor': mock_preprocessor,
                'quantum_model': mock_quantum_model,
                'xgboost_model': mock_xgb_model,
                'fusion_model': mock_fusion_model,
                'angles_train': [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
            }):
                with mock.patch('api_main.preprocessor', mock_preprocessor):
                    with mock.patch('api_main.qkernel', mock_qkernel):
                        with mock.patch('api_main.config', {
                            'uncertainty_enabled': True,
                            'jitter_repeats': 6,
                            'jitter_std': 0.001
                        }):
                            yield TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Quantum-Classical Fraud Detection" in response.text
    
    def test_form_endpoint(self, client):
        """Test form endpoint returns HTML form."""
        response = client.get("/form")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "quantum_kernel_ready" in data
        assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model_info")
        assert response.status_code == 200
        
        data = response.json()
        assert "loaded_models" in data
        assert "feature_columns" in data
        assert "n_qubits" in data
        assert "config" in data
        
        # Check expected feature columns
        expected_features = ['amount', 'sender_age_group', 'merchant_category', 
                           'hour_of_day', 'is_weekend', 'device_type']
        assert data["feature_columns"] == expected_features
        assert data["n_qubits"] == 4
    
    def test_predict_endpoint_valid_input(self, client):
        """Test prediction endpoint with valid input."""
        test_transaction = {
            "amount": 5000.0,
            "sender_age_group": "26-35",
            "merchant_category": "grocery",
            "hour_of_day": 14,
            "is_weekend": 0,
            "device_type": "mobile"
        }
        
        response = client.post("/predict", json=test_transaction)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check all required keys are present
        required_keys = ["quantum_score", "classical_score", "logical_score", 
                        "practical_score", "uncertainty"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Check value ranges (0-100 for scores)
        for score_key in ["quantum_score", "classical_score", "logical_score", "practical_score"]:
            score = data[score_key]
            assert isinstance(score, (int, float)), f"{score_key} should be numeric"
            assert 0 <= score <= 100, f"{score_key} should be in [0, 100], got {score}"
        
        # Check uncertainty
        uncertainty = data["uncertainty"]
        assert isinstance(uncertainty, (int, float)), "Uncertainty should be numeric"
        assert uncertainty >= 0, f"Uncertainty should be >= 0, got {uncertainty}"
    
    def test_predict_endpoint_with_logical_score(self, client):
        """Test prediction endpoint with logical score provided."""
        test_transaction = {
            "amount": 10000.0,
            "sender_age_group": "18-25",
            "merchant_category": "atm",
            "hour_of_day": 2,
            "is_weekend": 1,
            "device_type": "atm",
            "p_logic": 0.8
        }
        
        response = client.post("/predict", json=test_transaction)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should have logical score close to provided value (80%)
        assert abs(data["logical_score"] - 80.0) < 0.1
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_transaction = {
            "amount": 5000.0,
            "sender_age_group": "26-35"
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_values(self, client):
        """Test prediction endpoint with invalid field values."""
        invalid_transaction = {
            "amount": -1000.0,  # Negative amount
            "sender_age_group": "invalid_age",
            "merchant_category": "grocery",
            "hour_of_day": 25,  # Invalid hour
            "is_weekend": 2,  # Invalid boolean
            "device_type": "mobile"
        }
        
        response = client.post("/predict", json=invalid_transaction)
        # Should either return 422 (validation error) or handle gracefully
        assert response.status_code in [200, 422]
    
    def test_predict_endpoint_edge_cases(self, client):
        """Test prediction endpoint with edge case values."""
        edge_cases = [
            {
                "amount": 0.01,  # Very small amount
                "sender_age_group": "18-25",
                "merchant_category": "grocery",
                "hour_of_day": 0,
                "is_weekend": 0,
                "device_type": "mobile"
            },
            {
                "amount": 999999.99,  # Very large amount
                "sender_age_group": "50+",
                "merchant_category": "atm",
                "hour_of_day": 23,
                "is_weekend": 1,
                "device_type": "atm"
            }
        ]
        
        for transaction in edge_cases:
            response = client.post("/predict", json=transaction)
            assert response.status_code == 200
            
            data = response.json()
            # Verify all scores are valid
            for key in ["quantum_score", "classical_score", "logical_score", "practical_score"]:
                assert 0 <= data[key] <= 100
    
    def test_predict_endpoint_all_categories(self, client):
        """Test prediction endpoint with all possible categorical values."""
        categories = {
            "sender_age_group": ["18-25", "26-35", "36-50", "50+"],
            "merchant_category": ["grocery", "fuel", "restaurant", "online", "atm"],
            "device_type": ["mobile", "web", "atm"]
        }
        
        base_transaction = {
            "amount": 5000.0,
            "hour_of_day": 12,
            "is_weekend": 0
        }
        
        # Test each category combination (sample a few)
        for age in categories["sender_age_group"][:2]:  # Test first 2 to keep test fast
            for merchant in categories["merchant_category"][:2]:
                for device in categories["device_type"]:
                    transaction = base_transaction.copy()
                    transaction.update({
                        "sender_age_group": age,
                        "merchant_category": merchant,
                        "device_type": device
                    })
                    
                    response = client.post("/predict", json=transaction)
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert all(0 <= data[key] <= 100 for key in 
                             ["quantum_score", "classical_score", "logical_score", "practical_score"])
    
    def test_response_format_consistency(self, client):
        """Test that response format is consistent across requests."""
        test_transaction = {
            "amount": 5000.0,
            "sender_age_group": "26-35",
            "merchant_category": "grocery",
            "hour_of_day": 14,
            "is_weekend": 0,
            "device_type": "mobile"
        }
        
        # Make multiple requests
        responses = []
        for _ in range(3):
            response = client.post("/predict", json=test_transaction)
            assert response.status_code == 200
            responses.append(response.json())
        
        # Check that all responses have the same structure
        first_response = responses[0]
        for response in responses[1:]:
            assert set(response.keys()) == set(first_response.keys())
            
            # Values should be consistent (same input should give same output)
            for key in response.keys():
                if key != "uncertainty":  # Uncertainty might vary due to randomness
                    assert response[key] == first_response[key], \
                        f"Inconsistent {key}: {response[key]} vs {first_response[key]}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])