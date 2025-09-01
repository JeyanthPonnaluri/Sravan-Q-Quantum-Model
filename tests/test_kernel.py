"""
Unit tests for quantum kernel functionality.
"""
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qkernel import QuantumKernel


class TestQuantumKernel:
    """Test suite for QuantumKernel class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = 4
        self.qkernel = QuantumKernel(n_qubits=self.n_qubits)
        
        # Create test data
        self.n_samples = 5
        self.n_features = 4
        np.random.seed(42)
        self.angles_test = np.random.uniform(0, 2*np.pi, (self.n_samples, self.n_features))
    
    def test_kernel_matrix_shape(self):
        """Test that kernel matrix has correct shape."""
        K = self.qkernel.kernel_matrix(self.angles_test)
        
        assert K.shape == (self.n_samples, self.n_samples), \
            f"Expected shape {(self.n_samples, self.n_samples)}, got {K.shape}"
    
    def test_kernel_matrix_values_range(self):
        """Test that kernel values are in [0, 1] range."""
        K = self.qkernel.kernel_matrix(self.angles_test)
        
        assert np.all(K >= 0), "Kernel values should be >= 0"
        assert np.all(K <= 1), "Kernel values should be <= 1"
    
    def test_kernel_matrix_diagonal(self):
        """Test that diagonal elements are approximately 1."""
        K = self.qkernel.kernel_matrix(self.angles_test)
        diagonal = np.diag(K)
        
        # Diagonal should be close to 1 (self-similarity)
        assert np.allclose(diagonal, 1.0, atol=1e-10), \
            f"Diagonal elements should be ~1, got {diagonal}"
    
    def test_kernel_matrix_symmetry(self):
        """Test that kernel matrix is symmetric."""
        K = self.qkernel.kernel_matrix(self.angles_test)
        
        assert np.allclose(K, K.T, atol=1e-10), \
            "Kernel matrix should be symmetric"
    
    def test_kernel_matrix_max_value(self):
        """Test that maximum kernel value is approximately 1."""
        K = self.qkernel.kernel_matrix(self.angles_test)
        max_value = K.max()
        
        assert np.isclose(max_value, 1.0, atol=1e-10), \
            f"Maximum kernel value should be ~1, got {max_value}"
    
    def test_kernel_matrix_different_sets(self):
        """Test kernel matrix computation between different sets."""
        angles_A = self.angles_test[:3]
        angles_B = self.angles_test[2:]
        
        K = self.qkernel.kernel_matrix(angles_A, angles_B)
        
        expected_shape = (3, 3)  # 3 samples in A, 3 samples in B
        assert K.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {K.shape}"
        
        assert np.all(K >= 0) and np.all(K <= 1), \
            "Kernel values should be in [0, 1]"
    
    def test_single_fidelity_computation(self):
        """Test single fidelity computation."""
        angle_a = self.angles_test[0]
        angle_b = self.angles_test[1]
        
        fidelity = self.qkernel.compute_fidelity(angle_a, angle_b)
        
        assert isinstance(fidelity, float), "Fidelity should be a float"
        assert 0 <= fidelity <= 1, f"Fidelity should be in [0, 1], got {fidelity}"
    
    def test_self_fidelity(self):
        """Test that self-fidelity is 1."""
        angle = self.angles_test[0]
        fidelity = self.qkernel.compute_fidelity(angle, angle)
        
        assert np.isclose(fidelity, 1.0, atol=1e-10), \
            f"Self-fidelity should be 1, got {fidelity}"
    
    def test_feature_map_output(self):
        """Test that feature map returns valid quantum state."""
        angle = self.angles_test[0]
        state = self.qkernel.feature_map(angle)
        
        # Check state is normalized
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10), \
            f"Quantum state should be normalized, got norm {norm}"
        
        # Check state has correct dimension (2^n_qubits)
        expected_dim = 2**self.n_qubits
        assert len(state) == expected_dim, \
            f"State should have dimension {expected_dim}, got {len(state)}"
    
    def test_angle_padding_and_truncation(self):
        """Test handling of different angle array sizes."""
        # Test with fewer angles than qubits
        short_angles = np.array([0.5, 1.0])  # Only 2 angles for 4 qubits
        state_short = self.qkernel.feature_map(short_angles)
        assert len(state_short) == 2**self.n_qubits
        
        # Test with more angles than qubits
        long_angles = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # 6 angles for 4 qubits
        state_long = self.qkernel.feature_map(long_angles)
        assert len(state_long) == 2**self.n_qubits
    
    def test_jitter_functionality(self):
        """Test jitter addition for uncertainty estimation."""
        angles = self.angles_test[0]
        jitter_std = 0.001
        
        jittered = self.qkernel.add_jitter(angles, jitter_std)
        
        # Should have same shape
        assert jittered.shape == angles.shape
        
        # Should be different (with high probability)
        assert not np.allclose(angles, jittered), \
            "Jittered angles should be different from original"
        
        # Difference should be small
        diff = np.abs(angles - jittered)
        assert np.all(diff < 0.1), "Jitter should be small"
    
    def test_efficient_kernel_matrix(self):
        """Test efficient kernel matrix computation gives same results."""
        K_regular = self.qkernel.kernel_matrix(self.angles_test)
        K_efficient = self.qkernel.kernel_matrix_efficient(self.angles_test)
        
        assert np.allclose(K_regular, K_efficient, atol=1e-10), \
            "Efficient and regular kernel computation should give same results"
    
    def test_different_qubit_numbers(self):
        """Test kernel with different numbers of qubits."""
        for n_qubits in [2, 3, 5]:  # Test different qubit counts
            qkernel = QuantumKernel(n_qubits=n_qubits)
            
            # Create appropriate test data
            angles = np.random.uniform(0, 2*np.pi, (3, n_qubits))
            K = qkernel.kernel_matrix(angles)
            
            assert K.shape == (3, 3)
            assert np.all(K >= 0) and np.all(K <= 1)
            assert np.allclose(np.diag(K), 1.0, atol=1e-10)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])