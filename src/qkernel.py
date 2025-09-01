"""
Quantum kernel implementation using PennyLane.
Builds feature map circuits and computes kernel matrices.
"""
import pennylane as qml
import numpy as np
from typing import Callable, Tuple
import warnings
warnings.filterwarnings('ignore')


class QuantumKernel:
    def __init__(self, n_qubits: int = 4, shots: int = None):
        """
        Initialize quantum kernel with specified number of qubits.
        
        Args:
            n_qubits: Number of qubits (≤ 5 for efficiency)
            shots: Number of shots for QPU simulation (None for exact)
        """
        self.n_qubits = min(n_qubits, 5)  # Enforce ≤ 5 qubits constraint
        self.shots = shots
        self.device = qml.device('default.qubit', wires=self.n_qubits, shots=shots)
        
        # Build the quantum feature map
        self.feature_map = self.build_feature_state(self.n_qubits)
        
    def build_feature_state(self, n_qubits: int) -> Callable:
        """
        Build PennyLane feature map circuit with RY rotations and CNOT entangling.
        
        Args:
            n_qubits: Number of qubits in the circuit
            
        Returns:
            QNode that outputs the quantum state vector
        """
        @qml.qnode(self.device, interface='numpy')
        def feature_map_circuit(angles):
            """
            Quantum feature map: RY(angle) per qubit + shallow CNOT entangler.
            
            Args:
                angles: Array of rotation angles (length should match n_qubits)
            """
            # Ensure we have the right number of angles
            if len(angles) < n_qubits:
                # Pad with zeros if not enough angles
                padded_angles = np.zeros(n_qubits)
                padded_angles[:len(angles)] = angles
                angles = padded_angles
            elif len(angles) > n_qubits:
                # Truncate if too many angles
                angles = angles[:n_qubits]
            
            # Rotation layer: RY(angle) on each qubit
            for i in range(n_qubits):
                qml.RY(angles[i], wires=i)
            
            # Shallow entangling layer: CNOT gates
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Return the quantum state vector
            return qml.state()
        
        return feature_map_circuit
    
    def compute_fidelity(self, angles_a: np.ndarray, angles_b: np.ndarray) -> float:
        """
        Compute squared fidelity |⟨ψ(a)|ψ(b)⟩|^2 between two quantum states.
        
        Args:
            angles_a: Angles for first quantum state
            angles_b: Angles for second quantum state
            
        Returns:
            Squared fidelity value in [0, 1]
        """
        # Get quantum states
        state_a = self.feature_map(angles_a)
        state_b = self.feature_map(angles_b)
        
        # Compute fidelity: |⟨ψ_a|ψ_b⟩|^2
        overlap = np.abs(np.vdot(state_a, state_b))**2
        
        return float(overlap)
    
    def kernel_matrix(self, angles_A: np.ndarray, angles_B: np.ndarray = None) -> np.ndarray:
        """
        Compute kernel matrix between sets of quantum feature vectors.
        
        Args:
            angles_A: First set of angle vectors (n_samples_A, n_features)
            angles_B: Second set of angle vectors (n_samples_B, n_features)
                     If None, computes K(A, A)
                     
        Returns:
            Kernel matrix of shape (n_samples_A, n_samples_B)
        """
        if angles_B is None:
            angles_B = angles_A
        
        n_A = angles_A.shape[0]
        n_B = angles_B.shape[0]
        
        # Initialize kernel matrix
        K = np.zeros((n_A, n_B))
        
        # Compute pairwise fidelities
        for i in range(n_A):
            for j in range(n_B):
                K[i, j] = self.compute_fidelity(angles_A[i], angles_B[j])
        
        return K
    
    def kernel_matrix_efficient(self, angles_A: np.ndarray, angles_B: np.ndarray = None) -> np.ndarray:
        """
        More efficient kernel matrix computation using vectorized operations.
        
        Args:
            angles_A: First set of angle vectors (n_samples_A, n_features)
            angles_B: Second set of angle vectors (n_samples_B, n_features)
                     If None, computes K(A, A)
                     
        Returns:
            Kernel matrix of shape (n_samples_A, n_samples_B)
        """
        if angles_B is None:
            angles_B = angles_A
            symmetric = True
        else:
            symmetric = False
        
        n_A = angles_A.shape[0]
        n_B = angles_B.shape[0]
        
        # For small datasets, use the regular method
        if n_A * n_B <= 1000:
            return self.kernel_matrix(angles_A, angles_B)
        
        # For larger datasets, compute in batches
        batch_size = 50
        K = np.zeros((n_A, n_B))
        
        for i in range(0, n_A, batch_size):
            end_i = min(i + batch_size, n_A)
            for j in range(0, n_B, batch_size):
                end_j = min(j + batch_size, n_B)
                
                # Compute batch kernel
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        if symmetric and jj < ii:
                            K[ii, jj] = K[jj, ii]  # Use symmetry
                        else:
                            K[ii, jj] = self.compute_fidelity(angles_A[ii], angles_B[jj])
        
        return K
    
    def add_jitter(self, angles: np.ndarray, jitter_std: float = 0.001) -> np.ndarray:
        """
        Add small random jitter to angles for uncertainty estimation.
        
        Args:
            angles: Input angles
            jitter_std: Standard deviation of jitter noise
            
        Returns:
            Jittered angles
        """
        jitter = np.random.normal(0, jitter_std, angles.shape)
        return angles + jitter
    
    def compute_uncertainty(self, angles_input: np.ndarray, angles_train: np.ndarray, 
                          model, n_repeats: int = 20, jitter_std: float = 0.001) -> Tuple[float, float]:
        """
        Compute prediction uncertainty using jittered quantum kernels.
        
        Args:
            angles_input: Input angles for prediction
            angles_train: Training angles for kernel computation
            model: Trained quantum model
            n_repeats: Number of jittered predictions
            jitter_std: Standard deviation of jitter
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        predictions = []
        
        for _ in range(n_repeats):
            # Add jitter to input angles
            jittered_angles = self.add_jitter(angles_input.reshape(1, -1), jitter_std)
            
            # Compute kernel with training data
            K_input = self.kernel_matrix(jittered_angles, angles_train)
            
            # Get prediction
            pred_proba = model.predict_proba(K_input)[0, 1]  # Probability of fraud
            predictions.append(pred_proba)
        
        return np.mean(predictions), np.std(predictions)


# Utility functions for shot-based evaluation (QPU compatibility)
def setup_qpu_device(backend_name: str = 'default.qubit', shots: int = 1024):
    """
    Setup quantum device for shot-based evaluation.
    
    Args:
        backend_name: Name of the quantum backend
        shots: Number of measurement shots
        
    Returns:
        Configured quantum device
    """
    return qml.device(backend_name, shots=shots)


def estimate_fidelity_shots(kernel: QuantumKernel, angles_a: np.ndarray, 
                           angles_b: np.ndarray, n_shots: int = 1024) -> float:
    """
    Estimate fidelity using shot-based measurements (for QPU deployment).
    
    Args:
        kernel: QuantumKernel instance
        angles_a: First set of angles
        angles_b: Second set of angles
        n_shots: Number of measurement shots
        
    Returns:
        Estimated fidelity
    """
    # This is a placeholder for shot-based fidelity estimation
    # In practice, you'd implement SWAP test or other fidelity estimation protocols
    return kernel.compute_fidelity(angles_a, angles_b)


if __name__ == "__main__":
    # Test the quantum kernel
    print("Testing Quantum Kernel...")
    
    # Create test data
    n_samples = 10
    n_features = 4
    angles_test = np.random.uniform(0, 2*np.pi, (n_samples, n_features))
    
    # Initialize kernel
    qkernel = QuantumKernel(n_qubits=4)
    
    # Test kernel matrix computation
    K = qkernel.kernel_matrix(angles_test)
    
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel values range: [{K.min():.3f}, {K.max():.3f}]")
    print(f"Diagonal values (should be ~1): {np.diag(K)}")
    print(f"Is symmetric: {np.allclose(K, K.T)}")
    
    # Test single fidelity computation
    fidelity = qkernel.compute_fidelity(angles_test[0], angles_test[1])
    print(f"Sample fidelity: {fidelity:.3f}")
    
    print("Quantum kernel tests completed!")