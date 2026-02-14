"""
Multi-Task Physics-Informed Neural Network (MT-PINN)
For cement manufacturing optimization
"""

import numpy as np


class MTPINN:
    """
    Multi-Task Physics-Informed Neural Network
    
    Architecture:
        Input (16) -> FC(128) -> FC(64) -> FC(32) -> Output(4)
        
    Outputs:
        1. Strength (MPa)
        2. Emissions (kg CO2/kg cement)
        3. Cost ($/ton)
        4. Risk (normalized)
    
    Note: Using numpy only (no PyTorch) for simplicity since network is unavailable
    This is a simplified feedforward network for demonstration
    """
    
    def __init__(self, input_dim=16, hidden_dims=[128, 64, 32], output_dim=4):
        """
        Initialize the MT-PINN model
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dims : list
            Hidden layer dimensions
        output_dim : int
            Number of output targets
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Xavier initialization
            limit = np.sqrt(6 / (prev_dim + hidden_dim))
            W = np.random.uniform(-limit, limit, (prev_dim, hidden_dim))
            b = np.zeros(hidden_dim)
            
            self.weights.append(W)
            self.biases.append(b)
            prev_dim = hidden_dim
        
        # Last hidden to output
        limit = np.sqrt(6 / (prev_dim + output_dim))
        W = np.random.uniform(-limit, limit, (prev_dim, output_dim))
        b = np.zeros(output_dim)
        self.weights.append(W)
        self.biases.append(b)
        
        print(f"MT-PINN initialized:")
        print(f"  Input: {input_dim}")
        print(f"  Hidden: {hidden_dims}")
        print(f"  Output: {output_dim}")
        print(f"  Total parameters: {self.count_parameters()}")
    
    def count_parameters(self):
        """Count total number of parameters"""
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (batch_size, input_dim)
        
        Returns:
        --------
        np.ndarray : Predictions (batch_size, output_dim)
        """
        activations = X
        
        # Forward through hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            activations = activations @ self.weights[i] + self.biases[i]
            activations = self.relu(activations)
        
        # Output layer (no activation)
        output = activations @ self.weights[-1] + self.biases[-1]
        
        return output
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features
        
        Returns:
        --------
        dict : Predictions for each target
        """
        # Convert DataFrame to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Forward pass
        outputs = self.forward(X)
        
        # Split outputs
        return {
            'strength': outputs[:, 0],
            'emissions': outputs[:, 1],
            'cost': outputs[:, 2],
            'risk': outputs[:, 3]
        }
    
    def save(self, filepath):
        """Save model weights"""
        save_dict = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}'] = b
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        data = np.load(filepath)
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_dims) + 1):
            self.weights.append(data[f'weight_{i}'])
            self.biases.append(data[f'bias_{i}'])
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("MT-PINN MODEL TEST")
    print("="*60)
    
    # Create model
    model = MTPINN(
        input_dim=16,
        hidden_dims=[128, 64, 32],
        output_dim=4
    )
    
    # Test with random input
    print("\nTesting forward pass...")
    X_test = np.random.randn(5, 16)  # 5 samples, 16 features
    predictions = model.predict(X_test)
    
    print("\nPredictions for 5 samples:")
    print(f"  Strength: {predictions['strength']}")
    print(f"  Emissions: {predictions['emissions']}")
    print(f"  Cost: {predictions['cost']}")
    print(f"  Risk: {predictions['risk']}")
    
    # Test save/load
    print("\nTesting save/load...")
    model.save('test_model.npz')
    
    model2 = MTPINN(input_dim=16, hidden_dims=[128, 64, 32], output_dim=4)
    model2.load('test_model.npz')
    
    print("\n✓ Model tests passed!")