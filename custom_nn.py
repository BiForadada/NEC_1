import pandas as pd
import numpy as np

def load_data_from_csv(csv_path):
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file path '{csv_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return pd.DataFrame()

    X = df.iloc[:, :-1].values  # All columns except the last
    y = df.iloc[:, -1].values   # Just the last column
    
    return X, y

class NeuralNetworkCustom:
    def __init__(self, units_per_layer: list, activation_name: str):
        """Initialize the main structures of the neural network"""
        self.structure = units_per_layer
        self.weights = []
        self.biases = []

        # Define the activation functions and its derivatives
        self.activations = {
            'tanh': (lambda x: np.tanh(x), 
                     lambda x: 1.0 - np.tanh(x)**2),
            'relu': (lambda x: np.maximum(0, x), 
                     lambda x: (x > 0).astype(float)),
            'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), 
                        lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))))
        }
        if activation_name not in self.activations:
            raise ValueError(f"Choose from: {list(self.activations.keys())}")
        
        # Set the choosen function
        self.activate, self.derivative = self.activations[activation_name]
        # Initialize the weights randomly
        self._initialize_weights(units_per_layer)

    def _initialize_weights(self, units_per_layer):
        """Initialize the weights of the nn randomly, and the biases to 0"""
        for i in range(len(units_per_layer) - 1):
            n_input = units_per_layer[i]
            n_output = units_per_layer[i+1]
            
            # Create a matrix of random weights for this connection of the nn
            w = np.random.randn(n_input, n_output) * 0.1
            self.weights.append(w)
            
            # Create a bias vector with all 0 values
            b = np.zeros((1, n_output))
            self.biases.append(b)

    def forward(self, input_data):
        """Forward a vector of data through the model"""
        self.output_neurons = []
        self.activations_list = [input_data]
        
        current_data = input_data
        for i in range(len(self.weights)):
            layer_output = np.dot(current_data, self.weights[i]) + self.biases[i]
            self.output_neurons.append(layer_output)
            
            if i < len(self.weights) - 1:
                current_data = self.activate(layer_output)
            else:
                current_data = layer_output
            self.activations_list.append(current_data)
                
        return current_data
    
    def step_descent(self, X, y_true, learning_rate):
        """One step of Stochastic Gradient Descent"""
        # Forward pass to populate self.output_neurons and self.activations_list
        y_pred = self.forward(X)
        
        # Get the error at the output layer
        error = y_pred - y_true
        delta = error 

        # Backpropagate the error
        for i in reversed(range(len(self.weights))):
            # Compute gradient for weights: (Input to layer) * delta. And for biases: 1 * Delta
            grad_w = np.dot(self.activations_list[i].T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            # If not at the first layer, compute delta for the next layer of the nn (Previous since we are doing backpropagation)
            if i > 0:
                # delta = (delta_ahead * weights_ahead) * derivative_of_current_layer
                delta = np.dot(delta, self.weights[i].T) * self.derivative(self.output_neurons[i-1])
            
            # Update weights and biases
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
        return np.mean(error**2) # Return loss
    
    def train(self, n_epochs, X_data, y_data, lr):
        """Train the model for n_epochs for the full dataset"""
        print("Starting training...")
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(len(X_data)):
                row_x = X_data[i:i+1]
                row_y = y_data[i:i+1]
                loss = self.step_descent(row_x, row_y, lr)
                total_loss += loss
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Average MSE: {total_loss/len(X_data):.6f}")
    

# Set the variables
X_data, y_data = load_data_from_csv('life_expectancy_data_processed.csv')
brain = NeuralNetworkCustom([X_data.shape[1], 64, 1], 'tanh')

# Training stage
brain.train(10, X_data, y_data, 0.001)
