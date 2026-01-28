import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

class NeuralNet:
    def __init__(self, L, n, epochs, alpha, mu, fact, val_percent):
        """Initialize the class with all the data structures"""
        self.L = L                  # Number of layers
        self.n = n                  # Array of units per layer
        self.epochs = epochs
        self.alpha = alpha          # Learning rate
        self.mu = mu                # Momentum
        self.fact = fact            # Activation name
        self.val_percent = val_percent

        # Required arrays of arrays/matrices
        self.w = []                 # w (weights)
        self.theta = []             # thresholds (biases)
        self.xi = []                # activations (output of the neurons after the act. func.)
        self.h = []                 # fields (output of the neurons)
        
        # Momentum storage
        self.d_w_prev = [None] * (L-1)
        self.d_theta_prev = [None] * (L-1)

        # Errors
        self.train_errors = []
        self.val_errors = []

        # Activations dictionary
        self.activations = {
            'tanh': (lambda x: np.tanh(x), lambda x: 1.0 - np.tanh(x)**2),
            'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
            'sigmoid': (lambda x: 1/(1+np.exp(-x)), lambda x: (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))),
            'linear': (lambda x: x, lambda x: np.ones_like(x))
        }
        self.activate, self.derivative = self.activations[fact]
        
        self._initialize_weights(n)

    def _initialize_weights(self, n):
        """Initializes the weights."""
        for i in range(len(n) - 1):
            # weights (w) and thresholds (theta)
            self.w.append(np.random.randn(n[i], n[i+1]) * 0.01)
            self.theta.append(np.zeros((1, n[i+1])))
            
            # Initialize momentum trackers with zeros
            self.d_w_prev[i] = np.zeros((n[i], n[i+1]))
            self.d_theta_prev[i] = np.zeros((1, n[i+1]))

    def forward(self, input_data):
        """Forwards some data through the neural network"""
        self.h = [] # Reset fields
        self.xi = [input_data] # Reset activations with input
        
        current_data = input_data
        for i in range(len(self.w)):
            # z = (input * w) + theta
            layer_output = np.dot(current_data, self.w[i]) + self.theta[i]
            self.h.append(layer_output)
            
            if i < len(self.w) - 1:
                current_data = self.activate(layer_output)
            else:
                current_data = layer_output # Output layer linear
            self.xi.append(current_data)
                
        return current_data
    
    def step_descent(self, X, y_true, learning_rate):
        """Step of stochastic descent."""
        y_pred = self.forward(X)
        
        # Output layer error
        error = y_pred - y_true
        d = error # local delta
        
        # Backpropagate
        for i in reversed(range(len(self.w))):
            # Compute gradients
            grad_w = np.dot(self.xi[i].T, d)
            grad_theta = np.sum(d, axis=0, keepdims=True)
            
            # Update delta for next layer (if not at start)
            if i > 0:
                d = np.dot(d, self.w[i].T) * self.derivative(self.h[i-1])

            delta_w = (-learning_rate * grad_w) + (self.mu * self.d_w_prev[i])
            delta_theta = (-learning_rate * grad_theta) + (self.mu * self.d_theta_prev[i])
            
            # Clip the deltas so that it does not overflow with big errors (mostly at the start of training rounds)
            delta_w = np.clip(delta_w, -1, 1)
            delta_theta = np.clip(delta_theta, -1, 1)

            # Apply the changes to the weights and thresholds
            self.w[i] += delta_w
            self.theta[i] += delta_theta
            
            # Store the changes for the next step
            self.d_w_prev[i] = delta_w
            self.d_theta_prev[i] = delta_theta
            
        return np.mean(error**2)

    def fit(self, X, y):
        """Train the model."""
        if self.val_percent > 0:
            n_samples = len(X)
            # Shuffle the dataset
            indices = np.random.permutation(n_samples)
            n_val = int(n_samples * self.val_percent)
            
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            total_train_loss = 0
            # Train using only with the training set
            for i in range(len(X_train)):
                total_train_loss += self.step_descent(X_train[i:i+1], y_train[i:i+1], self.alpha)

            # Average training loss
            avg_train_loss = total_train_loss / len(X_train)
            self.train_errors.append(avg_train_loss)

            # Compute Validation Loss
            if self.val_percent > 0:
                y_val_pred = self.predict(X_val)
                # Compute MSE for validation
                avg_val_loss = np.mean((y_val_pred - y_val.reshape(-1, 1))**2)
                self.val_errors.append(avg_val_loss)
                val_status = f" | Val MSE: {avg_val_loss:.6f}"
            else:
                val_status = ""
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch} | Train MSE: {avg_train_loss:.6f}{val_status}")


    
    def loss_epochs(self):
        """Returns two arrays of size (n_epochs, 2)."""
        # Create an array of epoch indices: [0, 1, 2, ..., n_epochs-1]
        epochs_range = np.arange(len(self.train_errors)).reshape(-1, 1)
        
        # Reshape the train errors into a single column
        train_loss_vec = np.array(self.train_errors).reshape(-1, 1)
        # Combine into a (n_epochs, 2) array
        train_results = np.hstack((epochs_range, train_loss_vec))

        
        # Do the same for validation if it exists, otherwise return zeros or empty
        if len(self.val_errors) > 0:
            val_loss_vec = np.array(self.val_errors).reshape(-1, 1)
            val_results = np.hstack((epochs_range, val_loss_vec))
        else:
            val_results = np.hstack((epochs_range, np.zeros_like(train_loss_vec)))

        return train_results, val_results


    def predict(self, X):
        """Predicts values for a given set of samples."""
        predictions = []
        for i in range(len(X)):
            sample_prediction = self.forward(X[i:i+1])
            predictions.append(sample_prediction)
            
        return np.array(predictions)

# Load the data
X_data, y_data = load_data_from_csv('life_expectancy_data_processed.csv')

# Define the NeuralNet class
architecture = [X_data.shape[1], 64, 32, 1] 
model = NeuralNet(
    L=len(architecture),
    n=architecture,
    epochs=30,
    alpha=0.0001,
    mu=0.9,
    fact='relu',
    val_percent=0.2
)

# Training
model.fit(X_data, y_data)

# Test predict function
sample_predictions = model.predict(X_data[:10])
actual_values = y_data[:10]
print("\n--- Prediction Test (First 10 samples) ---")
for i in range(len(sample_predictions)):
    pred = sample_predictions[i].item() 
    actual = actual_values[i].item()
    print(f"Sample {i}: Predicted = {pred:.2f}, Actual = {actual:.2f}")

# Test the loss per epoch function and visualize it
train_log, val_log = model.loss_epochs()
plt.figure(figsize=(10, 6))
plt.plot(train_log[:, 0], train_log[:, 1], label='Training Loss', color='blue')
plt.plot(val_log[:, 0], val_log[:, 1], label='Validation Loss', color='red', linestyle='--')
plt.title('Model Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

