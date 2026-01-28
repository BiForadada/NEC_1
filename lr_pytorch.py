import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load and split the data
df = pd.read_csv('life_expectancy_data_processed.csv')
X = df.iloc[:, :-1].values.astype(np.float32) 
y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

# Convert to PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train_val)
y_train_tensor = torch.from_numpy(y_train_val)
X_test_tensor = torch.from_numpy(X_test)

# Define the neural network structure
# I chose the structure with the best results at the baseline implementation
class BP_F_Network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
model = BP_F_Network(input_dim)
criterion = nn.MSELoss()
# Define the optimizer (I wrote both to make tests)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Training the neural network
epochs = 500
print("Starting BP-F Training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Back-propagation
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Prediction
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

# Compute metrics 
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\n--- Evaluation Metrics ---")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='purple', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Real Values')
plt.legend()
plt.grid(True)
plt.show()