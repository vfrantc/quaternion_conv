import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import matplotlib.pyplot as plt

# Load iris dataset
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# MLP model with BatchNorm
class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x

# MLP model without BatchNorm
class MLPWithoutBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
def train(model, X, y, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    activations = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store activations for visualization
        with torch.no_grad():
            activations.append(model.fc1(X).detach().numpy().flatten())

    return activations

# Train MLP with BatchNorm
input_size = X.shape[1]
hidden_size = 20
num_classes = 3
model_with_bn = MLPWithBatchNorm(input_size, hidden_size, num_classes)
activations_with_bn = train(model_with_bn, X, y)

# Train MLP without BatchNorm
model_without_bn = MLPWithoutBatchNorm(input_size, hidden_size, num_classes)
activations_without_bn = train(model_without_bn, X, y)

# Plot histograms of activations
plt.figure(figsize=(20,10))
for i, activations in enumerate([activations_with_bn, activations_without_bn]):
    for j, a in enumerate(activations[::10]):
        plt.subplot(2, 10, i*10 + j + 1)
        plt.hist(a, bins=20)

plt.suptitle("Histograms of activations during training (BN = blue, No BN = orange)")
plt.show()
'''

def train(model, X, y, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    accuracy_list = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track accuracy
        with torch.no_grad():
            y_pred = model(X)
            correct = (torch.argmax(y_pred, dim=1) == y).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    return accuracy_list

# Train MLP with BatchNorm
input_size = X.shape[1]
hidden_size = 20
num_classes = 3
model_with_bn = MLPWithBatchNorm(input_size, hidden_size, num_classes)
accuracy_with_bn = train(model_with_bn, X, y)

# Train MLP without BatchNorm
model_without_bn = MLPWithoutBatchNorm(input_size, hidden_size, num_classes)
accuracy_without_bn = train(model_without_bn, X, y)

# Plot accuracy
plt.plot(accuracy_with_bn.numpy(), label="With BatchNorm")
plt.plot(accuracy_without_bn.numpy(), label="Without BatchNorm")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy during training")
plt.show()
