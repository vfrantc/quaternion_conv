# A Solution to the 4-bit Parity Problem with a Single Quaternary Neuron
import itertools
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from core_qnn.quaternion_layers import QuaternionLinear
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = QuaternionLinear(4, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    # Initialize the network
    model = QNet()

    # Define the loss function (binary cross-entropy)
    criterion = nn.BCELoss()

    # Define the optimizer (Adam with a learning rate of 0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Generate training data
    x = np.asarray(list(itertools.product([0, 1], repeat=4)))
    x_ = x.copy()
    # y = np.asarray([np.sum(x[i]) % 2 for i in range(len(x))])
    y = np.asarray(list(itertools.product([0, 1], repeat=4)))
    y_ = y.copy()
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    # Train the network for 100 epochs
    for epoch in range(100):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/100], Loss: {:.4f}'.format(epoch + 1, loss.item()))

        # Evaluate the model on the training data
    with torch.no_grad():
        outputs = model(x)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y).float().mean()
        print('Accuracy on the training data: {:.2f}%'.format(accuracy.item() * 100))



    feature_1, feature_2, feature_3, feature_4 = np.meshgrid(
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100))
    grid = np.stack([feature_1, feature_2, feature_3, feature_4]).T
    grid = torch.FloatTensor(grid)
    # Make predictions on the test data
    with torch.no_grad():
        outputs = model(grid)
        predicted = (outputs > 0.5).float().numpy().flatten()

    y_pred = np.reshape(predicted, feature_1.shape)
    sl = 25
    display = DecisionBoundaryDisplay(xx0=feature_1[:, :, sl, sl], xx1=feature_2[:, :, sl, sl], response=y_pred[:, :, sl, sl])
    display.plot()
    display.ax_.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor="black")
    display.ax_.set_xlabel('petal length (scaled)')
    display.ax_.set_ylabel('petal width (scaled)')
    plt.show()