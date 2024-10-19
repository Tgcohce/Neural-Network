import numpy as np

# Define Neural Network class
class NeuralNetwork:
    # Initialize weights, biases, learning rate
    def __init__(self, input_size, hidden_sizes, output_size, initial_learning_rate=0.001, decay_rate=0.00001):
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * 0.01
        self.b1 = np.zeros((1, hidden_sizes[0]))
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * 0.01
        self.b2 = np.zeros((1, hidden_sizes[1]))
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    # Leaky ReLU activation
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    # Derivative of Leaky ReLU
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    # Forward pass
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.leaky_relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.z3  # Output layer
        return self.a3

    # Backpropagation
    def backward(self, X, y):
        a3_error = self.a3 - y
        a3_delta = a3_error

        a2_error = np.dot(a3_delta, self.W3.T)
        a2_delta = a2_error * self.leaky_relu_derivative(self.z2)

        a1_error = np.dot(a2_delta, self.W2.T)
        a1_delta = a1_error * self.leaky_relu_derivative(self.z1)

        # Update weights and biases
        current_lr = self.initial_learning_rate / (1 + self.decay_rate)
        self.W3 -= current_lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= current_lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.W2 -= current_lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= current_lr * np.sum(a2_delta, axis=0, keepdims=True)
        self.W1 -= current_lr * np.dot(X.T, a1_delta)
        self.b1 -= current_lr * np.sum(a1_delta, axis=0, keepdims=True)

        loss = np.mean((y - self.a3) ** 2)
        return loss

    # Train the network
    def train(self, X, y, epochs=15000):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    # Predict new data
    def predict(self, X):
        return self.forward(X)


# Training data: normalize inputs
X = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
X = (X - np.mean(X)) / np.std(X)
y = np.sin(X) + np.cos(2 * X)

# Initialize and train the neural network
nn = NeuralNetwork(input_size=1, hidden_sizes=[50, 50], output_size=1, initial_learning_rate=0.001, decay_rate=0.0001)
nn.train(X, y, epochs=15000)

# Test the trained model
test_X = np.array([[0], [np.pi / 4], [np.pi / 2], [3 * np.pi / 4], [np.pi]])
test_X = (test_X - np.mean(X)) / np.std(X)
predictions = nn.predict(test_X)

# Print test results
print("\nTesting:")
for i in range(len(test_X)):
    print(
        f"Input: {test_X[i]} - Predicted Output: {predictions[i]} - Actual Output: {np.sin(test_X[i]) + np.cos(2 * test_X[i])}")
