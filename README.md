# Neural Network from Scratch with NumPy

A simple implementation of a neural network using only `numpy`, without relying on any external deep learning libraries. 
### Included within the project:
forward and backward propagation, gradient descent, and training a neural network for regression tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a basic neural network using Python and `numpy`. It includes:
- Two hidden layers with leaky ReLU activation.
- Customizable learning rate and decay.
- A mean squared error loss function.
- Training data generated from a combination of `sin` and `cos` functions.

## Features

- **Fully Customizable**: Easily adjust the number of hidden layers, neurons, learning rate, and more.
- **Regression Task**: Trains on a function combining `sin` and `cos` and predicts outputs for new inputs.

## Installation

1. **Clone the repository**

2. **Create a virtual environment** (this is optional but i recommend it):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install requirements**:
   - For this script, `numpy` is the only dependency.
   ```bash
   pip install numpy
   ```

## Usage

1. **Run the script**:
   ```bash
   python main.py
   ```

2. **Modify the Neural Network**:
   - Adjust `input_size`, `hidden_sizes`, `output_size`, `initial_learning_rate`, and `decay_rate` to suit your needs.
   - Customize the training data by modifying the `X` and `y` values.

3. **View Training Progress**:
   - The script prints the loss every 100 epochs, showing how the network learns over time.

## Code Explanation

The main components of the script include:

- **Initialization**: Sets up weights, biases, learning rate, and decay rate.
- **Forward Pass**: Propagates inputs through the network.
- **Backward Pass**: Computes gradients and updates weights using gradient descent.
- **Training**: Iteratively adjusts weights to minimize loss.
- **Prediction**: Uses the trained model to predict new outputs.


## Results

After training for 15,000 epochs, the network can accurately approximate a function combining `sin` and `cos`:

```
Epoch 0, Loss: 1.089
...
Epoch 14900, Loss: 0.000572
```

**Test Predictions:**

| Input         | Predicted Output | Actual Output |
|---------------|------------------|---------------|
| `0`           | `0.9829`         | `1.0`         |
| `π/4`         | `0.6890`         | `0.7071`      |
| `π/2`         | `0.0058`         | `0.0`         |
| `3π/4`        | `-0.1603`        | `0.7071`      |
| `π`           | `-0.3285`        | `1.0`         |

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project. Here’s how you can contribute:
- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Commit your changes (`git commit -m 'Add a new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
