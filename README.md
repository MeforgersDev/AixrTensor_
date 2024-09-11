# 🧠 **AixrTensor** - The Next-Gen AI Tensor Library

[![License](https://img.shields.io/github/license/MeforgersDev/AixrTensor)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)
[![GitHub issues](https://img.shields.io/github/issues/MeforgersDev/Aixrtensor)](https://github.com/MeforgersDev/AixrTensor/issues)

AixrTensor is a **customizable and dynamic tensor library** designed to outperform TensorFlow with enhanced memory management, dynamic model structures, automated hyperparameter tuning, and simplified AI training workflows. Build, train, and deploy your machine learning models with **efficiency** and **ease**!

---

## 🌟 **Key Features**

- **Automatic Device Selection**: Automatically selects the best available hardware (CPU, GPU, or TPU) for optimal performance.
- **Mixed Precision Training**: Supports `float16` and `float32` for memory optimization without sacrificing accuracy.
- **Dynamic Model Structure**: Modify your neural network's structure during training! Add or remove layers as needed.
- **Automated Hyperparameter Tuning**: Built-in random and grid search for automatic hyperparameter optimization.
- **Advanced Memory Management**: Monitors system memory and moves tensors between devices dynamically to prevent memory overloads.
- **Built-in Checkpointing**: Automatically saves and loads model checkpoints during training.
- **Visualize Training**: Real-time plotting of loss and performance metrics for quick feedback on model performance.

---

## 🚀 **Installation**

To install AixrTensor from GitHub, simply run:

```bash
pip install git+https://github.com/MeforgersDev/AixrTensor_.git
```
Make sure to have numpy, matplotlib, and psutil installed for full functionality:
```bash
pip install numpy matplotlib psutil
```

## 📖 **Getting Started**

AixrTensor is designed to be intuitive and user-friendly. Below is a quick example to get you started.

**Example: Building and Training a Neural Network**
```python
import numpy as np
from aixrtensor import AixrTensor, AixrNeuralNetwork, AixrLinear, AixrReLU, AixrSGD, mse_loss

# Define a simple neural network
model = AixrNeuralNetwork([
    AixrLinear(10, 20),   # Fully connected layer
    AixrReLU(),           # Activation function
    AixrLinear(20, 1),    # Output layer
])

# Compile the model with SGD optimizer and mean squared error loss function
optimizer = AixrSGD(model.parameters(), lr=0.01)
model.compile(optimizer=optimizer, loss=mse_loss)

# Generate dummy data
x_train = AixrTensor(np.random.randn(100, 10))
y_train = AixrTensor(np.random.randn(100, 1))

# Train the model for 10 epochs
model.train(x_train, y_train, epochs=10)
```
**Adding Dynamic Layers During Training**
AixrTensor allows you to modify the model structure dynamically:
```python
# Add a new layer during training
model.add_layer(AixrLinear(1, 5), position=2)
```
## 🛠️ **Advanced Features**

- **Automatic Hyperparameter Tuning**
Easily perform random search or grid search for hyperparameter tuning during model training:
```python
from aixrtensor import HyperparameterTuner

# Define a parameter grid
param_grid = {
    'optimizer': [AixrSGD, AixrAdam],
    'loss': [mse_loss],
    'epochs': [5, 10]
}

# Initialize the tuner and search for the best parameters
tuner = HyperparameterTuner(model, param_grid)
best_params = tuner.grid_search(x_train, y_train, epochs=5)
print("Best Parameters:", best_params)
```
- **Mixed Precision Training**
For memory-efficient training, AixrTensor supports mixed precision training, allowing layers to use different floating-point precisions:
```python
# Set the first layer's weights to float16 precision
model.layers[0].weights.set_precision('float16')
```
- **Real-time Training Visualization**
Track the training progress with real-time plotting of loss:
```python
# Real-time visualization of training loss
model.visualizer.on_epoch_end(epoch, loss)
```
- **Advanced Memory Management**
Monitor memory usage and dynamically offload tensors to CPU when memory exceeds a certain threshold:
```python
# Monitor memory usage and optimize
model.memory_optimizer.monitor()
```
- **Model Checkpointing**
Save and load model checkpoints automatically during training:
```python
# Save model after each epoch
model.checkpoint.save(epoch=10)

# Load the saved model
model = model.checkpoint.load(epoch=10)
```

## 📊 **Visualizing Training with Loss Graphs**

Training visualization is built into AixrTensor. Each epoch's loss can be visualized with real-time graphs:
```python
import matplotlib.pyplot as plt

# Sample loss visualization code
def plot_loss(epochs, losses):
    plt.plot(epochs, losses)
    plt.title('Training Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Collect and visualize losses
plot_loss([1, 2, 3, 4, 5], [0.5, 0.4, 0.3, 0.25, 0.2])
```

## 🤝 **Contributing**

We welcome contributions from the community! If you'd like to contribute to AixrTensor, please follow these steps:
- Fork the repository.
- Create a new branch.
- Commit your changes.
- Push to the branch.
- Open a pull request.
  
## 📄 **License**

**This project is licensed under the MIT License**
