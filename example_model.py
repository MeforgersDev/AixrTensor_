import numpy as np
from aixrtensor import AixrTensor, AixrNeuralNetwork, AixrLinear, AixrReLU, AixrSGD, mse_loss, Checkpoint

# Create a simple neural network with mixed precision
layer1 = AixrLinear(10, 20)
layer1.weights.set_precision('float16')
layer1.bias.set_precision('float16')

model = AixrNeuralNetwork([
    layer1,
    AixrReLU(),
    AixrLinear(20, 1),
])

# Compile the model with optimizer and loss function
optimizer = AixrSGD(model.parameters(), lr=0.01)
model.compile(optimizer=optimizer, loss=mse_loss)

# Dummy data
x_train = AixrTensor(np.random.randn(100, 10))
y_train = AixrTensor(np.random.randn(100, 1))

# Train the model
model.train(x_train, y_train, epochs=10, optimizer=optimizer, loss_fn=mse_loss)

# Save the model
model.checkpoint.save(10)

# Load the model
loaded_model = model.checkpoint.load(10)
