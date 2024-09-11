class AixrNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.visualizer = TrainingVisualizer()
        self.memory_optimizer = MemoryOptimizer(self)
        self.checkpoint = Checkpoint(self, 'checkpoints')  # 'checkpoints' klasörünü oluşturmalısınız

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def train(self, x_train, y_train, epochs, optimizer, loss_fn):
        for epoch in range(epochs):
            self.memory_optimizer.monitor()
            output = self.forward(x_train)
            loss = loss_fn(output, y_train)
            print(f'Epoch {epoch + 1}, Loss: {loss.data}')
            self.visualizer.on_epoch_end(epoch + 1, loss.data)
            optimizer.zero_grad()
            loss.backward()
            self.backward(loss.grad)
            optimizer.step()
            self.checkpoint.save(epoch + 1)  # Her epoch sonunda checkpoint kaydedilir