import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.losses = []

    def on_epoch_end(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self._plot_loss()

    def _plot_loss(self):
        plt.plot(self.epochs, self.losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.show()
