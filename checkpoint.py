import pickle

class Checkpoint:
    def __init__(self, model, path):
        self.model = model
        self.path = path

    def save(self, epoch):
        with open(f"{self.path}/checkpoint_epoch_{epoch}.pth", 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, epoch):
        with open(f"{self.path}/checkpoint_epoch_{epoch}.pth", 'rb') as f:
            self.model = pickle.load(f)
        return self.model
