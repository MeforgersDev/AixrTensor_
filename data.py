import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, augment_fn=None, normalize=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_fn = augment_fn
        self.normalize = normalize

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i+self.batch_size]
            if self.augment_fn:
                batch = self.augment_fn(batch)
            if self.normalize:
                batch = self._normalize(batch)
            yield batch

    def _normalize(self, batch):
        return (batch - np.mean(batch, axis=0)) / np.std(batch, axis=0)
