import psutil

class MemoryOptimizer:
    def __init__(self, model, threshold=0.75):
        self.model = model
        self.threshold = threshold

    def monitor(self):
        memory_usage = psutil.virtual_memory().percent / 100
        if memory_usage > self.threshold:
            print(f"Memory usage: {memory_usage*100}% - Optimizing.")
            self.model.to('cpu')
        else:
            print(f"Memory usage: {memory_usage*100}% - Running on {self.model.device}.")
