import numpy as np
import psutil

class AixrTensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.precision = 'float32'
        self.device = self._select_best_device()
        self.requires_grad = requires_grad
        self.grad = None
        self._original_data = np.array(data, dtype=dtype)
        self.data = self._apply_precision(self._original_data)

    def _select_best_device(self):
        try:
            import cupy as cp
            return 'gpu'
        except ImportError:
            try:
                import jax
                return 'tpu'
            except ImportError:
                return 'cpu'

    def _apply_precision(self, data):
        if self.precision == 'float16':
            return data.astype(np.float16)
        return data.astype(np.float32)

    def set_precision(self, precision):
        """Change precision for mixed-precision training."""
        self.precision = precision
        self.data = self._apply_precision(self.data)

    def to(self, device):
        self.device = device
        if device == 'gpu':
            import cupy as cp
            self.data = cp.array(self._original_data)
        elif device == 'tpu':
            import jax.numpy as jnp
            self.data = jnp.array(self._original_data)
        else:
            self.data = np.array(self._original_data)
        return self

    def monitor_memory(self, threshold=0.8):
        """Monitor system memory and offload tensor if usage exceeds threshold."""
        mem_usage = psutil.virtual_memory().percent / 100
        if mem_usage > threshold:
            print(f"Memory usage high: {mem_usage*100}%, moving tensor to CPU.")
            self.to('cpu')
        else:
            print(f"Memory usage: {mem_usage*100}% - Running on {self.device}.")
