import numpy as np
from .tensor import AixrTensor

class AixrReLU:
    @staticmethod
    def forward(x):
        result = AixrTensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        result._grad_fn = AixrReLU
        return result

    @staticmethod
    def backward(grad):
        x, = AixrReLU.saved_tensors
        grad_input = grad * (x.data > 0)
        if x.requires_grad:
            x.backward(grad_input)

class AixrSigmoid:
    @staticmethod
    def forward(x):
        result = 1 / (1 + np.exp(-x.data))
        result = AixrTensor(result, requires_grad=x.requires_grad)
        result._grad_fn = AixrSigmoid
        return result

    @staticmethod
    def backward(grad):
        x, = AixrSigmoid.saved_tensors
        grad_input = grad * (1.0 - x.data) * x.data
        if x.requires_grad:
            x.backward(grad_input)

class AixrTanh:
    @staticmethod
    def forward(x):
        result = np.tanh(x.data)
        result = AixrTensor(result, requires_grad=x.requires_grad)
        result._grad_fn = AixrTanh
        return result

    @staticmethod
    def backward(grad):
        x, = AixrTanh.saved_tensors
        grad_input = grad * (1 - np.square(x.data))
        if x.requires_grad:
            x.backward(grad_input)
