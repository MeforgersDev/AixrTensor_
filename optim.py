class AixrSGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class AixrAdam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * (1 - self.t / 10000)

        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                param.data -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
