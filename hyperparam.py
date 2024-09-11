import itertools
import numpy as np

class HyperparameterTuner:
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid

    def random_search(self, x_train, y_train, n_iter=10):
        """Random search for best hyperparameters."""
        best_loss = float('inf')
        best_params = None
        for _ in range(n_iter):
            params = self._random_params(self.param_grid)
            print(f"Testing params: {params}")
            self.model.compile(optimizer=params['optimizer'], loss=params['loss'])
            self.model.train(x_train, y_train, epochs=params['epochs'])
            loss = self._evaluate(x_train, y_train)
            if loss < best_loss:
                best_loss = loss
                best_params = params
        return best_params

    def grid_search(self, x_train, y_train, epochs=5):
        """Grid search for best hyperparameters."""
        best_params = None
        best_loss = float('inf')
        for params in self._param_combinations(self.param_grid):
            print(f"Testing params: {params}")
            self.model.compile(optimizer=params['optimizer'], loss=params['loss'])
            self.model.train(x_train, y_train, epochs=epochs)
            loss = self._evaluate(x_train, y_train)
            if loss < best_loss:
                best_loss = loss
                best_params = params
        return best_params

    def _param_combinations(self, param_grid):
        keys = param_grid.keys()
        values = (param_grid[key] for key in keys)
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _random_params(self, param_grid):
        """Generate random parameter combinations."""
        return {k: np.random.choice(v) for k, v in param_grid.items()}

    def _evaluate(self, x_train, y_train):
        output = self.model.forward(x_train)
        return self.model.loss(output, y_train).data
