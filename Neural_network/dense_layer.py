import numpy as np
class DenseLayer:
    def __init__(self, n_entradas: int, n_neuronas: int):
        self.weight = 0.01 * np.random.randn(n_entradas, n_neuronas)
        self.biases = np.zeros((1, n_neuronas))

    def forward(self, batch):
        self.output = np.dot(batch, self.weight) + self.biases
        return self.output