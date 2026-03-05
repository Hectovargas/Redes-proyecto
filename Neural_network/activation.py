import numpy as np

class Activation_Relu:
    def forward(self, batch):
        self.output = np.maximum(0, batch)
        return self.output

class softmax:
    def forward(self, batch):
        exp = np.exp(batch - np.max(batch, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output