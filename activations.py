import numpy as np

class ReluActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class SoftmaxActivation:
    def forward(self, inputs):
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sums = np.sum(exps, axis=1, keepdims=True)
        probabilities = exps / sums
        self.output = probabilities
