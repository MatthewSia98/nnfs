import numpy as np

class ReluActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)