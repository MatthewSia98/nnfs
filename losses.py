import numpy as np

class CategoricalCrossEntropyLoss:
    def forward(self, prediction, target):
        self.output = -np.sum(target * np.log(prediction))
