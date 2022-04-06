import numpy as np

class Loss:
    def calculate(self, prediction, target):
        losses = self.forward(prediction, target)
        batch_loss = np.mean(losses)
        return batch_loss    

class CategoricalCrossEntropyLoss(Loss):
    def __init__(self, one_hot=False):
        self.one_hot = one_hot

    def forward(self, prediction, target):
        clipped_prediction = np.clip(prediction, 1e-7, 1-1e-7)

        if self.one_hot:
            losses = -np.sum(target * np.log(clipped_prediction), axis=1)
            self.output = losses   
        else:
            losses = -np.log(clipped_prediction[range(len(target)), target])
            
        return losses

"""
softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])
one_hot_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
print(-np.log(softmax_outputs[[0, 1, 2], one_hot_targets]))
print(CategoricalCrossEntropyLoss().forward(softmax_outputs, one_hot_targets))
"""