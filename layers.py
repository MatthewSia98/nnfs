import numpy as np

class DenseLayer:
    MIN_START_WEIGHT = -1
    MAX_START_WEIGHT = 1
    MIN_START_BIAS = 0
    MAX_START_BIAS = 0

    def __init__(self, prev_n_neurons, n_neurons):
        #self.weights = np.random.uniform(DenseLayer.MIN_START_WEIGHT, DenseLayer.MAX_START_WEIGHT, (prev_n_neurons, n_neurons))
        #self.biases = np.random.uniform(DenseLayer.MIN_START_BIAS, DenseLayer.MAX_START_BIAS, (1, n_neurons))
        self.weights = 0.01 * np.random.randn(prev_n_neurons, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases       
        return self.output 
