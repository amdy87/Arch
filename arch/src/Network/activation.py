#activation layer
import numpy as np

class Activation():
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Tanh(Activation):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.tanh(input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, 1-np.tanh(self.input)**2)
