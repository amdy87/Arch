#Layer parent class for all layer types

class Layer():
    def __init__(self):
        pass


    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass
