# Basic neural network architecture to solve the xor problem

import math
import random
import numpy as np



class Dense():
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize, 1)

    def forward(self, x):
        out = np.dot(self.weights, x) + self.bias
        print(out)
        return out

    def backward(self):
        pass


class Network():
    def __init__(self, network):
        self.network = network

    def train(self, X,Y):
        pass

    def test(self, x):
        output = x
        for layer in self.network:
            output = layer.forward(output)

        return output
        


def main():
    network = [
        Dense(2,2),
        Dense(2,1)
        ]

    brain = Network(network)
    while(True):

        x1 = int(input("Enter the first value: "))
        x2 = int(input("Enter the second value: "))
        x = np.matrix([x1,x2]).T

        ans = brain.test(x)

        if(ans > 0.5):
            print("True")
        else:
            print("False")


main()
