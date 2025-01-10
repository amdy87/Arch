# Basic neural network architecture to solve the xor problem

import math
import random
import numpy as np


class activation:
    def __init__(self):
        pass

    def forward(self, X):
        self.input = X
        return np.tanh(X)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,1-np.tanh(self.input)**2) 


class Dense():
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize, 1)

    def forward(self, x):
        self.input = x
        out = np.dot(self.weights, x) + self.bias
        return out

    def backward(self,output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient,self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)

class Network():
    def __init__(self, network, loss_func=None):
        self.network = network
        self.loss_func = loss_func

    def train(self, epochs, learning_rate, X,Y, logError=True):
        for e in range(epochs):

            error = 0
        
            for x,y in zip(X,Y):
                output = self.test(x)

                error += self.errorCalc(y,output)
                
                grad = self.errorCalcPrime(y,output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(X)
            if(logError and e%100 == 0):
                print('%d/%d, error=%f' % (e+1, epochs, error))
            

    def test(self, x, showOutput=False):

        #runs the network through the feed forward algorithm
        output = x
        for layer in self.network:
            output = layer.forward(output)
        if(showOutput):
            print(output)
        return output

    def errorCalc(self, y, output):
        #mean squared error
        return np.mean(np.power(y-output,2))
        

    def errorCalcPrime(self, y, output):
        return 2 * (output-y) / np.size(y)
        


def main():
    network = [
        Dense(2,2),
        activation(),
        Dense(2,1),
        activation()
        ]

    brain = Network(network)

    X = np.reshape([[0,0],[1,0],[0,1],[1,1]],(4,2,1))
    Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

    learning_rate = 0.1
    epochs = 20000
    brain.train(epochs, learning_rate, X,Y)
    
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
