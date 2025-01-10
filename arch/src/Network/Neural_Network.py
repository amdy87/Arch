#Neural network base class

import numpy as np

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
                print('%d/%d, error=%f' % (e, epochs, error))
            

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
