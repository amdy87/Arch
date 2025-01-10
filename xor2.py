#solves the xor problem by using the arch external library package
#for more general problem solving

import arch
import numpy as np


network = [
    arch.Dense(2,2),
    arch.Tanh(),
    arch.Dense(2,1),
    arch.Tanh()
    ]

brain = arch.Network(network)

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
