import numpy as np
import matplotlib.pyplot as mpl

def tanh_deriv(x):
    return 1-(np.tanh(x)**2)

INPUT = 3
HIDDEN = 5
OUTPUT = 1

weights0 = np.random.rand(INPUT, HIDDEN)
weights1 = np.random.rand(HIDDEN, OUTPUT)

num_try = 0
error = np.zeros([])

input = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])
answer = np.array([1, 1, 0, 0])

alpha = 0.2

def backward(layers, i):
    global weights0, weights1
    layer2_delta = layers[2] - answer[i:i+1]
    layer1_delta = layer2_delta.dot(weights1.T)*tanh_deriv(layers[1])
    weights1-=alpha*layers[1].T.dot(layer2_delta)
    weights0-=alpha*layers[0].T.dot(layer1_delta)

def forward():
    global weights0, weights1, error, num_try
    for i in range(len(input)):
        layer0 = input[i:i+1]
        layer1 = np.tanh(layer0.dot(weights0))
        layer2 = layer1.dot(weights1)
        gl_error = layer2-answer[i:i+1]
        np.append(error, gl_error**2)
        num_try+=1
        while (gl_error >alpha).all():
            backward((layer0, layer1, layer2), i)
            layer0 = input[i:i+1]
            layer1 = np.tanh(layer0.dot(weights0))
            layer2 = layer1.dot(weights1)
            gl_error = layer2-answer
            np.append(error, gl_error**2)
            num_try+=1

forward()

trys = np.zeros([num_try])
for t in range(len(trys)):
    trys[t] = t+1

mpl.xlabel('num trys')
mpl.ylabel('error')
mpl.plot(trys, error)
mpl.show()