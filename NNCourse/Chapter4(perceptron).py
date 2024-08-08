import numpy as np
import cv2 as cv
import os

num_inputs = 625
num_outputs = 5
alpha = 0.5
num_epochs = 10

weights = np.random.rand(num_inputs, num_outputs)

input = np.zeros(num_inputs)

path_to_weights = 'C:\\Dvelopment\\Python\\snipets\\Chapter4(snipets)\\weights.txt'
path_to_train_data = 'C:\\Dvelopment\\Python\\snipets\\Chapter4(snipets)\\trainning'
path_to_test_data = 'C:\\Dvelopment\\Python\\snipets\\Chapter4(snipets)\\testing'

names = []

def read_weights():
    global weights
    f = open(path_to_weights, 'r')
    text = f.read().split(',')
    text = text[: len(text)-1]
    iterator = 0
    for i in range(num_inputs):
        for j in range(num_outputs):
            weights[i][j] = float(text[iterator])
            iterator+=1
    return weights

def write_weights():
    f = open(path_to_weights, 'w')
    for w in weights:
        for k in w:
            f.write(str(k) + ',')

def read_data(directory, input_picture):
    path_i = directory+'\\'+str(input_picture)
    img = np.ravel(cv.imread(path_i, cv.IMREAD_GRAYSCALE))
    for i in range(num_inputs):
        img[i] = 0 if img[i] <=128 else 1
    return img

def get_name_list(path):
    global names
    names = np.ravel(np.array(os.listdir(path)))

def to_byte(p):
    out = np.zeros(len(p))
    for i in range(len(p)):
        out[i] = 1 if p[i] > 0 else 0
    return out

def forward(data):
    global weights
    weights = read_weights()
    pred = np.dot(data, weights)
    return pred

def backward(pred, answer, input):
    global alpha, weights
    error = np.array(pred - answer)[np.newaxis]
    input = np.array(input)[np.newaxis]
    weights_delta = input.T.dot(error)
    weights-=weights_delta*alpha
    write_weights()

def conversation(pred, answer):
    out = 0
    for i in range(len(pred)):
        if pred[i] == answer[i]:
            out+=1
    return True if out == len(pred) else False

def train():
    get_name_list(path_to_train_data)
    for _ in range(num_epochs):
        for i in names:
            answer_index = i[:1]
            answer_array = np.zeros(num_outputs)
            answer_array[int(answer_index)] = 1
            data = read_data(path_to_train_data, i)
            pred = to_byte(forward(data))
            
            backward(pred, answer_array, data)
            pred = to_byte(forward(data))
            print(pred)

def max_index(arr):
    max = 0
    for i in range(len(arr)):
        if arr[i] > arr[max]:
            max = i
    return max

def test():
    classes = ['iena', 'fire', 'tree', 'service', 'north']
    get_name_list(path_to_test_data)
    for i in names:
        data = read_data(path_to_test_data, i)
        pred = forward(data)
        pred = to_byte(pred)
        answer_index = max_index(pred)
        print(classes[answer_index])
        
train()

test()