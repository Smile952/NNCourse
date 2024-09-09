import numpy as np
import os
import cv2 as cv
import threading

INPUT = 784
HIDDEN = 140
OUTPUT = 10

weights = [np.random.rand(INPUT, HIDDEN), np.random.rand(HIDDEN, OUTPUT)]


num_epochs = 200
alpha = 0.1

dir_names = []
file_names = []

error = 0

path_to_weights = 'C:\\Dvelopment\\Python\\snipets\\Chapter5(snipets)\\weights.txt'
path_to_test = 'C:\\Dvelopment\\Python\\snipets\\Chapter5(snipets)\\test'
path_to_train = 'C:\\Dvelopment\\Python\\snipets\\Chapter5(snipets)\\train'

def sigmoid(arr):
    arr = 1/(1+np.exp(-arr))
    return arr

def sigm_deriv(arr):
    arr = sigmoid(arr)*(1-sigmoid(arr))
    return arr

def to_byte(arr):
    max = 0
    for i in range(len(arr)):
        max = i if arr[i] > arr[max] else max
    arr = np.zeros(len(arr))
    arr[max] = 1
    return arr

def read_weights():
    f = open(path_to_weights, 'r')
    text = f.read().split(',')
    text = text[:len(text)-1]
    iterator = 0
    for w in range(len(weights)):
        for i in range(len(weights[w])):
            for j in range(len(weights[w][i])):
                weights[w][i][j] = float(text[iterator])
                iterator += 1

def write_weights():
    with open(path_to_weights, 'w') as f:
        for w in range(len(weights)):
            for i in range(len(weights[w])):
                for j in range(len(weights[w][i])):
                    f.write(str(weights[w][i][j])+',')
        
def get_dir_names(path):
    global dir_names
    dir_names = os.listdir(path)
    return dir_names

def get_names_list(dir_name, path):
    global file_names
    file_names = os.listdir(path + '\\' + dir_name)
    return file_names

def read_data(path):
    img = np.ravel(np.array(cv.imread(path, cv.IMREAD_GRAYSCALE)))
    for i in range(INPUT):
        img[i] = 1 if img[i] > 128 else 0
    return img

def forward(data):
    global weights
    layer1 = sigmoid(np.dot(data, weights[0]))
    layer2 = to_byte(np.dot(layer1, weights[1]))
    return layer2, [layer1, layer2]

def backward(layers, input, ans_arr, pred):
    global weights

    layer1_delta = (pred - ans_arr)

    layer0_delta = layer1_delta.dot(weights[1].T)
    layer0_delta *= sigm_deriv(layers[0])

    weights12_delta = (np.array(layers[0])[np.newaxis].T.dot(layer1_delta[np.newaxis]))
    weights01_delta = (np.array(input[np.newaxis]).T.dot(layer0_delta[np.newaxis]))

    weights[1]-=weights12_delta*alpha
    weights[0]-=weights01_delta*alpha  

def train():
    global alpha
    get_dir_names(path_to_train)
    for ep in range(num_epochs):
        for d in dir_names:
            get_names_list(d, path_to_train)
            for n in file_names:
                answer = np.zeros(10)
                answer[int(d)] = 1
                data = read_data(path_to_train + '\\' + d + '\\' + n)
                pred, layers = forward(data)
                backward(layers, data, answer, pred)
        
    #write_weights()

def max_index(arr):
    max = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            max = i
            return max

def test():
    global error
    #read_weights()
    get_dir_names(path_to_test)
    for d in dir_names:
        get_names_list(d, path_to_test)
        for n in file_names:
            answer = np.zeros(10)
            answer[int(d)] = 1
            data = read_data(path_to_test + '\\' + d + '\\' + n)
            pred, layers = forward(data)
            if max_index(pred) != max_index(answer):
                error+=1
            #print(str(max_index(pred) + ' ' + str(max_index(answer)))
    print(error, alpha, num_epochs)
    error = 0


'''write_weights()
train()
test()'''
def start_with_parametrs():
    for i in range(5):
        train()
        test()
        weights = [np.random.rand(INPUT, HIDDEN), np.random.rand(HIDDEN, OUTPUT)]
start_with_parametrs()