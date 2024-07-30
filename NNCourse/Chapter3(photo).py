import numpy as np
import cv2 as cv
import matplotlib.pyplot as ppl
import random as rnd
import os

input = 625
amount_pictures = 6
weights = np.random.rand(input)
data_input = np.zeros(input)
alpha = 0.5
path = 'C:\\Dvelopment\\Python\\snipets'
num_epochs = 10

def to_byte(x):
    return 0 if x < 0 else 1

def get_name_list():
    global names
    names = np.ravel(os.listdir(os.chdir(path+'\\training')))

def read_weights():
    os.chdir(path)
    f = open('C:\\Dvelopment\\Python\\snipets\\weights\\weeights.txt', 'r')
    text = f.read().split(',')
    weights = np.array(text[: len(text)-1], float)
    return weights

def write_weights():
    os.chdir(path)
    f = open('C:\\Dvelopment\\Python\\snipets\\weights\\weeights.txt', 'w')
    global weights
    for i in weights:
        f.write(str(i)+',')

def read_data(directory, input_picture):
    path_i = directory +str(input_picture)
    img = np.ravel(np.array(cv.imread(path_i, cv.IMREAD_GRAYSCALE)))
    for i in range(img[0]):
        img[i] = 0 if img[i] <=128 else 1
    return img

def forward(data):
    global weights
    weights = read_weights()
    answer = np.dot(data, weights)
    return answer

def backward(data, pred, answer_byte):
    global weights
    error = pred - answer_byte
    weights -= data*error*alpha
    write_weights()


def start():
    global alpha
    get_name_list()
    for _ in range(num_epochs):
        for i in names:
            answer_word = i[:len(i)-5]
            answer_byte = 0 if answer_word == 'dot' else 1
            data = read_data('C:\\Dvelopment\\Python\\snipets\\training\\', i)
            pred = to_byte(forward(data))
            if pred != answer_byte:
                backward(data, pred, answer_byte)
                pred = to_byte(forward(data))
            print(str(answer_word) + ': ' + str(pred))

def test():
    test_names = np.ravel(os.listdir(os.chdir((path+'\\tasting'))))
    for n in test_names:
        answer_word = n[:len(n)-4]
        pred = to_byte(forward(read_data('C:\\Dvelopment\\Python\\snipets\\tasting\\', n)))
        print(str(n) + ': ' + str(answer_word) + ' ' + str(pred))

write_weights()
start()
test()
        