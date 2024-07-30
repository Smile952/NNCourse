import numpy as np


in_wgt = np.array([
                   [0.1, 0.2, -0.1],
                   [-0.1, 0.1, 0.9],
                   [0.1, 0.4, 0.1]]).T

hp_wgt = np.array([
                   [0.3, 1.1, -0.3],
                   [0.1, 0.2, 0],
                   [0, 1.3, 0.1]]).T

toes = np.array([8.5, 9.5, 9.9, 9])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1])

def neural_network():
    ans = 0
    for i in range(len(toes)):
        input = np.array([toes[i], wlrec[i], nfans[i]])
        pred_ans = input.dot(in_wgt)
        ans = pred_ans.dot(hp_wgt)
        print(ans)

neural_network()