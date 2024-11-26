import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import time
import pandas as pd
import softmax
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    data = loadmat('data/mnist-original.mat')['data'].transpose()/255.0
    label = loadmat('data/mnist-original.mat')['label'].transpose()
    data=data.tolist()
    for i in range(len(data)):
        data[i]=np.array(data[i])
        tmp=np.zeros((10,1))
        tmp[int(label[i][0])]=np.array([1])
        data[i]=[np.reshape(data[i],(784,1)),tmp]

    my_testdata=data[60000:]

    net = network.Network(sizes=(784,150, 10), cost=network.crossEntropy)
    net.SGD(data[:60000], 45, 50, 1, evaluationData=my_testdata, lmda=1, momentum_coefficient=0.2,
            monitor_training_cost=False, monitor_training_accuracy=False)
