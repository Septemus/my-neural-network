import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import time

if __name__=="__main__":
    data = loadmat('data/mnist-original.mat')['data'].transpose()/255.0
    label = loadmat('data/mnist-original.mat')['label'].transpose()
    data=data.tolist()
    for i in range(len(data)):
        data[i]=np.array(data[i])
        tmp=np.zeros((10,1))
        tmp[int(label[i][0])]=np.array([1])
        data[i]=[np.reshape(data[i],(784,1)),tmp]
        
    # my_testdata=data[60000:]
    
    training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
    net = network.Network(saves=("data/save/weights.npy","data/save/biases.npy"))
    all_data=training_data+data[:60000]
    # all_testdata=test_data+my_testdata
    net.SGD(all_data , 300, 60, 0.01, test_data=test_data)
    