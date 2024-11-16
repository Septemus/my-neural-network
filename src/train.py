import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import time
import crossEntropy

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
    
    # training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
    net = network.Network(sizes=(784,50,10),cost=crossEntropy.crossEntropy)
    # net=network.load("data/save/model.json")
    # all_data=training_data+data[:60000]
    all_data=data
    # all_testdata=test_data+my_testdata
    net.SGD(all_data[:60000] , 120, 100, 0.7, evaluationData=data[60000:],lmda=0.5,momentum_coefficient=0.5)
    