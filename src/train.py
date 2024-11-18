import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import time
import pandas as pd
import crossEntropy
from sklearn.preprocessing import MinMaxScaler

def my_transform(mydata):
    mydata=mydata.to_numpy()
    x=mydata[:,:7]
    scaler = MinMaxScaler()
    x=scaler.fit_transform(x)
    y=mydata[:,7]
    y=y.reshape(len(y),1)
    x=x.tolist()
    for i in range(len(x)):
        x[i]=np.array(x[i])
        tmp=np.zeros((15,1))
        tmp[int(y[i][0])]=np.array([1])
        x[i]=[np.reshape(x[i],(7,1)),tmp]
    return x

if __name__=="__main__":
    # data = loadmat('data/mnist-original.mat')['data'].transpose()/255.0
    # label = loadmat('data/mnist-original.mat')['label'].transpose()
    # data=data.tolist()
    # for i in range(len(data)):
    #     data[i]=np.array(data[i])
    #     tmp=np.zeros((10,1))
    #     tmp[int(label[i][0])]=np.array([1])
    #     data[i]=[np.reshape(data[i],(784,1)),tmp]
        
    # my_testdata=data[60000:]
    
    # training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
    mydata=pd.read_csv("data/real_train.csv")
    my_test_data=pd.read_csv("data/real_test.csv")
    x=my_transform(mydata)
    test=my_transform(my_test_data)
    net = network.Network(sizes=(7,150,15),cost=crossEntropy.crossEntropy)
    # net=network.load("data/save/model.json")
    # all_data=training_data+data[:60000]
    # all_data=data
    # all_testdata=test_data+my_testdata
    net.SGD(x , 45, 1000, 0.3, evaluationData=test,lmda=5,momentum_coefficient=0.5)
    