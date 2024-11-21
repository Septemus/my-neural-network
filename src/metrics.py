import pandas as pd
from train import my_transform
import network
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat

if __name__=="__main__":
    data = loadmat('data/mnist-original.mat')['data'].transpose()/255.0
    label = loadmat('data/mnist-original.mat')['label'].transpose()
    data=data.tolist()
    for i in range(len(data)):
        data[i]=np.array(data[i])
        tmp=np.zeros((10,1))
        tmp[int(label[i][0])]=np.array([1])
        data[i]=[np.reshape(data[i],(784,1)),tmp]

    my_testdata=data[60000:]
    
    model = network.load("data/save/model.json")
    test_results = [(np.argmax(model.feedforward(x)), np.argmax(y)) for (x, y) in my_testdata]
    test_results=np.array(test_results)
    confusion_matrix = metrics.confusion_matrix(test_results[:,1], test_results[:,0])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    
    
        
    