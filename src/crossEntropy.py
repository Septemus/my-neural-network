import numpy as np
class crossEntropy(object):
    @staticmethod
    def fn(a,y):
        return -np.sum(y*(np.log(a))+(1-y)*np.log(1-a))
    
    @staticmethod
    def delta(z,a,y):
        return a-y