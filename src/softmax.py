import numpy as np
class softmax(object):
    method="softmax"
    @staticmethod
    def fn(a,y):
        tmp=np.argmax(y)
        return -np.sum(np.nan_to_num(np.log(a[tmp])))
    
    @staticmethod
    def delta(z,a,y):
        return a-y
    
    @staticmethod
    def output(z_last):
        z_exp=np.exp(z_last)
        sum=np.sum(z_exp,axis=0)
        return z_exp*(1./sum)