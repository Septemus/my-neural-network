import numpy as np
import sigmoid
class quadratic(object):
    @staticmethod
    def fn(a,y):
        return 0.5*(np.linalg.norm(a-y)**2)
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)*sigmoid.sigmoid_prime(z)