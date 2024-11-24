import random
import sys
# Third-party libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import theano.tensor as T
import theano
import gzip
import pickle as cPickle
from theano.tensor.nnet import sigmoid
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import softmax
#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True.")

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)



class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout):
        self.inpt = inpt
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout, self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout):
        self.inpt = inpt
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout, self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


class CrossEntropyLayer(FullyConnectedLayer):
    def cost(self,net):
        "Return the cross-entropy cost."
        ce_y=T.zeros(self.output_dropout.shape)
        ce_y=T.set_subtensor(ce_y[T.arange(net.y.shape[0]),net.y],1)
        ret=-T.mean(
            T.sum(
                ce_y*T.log(self.output_dropout)+
                (1-ce_y)*T.log(1-self.output_dropout),
                axis=1
            )
        )
        return ret 


class Network(object):

    def __init__(self, hidden_layers,output_layer):
        self.hidden_layers=hidden_layers
        self.output_layer=output_layer
        self.layers=self.hidden_layers+[self.output_layer]
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")

    def SGD(self, training_data,validation_data,test_data, epochs, mini_batch_size, eta,lmda=0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        init_layer = self.hidden_layers[0]
        init_layer.set_inpt(self.x, self.x)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout)
        self.output = self.output_layer.output
        self.output_dropout = self.output_layer.output_dropout
        
        num_training_batches = size(training_data)/mini_batch_size
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        
        
        
        i = T.lscalar() # mini-batch index
        length=T.iscalar()
        cost = self.output_layer.cost(self)+\
               0.5*lmda*l2_norm_squared/length
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]
        train_mb = theano.function(
            [i,length], cost, updates=updates,
            givens={
                self.x:
                training_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                training_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        
        cost_train_mb = theano.function(
            [length],cost,givens={
                self.x:training_x,
                self.y:training_y
            }
        )
        cost_validation_mb = theano.function(
            [length],cost,givens={
                self.x:validation_x,
                self.y:validation_y
            }
        )
        cost_test_mb = theano.function(
            [length],cost,givens={
                self.x:test_x,
                self.y:test_y
            }
        )
        
        train_accuracy_mb = theano.function(
            [], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                training_x,
                self.y:
                training_y
            }
        )
        test_accuracy_mb = theano.function(
            [], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                test_x,
                self.y:
                test_y
            }
        )
        validation_accuracy_mb = theano.function(
            [], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                validation_x,
                self.y:
                validation_y
            }
        )
        training_costs, validation_costs,test_costs = [], [],[]
        training_accuracies, validation_accuracies,test_accuracies = [], [],[]
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                train_mb(minibatch_index,size(training_data))
                
            
            cost_train=cost_train_mb(size(training_data))
            cost_val=cost_validation_mb(size(validation_data))
            cost_test=cost_test_mb(size(test_data))
            
            training_costs.append(cost_train)
            validation_costs.append(cost_val)
            test_costs.append(cost_test)
            
            acc_train=train_accuracy_mb()
            acc_val=validation_accuracy_mb()
            acc_test=test_accuracy_mb()
            
            training_accuracies.append(acc_train)
            validation_accuracies.append(acc_val)
            test_accuracies.append(acc_test)
            
            print("This is epoch {}".format(epoch))
            
            
            
            print("accuracy on training data: {}".format(acc_train))
            print("accuracy on validation data: {}".format(acc_val))
            print("accuracy on test data: {}".format(acc_test))
            
            
            print("Cost on training data: {}".format(cost_train))
            print("Cost on validation data: {}".format(cost_val))
            print("Cost on test data: {}".format(cost_test))
            
            
            print("Epoch {0} complete\n".format(j))        

        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), training_costs,color='blue',marker='o')
        plt.plot(range(epochs), validation_costs,color='red',marker='+')
        plt.plot(range(epochs), test_costs,color='green',marker='^')
        plt.legend(["training cost", "validation cost","test cost"])
        plt.title("cost function curve")
        plt.xlabel("epoch")
        plt.ylabel("cost")

        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), training_accuracies,color='blue',marker='o')
        plt.plot(range(epochs), validation_accuracies,color='red',marker='+')
        plt.plot(range(epochs), test_accuracies,color='green',marker='^')
        plt.legend(["training accuracy", "validation accuracy","test accuracy"])
        plt.title("accuracy function curve")
        plt.xlabel("epoch")
        plt.ylabel("accuracy curve")
        plt.show()

