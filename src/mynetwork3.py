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
from theano.tensor.nnet import sigmoid,relu
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import softmax
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

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
        self.inpt = inpt.reshape((inpt.shape[0], self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((inpt_dropout.shape[0], self.n_in)), self.p_dropout)
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

class SoftmaxLayer(FullyConnectedLayer):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.activation_fn=softmax
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]


    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])


#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),activation_fn=sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers


class Network(object):

    @staticmethod
    def load_model(path):
        f = open(path, 'rb')
        model=cPickle.load(f)
        f.close()
        return model
    
    def __init__(self, hidden_layers,output_layer):
        self.hidden_layers=hidden_layers
        self.output_layer=output_layer
        self.layers=self.hidden_layers+[self.output_layer]
        self.params = [param for layer in self.layers for param in layer.params]
        self.weights=[layer.params[0] for layer in self.layers]
        self.biases=[layer.params[1] for layer in self.layers]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        
    def save_model(self,save_path='data/save/network3.save'):
        f = open(save_path, 'wb')
        cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    def SGD(self, training_data,validation_data,test_data, epochs, mini_batch_size, eta,lmda=0.0,momentum=0.0):
        self.mini_batch_size=mini_batch_size
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
        cost = self.output_layer.cost(self)+\
               0.5*lmda*l2_norm_squared/size(training_data)
        grads_weights = T.grad(cost, self.weights)
        grads_biases = T.grad(cost,self.biases)
        velocities=[
            theano.shared(
                np.asarray(
                    np.zeros(w.get_value(borrow=True).shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            ) for w in self.weights
        ]
        
        updates = [
            (biase, biase-eta*grad) for biase, grad in zip(self.biases, grads_biases)
        ]+[
            (v, momentum*v-eta*grad) for v,grad in zip(velocities,grads_weights)
        ]+[
            (w,w+v) for w,v in zip(self.weights,velocities)
        ]
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                training_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        
        cost_train_mb = theano.function(
            [i],cost,givens={
                self.x:
                training_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                training_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        cost_validation_mb = theano.function(
            [i],cost,givens={
                self.x:
                validation_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                validation_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        cost_test_mb = theano.function(
            [i],cost,givens={
                self.x:
                test_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                test_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        
        train_accuracy_mb = theano.function(
            [i], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                training_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                training_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        test_accuracy_mb = theano.function(
            [i], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                test_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                test_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        validation_accuracy_mb = theano.function(
            [i], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                validation_x[i*mini_batch_size: (i+1)*mini_batch_size],
                self.y:
                validation_y[i*mini_batch_size: (i+1)*mini_batch_size]
            }
        )
        training_costs, validation_costs,test_costs = [], [],[]
        training_accuracies, validation_accuracies,test_accuracies = [], [],[]
        for epoch in range(epochs):
            tmp=[[[],[],[]],[[],[],[]]]
            for minibatch_index in range(num_training_batches):
                train_mb(minibatch_index)
                cost_train=cost_train_mb(minibatch_index)
                tmp[0][0].append(cost_train)
                acc_train=train_accuracy_mb(minibatch_index)
                tmp[1][0].append(acc_train)
                
                if mini_batch_size*minibatch_index<size(validation_data):
                    cost_val=cost_validation_mb(minibatch_index)
                    tmp[0][1].append(cost_val)
                    acc_val=validation_accuracy_mb(minibatch_index)
                    tmp[1][1].append(acc_val)
                
                if mini_batch_size*minibatch_index<size(test_data):
                    cost_test=cost_test_mb(minibatch_index)
                    tmp[0][2].append(cost_test)
                    acc_test=test_accuracy_mb(minibatch_index)
                    tmp[1][2].append(acc_test)
            
            if(len(test_accuracies)==0 or np.max(test_accuracies)<np.mean(tmp[1][2])):
                self.save_model()
                print("model is best by far.saving it...")
            
            training_costs.append(np.mean(tmp[0][0]))
            validation_costs.append(np.mean(tmp[0][1]))
            test_costs.append(np.mean(tmp[0][2]))
            
            training_accuracies.append(np.mean(tmp[1][0]))
            validation_accuracies.append(np.mean(tmp[1][1]))
            test_accuracies.append(np.mean(tmp[1][2]))
            
            print("This is epoch {}".format(epoch))
            
            
            
            print("accuracy on training data: {0:.2%}".format(training_accuracies[-1]))
            print("accuracy on validation data: {0:.2%}".format(validation_accuracies[-1]))
            print("accuracy on test data: {0:.2%}".format(test_accuracies[-1]))
            
            
            print("Cost on training data: {}".format(training_costs[-1]))
            print("Cost on validation data: {}".format(validation_costs[-1]))
            print("Cost on test data: {}".format(test_costs[-1]))
            
            
            print("Epoch {0} complete\n".format(epoch))        

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

    def predict(self,target_x):
        target_x=np.broadcast_to(target_x,(self.mini_batch_size,target_x.shape[0]))
        target_x=target_x.astype(theano.config.floatX)
        predict_mb=theano.function([],self.output_layer.y_out,givens={
            self.x:target_x
        })
        ret=predict_mb()
        return ret[0]
    def accuracy(self,target_data):
        target_x,target_y=target_data
        i = T.lscalar() # mini-batch index
        target_accuracy_mb = theano.function(
            [i], self.output_layer.accuracy(self.y),
            givens={
                self.x:
                target_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                target_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }
        )
        num_batches=size(target_data)/self.mini_batch_size
        return np.mean([target_accuracy_mb(j) for j in range(num_batches)])