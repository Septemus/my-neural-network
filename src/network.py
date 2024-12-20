"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Libraries
# Standard library
import random
import sys
# Third-party libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import json

# Miscellaneous functions
class quadratic(object):
    @staticmethod
    def fn(a,y):
        return 0.5*(np.linalg.norm(a-y)**2)
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)*sigmoid.sigmoid_prime(z)

class softmax(object):
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


class crossEntropy(object):
    @staticmethod
    def fn(a,y):
        return -np.sum(np.nan_to_num(y*(np.log(a))+(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z,a,y):
        return a-y

def vectorized_result(j,dimension=10):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((dimension, 1))
    e[j] = 1.0
    return e


class Network(object):

    def __init__(self, sizes,cost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.cost=cost
        self.sizes = sizes
        self.preprocessing = 0
        self.num_layers = len(sizes)
        self.weight_initializer()
        self.velocity = [np.zeros(w.shape) for w in self.weights]

    def weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        count=0
        a = a.reshape(len(a), 1)
        for b, w in zip(self.biases, self.weights):
            count+=1
            z=np.dot(w, a)+b
            if self.cost.__name__=="softmax" and count==self.num_layers-1:
                a=self.cost.output(z)
            else:
                a = sigmoid(z)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmda=0.0,
            momentum_coefficient=0.0,
            evaluationData=None, monitor_evaluation_cost=True, monitor_training_cost=True, monitor_evaluation_accuracy=True, monitor_training_accuracy=True):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        # fig=plt.figure()
        eta_decay_count=0
        if evaluationData:
            n_test = len(evaluationData)
        n = len(training_data)
        training_cost, evaluation_cost = [], []
        training_accuracy, evaluation_accuracy = [], []
        for j in range(epochs):
            start_time = time.time()
            for i in range(len(training_data)):
                training_data[i] = [training_data[i]
                                    [0].flatten(), training_data[i][1]]
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # fig.add_subplot(1,2,1)
            # plt.imshow(mini_batches[15][8][0].reshape(28,28), cmap=plt.get_cmap('gray'))
            # plt.title(mini_batches[15][8][1])
            # plt.show()
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, n, lmda, momentum_coefficient)
            if monitor_evaluation_cost and evaluationData:
                cost = self.total_cost(evaluationData, (lmda*len(evaluationData))/len(training_data))
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_evaluation_accuracy and evaluationData:
                correct = self.evaluate(evaluationData)
                acc = correct/(n_test*1.0)
                acc *= 100
                evaluation_accuracy.append(acc)
                print(
                    "Accuracy on evaluation data: {0} / {1} ({2}%)".format(correct, n_test, acc))
            if monitor_training_accuracy:
                correct = self.evaluate(training_data)
                acc = correct/(n*1.0)
                acc *= 100
                training_accuracy.append(acc)
                print(
                    "Accuracy on training data: {0} / {1} ({2}%)".format(correct, n, acc))
            end_time = time.time()
            elapsed_time = end_time - start_time

            print("Epoch Elapsed time: {0} seconds\nPreprocessing time: {1} seconds\n".format(
                elapsed_time, self.preprocessing))
            print("Epoch {0} complete\n".format(j))
            self.preprocessing = 0
            # np.save("data/save/weights.npy", self.weights)
            # np.save("data/save/biases.npy", self.biases)
            self.save("data/save/model.json")
            # if(monitor_evaluation_accuracy and evaluation_accuracy and j>5):
            #     if(evaluation_accuracy[j]<evaluation_accuracy[j-5] and eta_decay_count<10):
            #         eta/=2.0
            #         eta_decay_count+=1
                    

        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), training_cost,color='blue',marker='o')
        plt.plot(range(epochs), evaluation_cost,color='red',marker='+')
        plt.legend(["training cost", "evaluation cost"])
        plt.title("cost function curve")
        plt.xlabel("epoch")
        plt.ylabel("cost")

        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), training_accuracy,color='blue',marker='o')
        plt.plot(range(epochs), evaluation_accuracy,color='red',marker='+')
        plt.legend(["training accuracy", "evaluation accuracy"])
        plt.title("accuracy function curve")
        plt.xlabel("epoch")
        plt.ylabel("accuracy curve")
        plt.show()

    def update_mini_batch(self, mini_batch, eta, n=1, lmda=0.0, momentum_coefficient=0.0):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # nabla_b = [np.zeros(b.shape) for b in self.biases]
        # nabla_w = [np.zeros(w.shape) for w in self.weights]
        # for x, y in mini_batch:
        #     delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        #     nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        nabla_b, nabla_w = self.my_backprop(mini_batch)
        self.velocity = [momentum_coefficient*v-(eta/len(mini_batch))*nw
                         for v, nw in zip(self.velocity, nabla_w)]
        self.weights = [(1.0-(eta*lmda)/n)*w + v
                        for w, v in zip(self.weights, self.velocity)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # def backprop(self, x, y):
    #     """Return a tuple ``(nabla_b, nabla_w)`` representing the
    #     gradient for the cost function C_x.  ``nabla_b`` and
    #     ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    #     to ``self.biases`` and ``self.weights``."""
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     # feedforward
    #     activation = x
    #     activations = [x]  # list to store all the activations, layer by layer
    #     zs = []  # list to store all the z vectors, layer by layer
    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation)+b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)
    #     # backward pass
    #     delta = self.cost_derivative(activations[-1], y) * \
    #         sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #     # Note that the variable l in the loop below is used a little
    #     # differently to the notation in Chapter 2 of the book.  Here,
    #     # l = 1 means the last layer of neurons, l = 2 is the
    #     # second-last layer, and so on.  It's a renumbering of the
    #     # scheme in the book, used here to take advantage of the fact
    #     # that Python can use negative indices in lists.
    #     for l in range(2, self.num_layers):
    #         z = zs[-l]
    #         sp = sigmoid_prime(z)
    #         delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    #     return (nabla_b, nabla_w)

    def my_backprop(self, mini_batch):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        start_time = time.time()
        x = [item[0] for item in mini_batch]
        y = [item[1] for item in mini_batch]
        # for i in range(len(x)):
        #     x[i]=x[i].tolist()
        #     x[i]=[item[0] for item in x[i]]
        for i in range(len(y)):
            y[i] = [item[0] for item in y[i]]
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.preprocessing += elapsed_time
        x = np.array(x)
        y = np.array(y)
        x = x.transpose()
        y = y.transpose()

        # print("mini-batch preprocessing time Elapsed time: {0} seconds".format(elapsed_time))

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        if self.cost.__name__=="softmax":
            activations[-1]=self.cost.output(zs[-1])
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        try:
            ret = sum(int(x == y) for (x, y) in test_results)
        except TypeError:
            ret = sum(int(x == np.argmax(y)) for (x, y) in test_results)
        return ret

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def total_cost(self, target_data, lmda=0.0):
        ret = 0.0
        for x, y in target_data:
            a = self.feedforward(x)
            try:
                ret += (self.cost.fn(a, y)/len(target_data))
            except TypeError:
                y = vectorized_result(y,self.sizes[-1])
                ret += self.cost.fn(a, y)/len(target_data)
        ret += 0.5*(lmda/len(target_data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return ret

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "velocity": [v.tolist() for v in self.velocity],
                "cost": str(self.cost.__name__)
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        print("save successful!\n")

# Miscellaneous functions


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# Loading a Network


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"],cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    net.velocity = [np.array(v) for v in data["velocity"]]
    print("load successful!\n")
    return net
