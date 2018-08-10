# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:02:28 2018

@author: 周宝航
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    
    def __init__(self, sizes, num_iters=None, alpha=None, lam_bda=None):
        # layers numbers
        self.num_layers = len(sizes)
        # parameter sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # bias sizes
        self.bias = [np.zeros((y, 1)) for y in sizes[1:]]
        # iteration numbers
        self.num_iters = num_iters if num_iters else 100
        # learning rate
        self.alpha = alpha if alpha else 1
        # regularization
        self.lam_bda = lam_bda if lam_bda!=None else 0
        # logger
        self.logger = logging.getLogger()
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        
    def __sigmoid(self, z, derive=False):
        if derive:
            return self.__sigmoid(z) * (1.0 - self.__sigmoid(z))
        else:
            return 1.0 / (1.0 + np.exp(-z))
    
    def save(self, file_path):
        if file_path:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.weights, f)
    
    def load(self, file_path):
        if file_path:
            import pickle
            with open(file_path, 'rb') as f:
                self.weights = pickle.load(f)
    
    def forwardprop(self, X):
        activation = X
        activations = [X]
        zs = []
        cnt = 1
        for w, b in zip(self.weights, self.bias):
            z = w.dot(activation) + b
            zs.append(z)
            if cnt == self.num_layers-1:
                exp_z = np.exp(z)
                activation = exp_z / np.sum(exp_z)
            else:
                activation = self.__sigmoid(z)
            activations.append(activation)
        return (activations, zs)
    
    def costFunction(self, y, _y):
        m = y.shape[1]
        regularization = sum([np.sum(weight**2) for weight in self.weights])
        return -np.sum(y*np.log(_y) + (1-y)*np.log(1-_y)) / m + self.lam_bda / (2*m) * regularization

    def backprop(self, X, y):
        m = y.shape[1]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        nable_b = [np.zeros(b.shape) for b in self.bias]
        # forward propagation
        activations, zs = self.forwardprop(X)
        # cost
        # delta^(l) = a^(l) - y
        cost = activations[-1] - y
        # calc delta
        delta = cost * self.__sigmoid(zs[-1], derive=True)
        nable_b[-1] = delta
        nable_w[-1] = delta.dot(activations[-2].T)
        # back propagation
        for l in range(2, self.num_layers):
            # delta^(l) = weights^(l)^T delta^(l+1)
            delta = self.weights[-l+1].T.dot(delta) * self.__sigmoid(zs[-l], derive=True)
            nable_b[-l] = delta
            nable_w[-l] = delta.dot(activations[-l-1].T)
        
        # update bias, weights
        self.bias = [b-self.alpha/m*delta_b for b, delta_b in zip(self.bias, nable_b)]
        self.weights = [(1-self.lam_bda*self.alpha/m)*w-self.alpha*delta_w for w, delta_w in zip(self.weights, nable_w)]
        return activations[-1]
    
    def train_model(self, training_data, mini_batch_size=100, test_data=None):
        J_history = []
        m = len(training_data)
        for i in range(self.num_iters):
            cost = 0
            np.random.shuffle(training_data)
            for j in range(0, m, mini_batch_size):
                X = None
                y = None
                for k, batch in enumerate(training_data[j:j+mini_batch_size]):
                    inp, oup = batch
                    X = np.c_[X, inp] if k != 0 else inp
                    y = np.c_[y, oup] if k != 0 else oup
                _y = self.backprop(X, y)
                cost += self.costFunction(y, _y)
            cost /= (m / mini_batch_size)
            if i % (self.num_iters/10) == 0:
                if test_data:
                    acc = self.evaluate(test_data)
                    self.logger.info("epoch {0}/{1} acc : {2}".format(i, self.num_iters, acc))
                else:
                    self.logger.info("epoch {0}/{1} cost : {2}".format(i, self.num_iters, cost))
            J_history.append(cost)
        fig = plt.figure()
        ax_loss = fig.add_subplot(1,1,1)
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(0,self.num_iters)
        ax_loss.plot(J_history)
        plt.show()
    
    def predict(self, X):
        activations, _ = self.forwardprop(X)
        y = activations[-1]
        return np.where(y == np.max(y))[0][0]

    def evaluate(self, test_data):
        test_results = 0
        for (X, y) in test_data:
            if self.predict(X) == y:
                test_results += 1
        return test_results / len(test_data)