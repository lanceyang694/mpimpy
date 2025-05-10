'''
This demo is revised based on the original code from https://github.com/wchstu/Deep_Learning_Coding
'''

import numpy as np
from datasets import mnist
import matplotlib.pyplot as plt
from mpimpy import memmatdp

# dpe_int = memmat.bitslicedpe(HGS=1/1.3e5, LGS=1/2.23e6, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93, 
#                              rdac=256, radc=256, vread=0.1, array_size=(32, 32))

dpe_dp = memmatdp.diffpairdpe(HGS=1/1.3e5, LGS=1/2.23e6, g_level=32, var=0.05, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

class MLP:
    
    eps = 1e-8
    switcher = {
        'sigmoid': lambda x: 1.0 /(1.0+np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.maximum(0.01*x, x),
        'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    }
    
    def __init__(self, size=(784, 100, 10), lr=0.01, lr_ratio=0.9, act_func='relu'):
        
        self.lr = lr
        self.lr_ratio = lr_ratio
        self.weights = []
        self.bias = []
        
        try:
            self.act_func = self.switcher[act_func]
        except KeyError:
            raise ValueError('Activation function not supported. Please choose from: sigmoid, tanh, relu, leaky_relu, softmax.')
        
        assert isinstance(size, list) or isinstance(size, tuple), 'size must be a list or tuple.'
        assert len(size) > 1, 'the length of size must be greater than 1.'

        for i in range(1, len(size)):
            self.weights.append(np.random.normal(0, 0.01, (size[i-1], size[i])))
            self.bias.append(np.random.normal(0, 0.01, (1, size[i])))

    def __call__(self, X, Y):
        
        self.target = Y
        self.z = []
        self.a = [X]
        
        for w, b in zip(self.weights, self.bias):
            # self.z.append(np.dot(self.a[-1], w) + b)
            self.z.append(dpe_dp.MapReduceDot(self.a[-1], w) + b)
            self.a.append(self.act_func(self.z[-1]))
        self.a[-1] = self.switcher['softmax'](self.z[-1])

        loss = -np.sum(Y * np.log(self.a[-1])) / len(Y)

        return loss

    def backward(self):
        
        W_grad = []
        b_grad = []
        delta = (self.a[-1] - self.target) / len(self.target)
        for i in range(len(self.weights)-1, -1, -1):
            W_grad.insert(0, np.dot(self.a[i].T, delta))
            b_grad.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i == 0:
                break
            # delta = np.dot(delta, (self.weights[i].T)) * (self.z[i-1] > 0.0) # * self.a[i] * (1.0 - self.a[i])
            delta = dpe_dp.MapReduceDot(delta, (self.weights[i].T)) * (self.z[i-1] > 0.0)

        # update trained parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * W_grad[i]
            self.bias[i] -= self.lr * b_grad[i]

    def predict(self, X):
        
        a = X
        for i in range(len(self.weights)):
            z = dpe_dp.MapReduceDot(a, self.weights[i]) + self.bias[i]
            a = self.act_func(z)

        return self.switcher['softmax'](z)

    def validate(self, X, Y):
        
        pred = np.argmax(self.predict(X), axis=1)
        org = np.argmax(Y, axis=1)
        acc = sum(pred == org) / len(pred)
        
        return acc

    def lr_step(self):
        self.lr *= self.lr_ratio
        
    def fit(self, X, Y, batch_size, epochs):
        
        acc_s = []
        loss_s = []
        
        for epoch in range(epochs):
            indexs = np.arange(len(X))
            steps = len(X) // batch_size
            np.random.shuffle(indexs)
            loss_i = []
            acc_i = []
            
            for i in range(steps):
                ind = indexs[i*batch_size:(i+1)*batch_size]
                x = X[ind]
                y = Y[ind]
                loss = self(x, y)
                self.backward()
                loss_i.append(loss)
                
                if (i + 1) % 100 == 0:
                    accuracy = self.validate(X, Y)
                    acc_i.append(accuracy)
                    print('Epoch[{}/{}] \t Step[{}/{}] \t Loss = {:.6f} \t Acc = {:.3f}'.format(epoch+1, epochs, i+1, steps, loss, accuracy))
                    
            acc_s.append(np.mean(acc_i))
            loss_s.append(np.mean(loss_i))
            model.lr_step()
            
        print('Test Accuracy = ', model.validate(test_set[0], test_set[1]))
        
        return acc_s, loss_s


if __name__ == '__main__':
    
    epochs = 8
    batch_size = 64
    train_set, valid_set, test_set = mnist(r'data\mnist.pkl.gz', one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]

    model = MLP([784, 100, 10])
    acc_s, loss_s = model.fit(X_train, Y_train, batch_size, epochs)
    
    plt.plot(range(epochs), acc_s)
    # plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    plt.plot(range(epochs), loss_s, label='Loss')
    # plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.show()