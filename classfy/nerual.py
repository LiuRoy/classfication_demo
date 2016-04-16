# -*- coding=utf8 -*-
"""
    神经网络分类
"""
from __future__ import division
import numpy as np


class NN(object):
    """神经网络"""
    def __init__(self, data, label, nn_hdim):
        self.data = data
        self.label = label
        self.nn_hdim = nn_hdim

        self.num_examples, self.nn_input_dim = np.shape(data)
        self.nn_output_dim = 2

        # 梯度下降参数 epsilon是学习速率 reg_lambda是正则化强度
        self.epsilon = 0.01
        self.reg_lambda = 0.01

        # 神经网络权重不能初始化0, 需要随机生成
        np.random.seed(0)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hdim))
        self.W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def batch_gradient_descent(self, num_passes=20000):
        """批量梯度下降训练模型"""
        for i in xrange(0, num_passes):
            # Forward propagation
            z1 = self.data.dot(self.W1) + self.b1
            a1 = np.tanh(z1)
            z2 = a1.dot(self.W2) + self.b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs

            delta3[range(self.num_examples), self.label] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(self.data.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -self.epsilon * dW1
            self.b1 += -self.epsilon * db1
            self.W2 += -self.epsilon * dW2
            self.b2 += -self.epsilon * db2

    def stochastic_gradient_descent(self, num_passes=200):
        """随机梯度下降训练模型"""
        for i in xrange(0, num_passes):
            data_index = range(self.num_examples)

            for j in xrange(self.num_examples):
                rand_index = int(np.random.uniform(0, len(data_index)))
                x = np.mat(self.data[rand_index])
                y = self.label[rand_index]

                # Forward propagation
                z1 = x.dot(self.W1) + self.b1
                a1 = np.tanh(z1)
                z2 = a1.dot(self.W2) + self.b2
                exp_scores = np.exp(z2)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Backpropagation
                delta3 = probs
                if y:
                    delta3[0, 0] -= 1
                else:
                    delta3[0, 1] -= 1
                dW2 = (a1.T).dot(delta3)
                db2 = np.sum(delta3, axis=0, keepdims=True)
                va = delta3.dot(self.W2.T)
                vb = 1 - np.power(a1, 2)
                delta2 = np.mat(np.array(va) * np.array(vb))
                dW1 = x.T.dot(delta2)
                db1 = np.sum(delta2, axis=0)

                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW2 += self.reg_lambda * self.W2
                dW1 += self.reg_lambda * self.W1

                # Gradient descent parameter update
                self.W1 += -self.epsilon * dW1
                self.b1 += -self.epsilon * db1
                self.W2 += -self.epsilon * dW2
                self.b2 += -self.epsilon * db2

                del(data_index[rand_index])

    def predict(self, x):
        """预测函数"""
        # Forward propagation
        z1 = x.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

if __name__ == '__main__':
    from common import gen_train_data, plot_decision_boundary
    data, label = gen_train_data()

    nn = NN(data, label, 3)
    #nn.stochastic_gradient_descent()
    nn.batch_gradient_descent()
    plot_decision_boundary(lambda x: nn.predict(x), data, label)
