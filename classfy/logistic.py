# -*- coding=utf8 -*-
"""
    用logistic回归分类
"""
from __future__ import division
from numpy import *


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


class Logistic(object):
    """logistic回归模型"""
    def __init__(self, data, label):
        self.data = data
        self.label = label

        self.data_num, n = shape(data)
        self.weights = ones(n)
        self.b = 1

    def train(self, num_iteration=150):
        """随机梯度上升算法

        Args:
            data (numpy.ndarray): 训练数据集
            labels (numpy.ndarray): 训练标签
            num_iteration (int): 迭代次数
        """
        for j in xrange(num_iteration):
            data_index = range(self.data_num)
            for i in xrange(self.data_num):
                # 学习速率
                alpha = 0.01
                rand_index = int(random.uniform(0, len(data_index)))
                error = self.label[rand_index] - sigmoid(sum(self.data[rand_index] * self.weights + self.b))
                self.weights += alpha * error * self.data[rand_index]
                self.b += alpha * error
                del(data_index[rand_index])

    def predict(self, predict_data):
        """预测函数"""
        result = map(lambda x: 1 if sum(self.weights * x + self.b) > 0 else 0,
                     predict_data)
        return array(result)


if __name__ == '__main__':
    from common import gen_train_data, plot_decision_boundary
    data, label = gen_train_data()

    logistic = Logistic(data, label)
    logistic.train(200)
    plot_decision_boundary(lambda x: logistic.predict(x), data, label)
