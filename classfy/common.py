# -*- coding=utf8 -*-
"""
    公共模块：生成训练数据，画曲线...
"""

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(0)
data, model = sklearn.datasets.make_moons(200, noise=0.30)


def gen_train_data():
    """生成训练数据"""
    global data, model
    return data, model


def plot_decision_boundary(predict_func, data, label):
    """画出结果图

    Args:
        pred_func (callable): 预测函数
        data (numpy.ndarray): 训练数据集合
        label (numpy.ndarray): 训练数据标签
    """
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral)
    plt.show()


# if __name__ == '__main__':
#     import sklearn.linear_model
#
#     X, y = gen_train_data()
#     clf = sklearn.linear_model.LogisticRegressionCV()
#     clf.fit(X, y)
#     plot_decision_boundary(lambda x: clf.predict(x), X, y)
