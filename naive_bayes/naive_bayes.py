#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2019/8/30 16:08
# Description: 统计学习方法第三章，朴素贝叶斯算法实现
# 1. 计算先验概率和条件概率
#   先验概率：需要一个数组存放所有类别
#   条件概率：每个特征的可能取值不同。所以可以用字典来表示不同的特征取值下的条件概率，
#             我这里使用所有特征取值的(最大值 + 1)新建数组 [K(label number),J(feature number),max_feature_value + 1]
#          **注：需要保证输入的训练数据是数值型类别数据**
# 2. 对于给定的实例，计算估计值
# 3. 对每个类别的估计值排序，去最高者作为当前实例的类别
# 代码中 label = class


"""

数据集：mnist
Train：60000
test：10000

特征二值化  > 128 为 1  <= 128 为 0

var_smoothing = 1
accuracy:  0.8436
running time : 47.50s
"""
import numpy as np
import time


class NaiveBayes:
    def __init__(self, var_smoothing=1.0):
        """
        初始化方法
        :param var_smoothing: 拉普拉斯平滑参数， lambda
        """
        self.var_smoothing = var_smoothing
        # 类别先验概率
        self.class_prior = None
        # 类别数组
        self.classes_ = None
        # 条件概率存储数组
        self.conditional_prob = None

    def fit(self, train_data, train_label):
        """
        训练朴素贝叶斯
        :param train_data: 训练数据
        :param train_label: 训练数据标签
        :return:
        """
        X = np.array(train_data)
        y = np.array(train_label)

        #  np.unique(y) return the sorted unique values
        self.classes_ = np.unique(y)
        classes = self.classes_
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.class_prior = np.zeros(n_classes, dtype=np.float64)
        # 使用 X.max() + 1 ， 由于是数值型类别特征，以防特征值从 0 开始
        self.conditional_prob = np.zeros((n_classes, n_features, X.max() + 1))

        # 每个 label
        for y_i in classes:
            # 当前 label 在 classes 数组中的索引
            i = classes.searchsorted(y_i)
            # 获取当前 label 的样本
            X_i = X[y == y_i, :]
            # 当前 label 的样本数
            label_count = X_i.shape[0]
            # 当前 label 的先验概率
            self.class_prior[i] = label_count * 1.0 / X.shape[0]

            # 计算条件概率
            for j in range(n_features):
                # 特征 j ， 类别  y_i 对应的所有样本值
                for x_j in X_i[:, j]:
                    # 累加计数 特征 j ， 类别  y_i 的个数
                    self.conditional_prob[i, j, x_j] += 1

            for j in range(n_features):
                # 特征 j ， 类别  y_i 对应的可能取值的条件概率
                tmp_prob = np.unique(X_i[:, j])
                for v in tmp_prob:
                    # 贝叶斯估计
                    self.conditional_prob[i, j, v] = (self.conditional_prob[i, j, v] + self.var_smoothing) / \
                                                     (label_count + tmp_prob.shape[0] * self.var_smoothing)
        return self

    def predict(self, test_data):
        """
        根据测试集返回预测结果
        :param test_data:
        :return: array
        """

        all_label_prob = self.predict_prob(test_data)
        # 存储预测类别
        predict_label = np.zeros(all_label_prob.shape[0])

        for i in range(all_label_prob.shape[0]):
            now_prob = all_label_prob[i]
            # 排序后取最大值对应的类别
            predict_label[i] = self.classes_[np.argsort(now_prob)[-1]]

        return predict_label

    def predict_prob(self, test_data):
        """
        根据测试集返回预测概率
        :param test_data:
        :return: array
        """
        Y = np.array(test_data)
        # 存储每个样本对于每个类别的预测概率值
        all_label_prob = np.zeros((Y.shape[0], len(self.classes_)))

        # 遍历每个样本
        for n in range(Y.shape[0]):
            sample = Y[n]
            # 存储每个类别的预测值
            label_prob = np.zeros(len(self.classes_))
            for y_i in self.classes_:
                # 当前 label 在 classes 数组中的索引
                i = self.classes_.searchsorted(y_i)
                prob = 1
                # 样本中的每个特征值对应的条件概率乘积
                for j in range(len(sample)):
                    prob *= self.conditional_prob[i, j, sample[j]]
                # 乘先验概率
                prob *= self.class_prior[i]
                label_prob[i] = prob
            all_label_prob[n] = label_prob

        return all_label_prob

    def score(self, test_data, test_label):
        """
        :param test_data: 测试数据
        :param test_label: 测试数据标签
        :return:
        """
        predict_label = self.predict(test_data)

        error_cnt = 0
        for k in range(len(predict_label)):
            # 记录误分类数
            if predict_label[k] != test_label[k]:
                error_cnt += 1
        # 正确率 = 1 - 错误分类样本数 / 总样本数
        print('accuracy: ', 1 - error_cnt / len(predict_label))


def load_data(train_path):
    """
    加载数据集
    :param file_path: 文件路径
    :return:
    """
    print('reading data.....')

    # 数据与标签
    data = []
    label = []
    with open(train_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            # 存储类别
            label.append(int(line[0]))
            # 存储特征
            # 将特征进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
            data.append([int(int(x) > 128) for x in line[1:]])
    return data, label


if __name__ == '__main__':
    train_path = '../mnist/mnist_train.csv'
    test_path = '../mnist/mnist_test.csv'
    train_data, train_label = load_data(train_path)
    test_data, test_label = load_data(test_path)

    start = time.time()
    nb = NaiveBayes(var_smoothing=1)
    # 训练
    nb.fit(train_data, train_label)

    # 测试
    nb.score(test_data, test_label)

    end = time.time()
    # 程序运行时长
    print('running time : {:.2f}s'.format(end - start))
