#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2019/10/19 16:12
# Description: 决策树 ID3 算法实现，没有剪枝
"""
数据集：mnist
Train：60000
test：10000

特征二值化  > 128 为 1  <= 128 为 0

accuracy:  0.8591
"""
import numpy as np
import pandas as pd
from math import log
import time

# 定义树节点类, 分类树
class Node:
    def __init__(self, root=True, label=None, feature_name=None):
        # root 标记是否为单结点树（即叶子结点）
        self.root = root
        # label 为当前节点的类别标记
        self.label = label
        # feature_name 为当前节点的特征名
        self.feature_name = feature_name

        # 子树，字典形式，（特征取值：子树）
        self.tree = {}
        self.result = {
            'lable': self.label,
            'feature': self.feature_name,
            'tree': self.tree
        }

    def __repr__(self):
        """
        重写 __repr__ 用于显示对象，也可以 使用 __str__
        dt = Node()
        print(dt) 就可以显示自己定义的信息了
        这里  self.result 中包含 self.tree，所以会递归显示所有子树信息
        :return:
        """
        return '{}'.format(self.result)

    def add_node(self, val, node):
        """
        :param val: 特征的取值
        :param node:
        :return:
        """
        """
        增加树节点
        """
        self.tree[val] = node

    def predict(self, x_sample):
        """
        :param x_sample: 一行数据 Series 格式，能通过特征名索引特征值
        :return: 当前样本的预测值
        """
        if self.root is True:
            return self.label
        # x_sample[self.feature_name] 当前特征下样本实例的取值
        return self.tree[x_sample[self.feature_name]].predict(x_sample)


class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2)
                    for p in label_count.values()])
        return ent

    # 经验条件熵 H(D|A)
    def cond_ent(self, datasets, index=0):
        """
        :param index: 当前特征列 A 在 datasets 中的索引
        """
        data_length = len(datasets)
        # 其中每个元素存储当前特征值时的所有样本
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][index]
            if feature not in feature_sets:
                feature_sets[feature] = []
            # 当前样本 datasets[i] 加入列表中
            feature_sets[feature].append(datasets[i])
        # H(D|A)
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        # 对所有特征计算信息增益
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, index=c))
            best_feature.append((c, c_info_gain))

        # 选择信息增益最大的特征作为根节点
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        :param train_data: 训练数据， pandas 格式，最后一列为 label
        :return: 决策树
        """
        # y_train: 最后一列 label ， features : 特征名
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[: -1]

        # 1. 若 D 中实例属于同一类 Ck， 则 T 为单节点树，并将类 Ck 作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])
        # 2. 若特征集 A 为空，则T为单节点树，将 D 中实例数最大的类 Ck 作为该结点的类标记，返回 T
        if len(features) == 0:
            # 对 Series 排序，去索引第一的作为类标
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 3. 计算 A 中各特征对 D 的最大信息增益，选择 Ag
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4. Ag 的信息增益小于阈值 eta，则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5. 构建 Ag 子集，对每一个 Ag = ai 将D分割为若干个子集（相当于多个子树），子集中实例数最大的类别作为当前节点
        node_tree = Node(root=False, feature_name=max_feature_name)
        # 获取特征 Ag 的所有取值
        feature_list = train_data[max_feature_name].unique()
        for f in feature_list:
            # 获取子集，删掉当前特征
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 6. 递归调用，生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict_one_sample(self, x_vec):
        """
        预测一条样本
        :param x_vec: 样本向量
        :return:
        """
        return self._tree.predict(x_vec)

    def predict(self, X_test):
        """
        预测结果
        :param X_test: DataFrame 格式
        :return:
        """
        result = []
        for i in range(X_test.shape[0]):
            result.append(self.predict_one_sample(X_test.iloc[i]))
        return result

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

def create_data():
    """
    统计学习书本上的例子
    :return:
    """
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    feature_names = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    # 返回数据集和每个维度的名称
    return datasets, feature_names

def load_data(train_path):
    """
    加载数据集
    :param file_path: 文件路径
    :return:
    """
    print('reading data.....')
    # 数据与标签
    data = []
    with open(train_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            # 存储特征
            # 将特征进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
            sample = [int(int(x) > 128) for x in line[1:]]
            sample.append(int(line[0]))
            data.append(sample)
    return data


if __name__ == '__main__':
    # 测试书上 5.3 的例子
    print("5.3 example")
    datasets, labels = create_data()
    data_df = pd.DataFrame(datasets, columns=labels)
    dt = DTree()
    tree = dt.fit(data_df)
    print(tree)
    X_train = data_df.iloc[:, :-1]
    y_train = data_df.iloc[:, -1]
    print(X_train.iloc[4])
    print(dt.predict_one_sample(X_train.iloc[5]))
    dt.score(X_train, y_train)

    print("Minist")
    # test minist
    start = time.time()
    train_path = '../mnist/mnist_train.csv'
    test_path = '../mnist/mnist_test.csv'
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    dt2 = DTree()
    tree2 = dt2.fit(train_df)
    print(tree2)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    dt2.score(X_test, y_test)
    end = time.time()
    # 程序运行时长
    print('running time : {:.2f}s'.format(end - start))
