#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2019/8/30 16:08
# Description: 统计学习方法第三章，朴素贝叶斯算法实现

class NaiveBayes:
    def __init__(self, laplace=1):
        """
        初始化方法
        :param laplace: 拉普拉斯平滑参数， lambda
        """
        self.laplace = laplace

    def fit(self, train_data, train_label):


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