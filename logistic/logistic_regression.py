#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2019/9/9 16:27
# Description:
import math
import sys

def predict(w, x):
    '''
    预测标签
    :param w:训练过程中学到的w
    :param x: 要预测的样本
    :return: 预测结果
    '''
    # dot为两个向量的点积操作，计算得到w * x
    wx = dot(w, x)
    #计算标签为1的概率
    P1 = 1/(1 + math.exp(-wx))
    #如果为1的概率大于0.5，返回1
    if P1 >= 0.5:
        return 1
    #否则返回0
    return 0

def dot(vec1,vec2):
    result = 0
    for i in range(len(vec1)):
        result += vec1[i] + vec2[i]

    return result

def vec_add(vec1, vec2):
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result

def vec_add_num(vec1, num):
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] + num)
    return result

def vec_mult(vec,w):
    result = []
    for i in range(len(vec)):
        result.append(vec[i] * w)
    return result

def logisticRegression(trainDataList, trainLabelList, iter = 100, alpha = 0.1, re=10):
    """
    逻辑斯蒂回归训练过程
    :param trainDataList:训练集
    :param trainLabelList: 标签集
    :param iter: 迭代次数
    :param alpha: 学习率
    :param re: 正则参数
    :return: 习得的w
    """

    feature_num = len(trainDataList[0])
    # 初始化w
    w = [1] * feature_num

    #迭代iter次进行梯度下降
    for i in range(iter):
        left = [0] * feature_num
        regular = 0
        #每次迭代冲遍历一次所有样本，进行随机梯度下降
        for j in range(len(trainDataList)):
            yi = trainLabelList[j]
            xi = trainDataList[j]
            wx = dot(xi,w)
            tmp = alpha * (yi - 1/(1 +  math.exp(-wx)))
            left = vec_add(left, vec_mult(xi,tmp))
            regular += w[j]

        left = vec_mult(left, 1/feature_num)
        regular = regular * re / feature_num

        w = vec_add(w, vec_add_num(left, -regular))

    #返回学到的w
    return w

count = 0
alpha = 0.1
re = 10
iter = 100
train_len = 10
test_len = 10
feature = 5

train_data = []
train_label = []
test_data = []

for line in sys.stdin:
    a = line.split(" ")
    if count == 0:
        alpha = float(a[0])
        re = int(a[1])
        iter = int(a[2])
        feature = int(a[3])
        train_len = int(a[4])
        test_len = int(a[4])
        count += 1
    elif count >=1 and count <= train_len:
        tmp = [float(x) for x in a]
        train_data.append(tmp[:len(tmp) - 1])
        train_label.append(int(a[-1]))
    else:
        train_data.append([float(x) for x in a])

w = logisticRegression(train_data, train_label, iter, alpha,re)

for i in range(len(test_data)):
    print(predict(w, test_data[i]))
