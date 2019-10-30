#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2019/7/25 17:55
# Description: 统计学习方法第二章，感知机算法实现

"""
数据集：mnist
Train：60000
test：10000

特征为 0~255 之间的数，标签为 0~9

alpha=0.0001, iter_n=30

accuracy:  0.8172
running time : 20.67824387550354

"""
import numpy as np
import time


class Perceptron:

	def __init__(self, alpha=0.0001, iter_n=20):
		"""
		:param alpha: 步长
		:param iter: 迭代次数
		"""
		self.alpha = alpha
		self.iter_n = iter_n
		self.w = None
		self.b = 0

	def fit(self, train_data, train_label):
		"""
		:param train_data: 训练数据
		:param train_label: 训练数据标签
		:return:
		"""
		train_mat = np.mat(train_data)
		# m 为样本数，n 为特征维数
		m, n = train_mat.shape
		# 初始化权重向量为 0，偏置 b 为 0
		w = np.zeros((1, n))
		b = 0
		# 迭代次数
		for k in range(self.iter_n):
			for i in range(m):
				y_i = train_label[i]
				x_i = train_mat[i]
				# 判断是否为误分类样本 yi(w*xi+b) <= 0
				if y_i * (np.inner(x_i, w) + b) <= 0:
					w = w + self.alpha * y_i * x_i
					b = b + self.alpha * y_i
			# 打印训练进度
			print('iter %d' % k)
		self.w = w
		self.b = b

	def score(self, test_data, test_label):
		"""
		:param test_data: 测试数据
		:param test_label: 测试数据标签
		:return:
		"""
		test_mat = np.mat(test_data)
		m, n = test_mat.shape
		error_cnt = 0
		for k in range(m):
			# 计算 yi(w*xi+b), 记录误分类数
			if test_label[k] * (np.inner(self.w, test_mat[k]) + self.b) <= 0:
				error_cnt += 1
		# 正确率 = 1 - 错误分类样本数 / 总样本数
		print('accuracy: ', 1 - error_cnt / m)


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
			# 0 ~ 9 标签中大于5的设为正样本
			label.append(1 if int(line[0]) >= 5 else -1)
			# 除以 255 归一化
			data.append([int(x) / 255 for x in line[1:]])
	return data, label


def load_data2(train_path):
	"""
	直接使用 np 加载数据集
	:param file_path: 文件路径
	:return:
	"""
	print('reading data.....')

	# 数据与标签
	data = np.loadtxt(train_path, delimiter=',')
	label = data[..., 0]
	# 0 ~ 9 标签中大于5的设为正样本
	label = [1 if x >= 5 else -1 for x in label]
	# 除以 255 归一化
	return data[..., 1:] / 255, label


if __name__ == '__main__':
	train_path = '../mnist/mnist_train.csv'
	test_path = '../mnist/mnist_test.csv'
	train_data, train_label = load_data(train_path)
	test_data, test_label = load_data(test_path)
	start = time.time()
	per = Perceptron(alpha=0.0001, iter_n=30)
	# 训练
	per.fit(train_data, train_label)
	# 测试
	per.score(test_data, test_label)
	end = time.time()
	# 程序运行时长
	print('running time :', end - start)

