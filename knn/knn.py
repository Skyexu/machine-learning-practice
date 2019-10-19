#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2019/7/29 21:09
# Description: 统计学习方法第三章，k近邻（k-nearest neighbor, k-nn）算法实现

"""
数据集：mnist
Train：60000
test：10000

特征为 0~255 之间的数，标签为 0~9

---------------
为了计算速度，只测试测试集中前100组数据, k = 20

欧式距离
accuracy:  0.97
running time : 92.55344700813293

曼哈顿距离
accuracy:  0.95
running time : 83.68424868583679
"""
import numpy as np
import time


class KNN:

	def __init__(self, k=5, dist_method='eu'):
		"""
		:param k: 近邻个数
		:param dist_method: 距离计算方法，'eu' -- 欧式距离  'ma' 曼哈顿距离
		"""
		self.k = k
		self.dist_method = dist_method

	def cal_dist(self, x1, x2, dist_method):
		"""
		计算向量距离
		:param x1:
		:param x2:
		:param dist_method:
		:return:
		"""
		if dist_method == 'eu':
			return np.sqrt(np.sum(np.square(x1-x2)))
		elif dist_method == 'ma':
			return np.sum(np.abs(x1-x2))
		else:
			raise Exception("distance method error!")

	def get_knn_pre(self, train_mat, x, train_label):
		"""
		计算最近邻，并返回预测类别
		:param train_mat: 训练数据向量
		:param x: 输入实例向量
		:param train_label: 训练数据标签
		:return:
		"""
		top_k = self.k
		m, n = train_mat.shape

		# 列表保存距离
		dists = [0] * m
		for i in range(m):
			dists[i] = self.cal_dist(train_mat[i], x, self.dist_method)

		# 对列表进行排序，并输出原始位置

		# np.argsort 返回从小到大排序后的数组坐标
		# Returns the indices that would sort an array.
		# One dimensional array:
		# >>> x = np.array([3, 1, 2])
		# >>> np.argsort(x)
		# array([1, 2, 0])

		top_k_list = np.argsort(np.array(dists))[:top_k]

		# 一个列表存储， 10 个类别的计数
		label_count = [0] * 10
		for n in top_k_list:
			label_count[train_label[n]] += 1

		# 列表中最大值的索引即为预测的类别
		return label_count.index(max(label_count))

	def score(self, train_data, test_data, train_label, test_label,):
		"""
		:param train_data: 训练数据
		:param test_data: 测试数据
		:param train_label: 训练数据标签
		:param test_label: 测试数据标签
		:return:
		"""
		train_mat = np.mat(train_data)
		test_mat = np.mat(test_data)

		error_cnt = 0
		# 预测 100 组数据
		for k in range(100):
			y = self.get_knn_pre(train_mat, test_mat[k], train_label)
			if y != test_label[k]:
				error_cnt += 1
		# 正确率 = 1 - 错误分类样本数 / 总样本数
		print('accuracy: ', 1 - error_cnt / 100)


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
			data.append([int(x) for x in line[1:]])
	return data, label


if __name__ == '__main__':
	train_path = '../mnist/mnist_train.csv'
	test_path = '../mnist/mnist_test.csv'
	train_data, train_label = load_data(train_path)
	test_data, test_label = load_data(test_path)
	start = time.time()
	knn = KNN(20, 'eu')
	# 测试
	knn.score(train_data, test_data, train_label, test_label)

	end = time.time()
	# 程序运行时长
	print('running time :', end - start)
