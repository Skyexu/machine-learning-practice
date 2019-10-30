## 机器学习算法实现

>  统计学习方法 李航

**Features**

- 代码使用 scikit-learn 的形式组织，方便学习和实验
- 代码按照统计学习方法中的思路实现，包含详细实现思路与注释

**Example**

```python
# 加载数据
train_path = '../mnist/mnist_train.csv'
test_path = '../mnist/mnist_test.csv'
train_data, train_label = load_data(train_path)
test_data, test_label = load_data(test_path)

# 创建朴素贝叶斯分类器
nb = NaiveBayes(var_smoothing=1)
# 训练
nb.fit(train_data, train_label)
# 测试
nb.score(test_data, test_label)
```



数据集: `mnist`， 请使用 `./data/transMinist.py` 生成

- [x] K 近邻： `KNN`
- [x] 感知机： `Perceptron`
- [x] 朴素贝叶斯：`NaiveBayes`

- [x] 决策树：`DicisionTree`
- [x] 逻辑斯谛回归：`LogisticRegression`

参考：

- https://github.com/Dod-o/Statistical-Learning-Method_Code
- 



## 比赛实践与学习

- 泰坦尼克生存预测



