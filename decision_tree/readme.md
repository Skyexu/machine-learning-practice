# 决策树

1．分类决策树模型是表示基于特征对实例进行分类的树形结构。决策树可以转换成一个**if-then**规则的集合，也可以看作是定义在特征空间划分上的类的条件概率分布。

2．决策树学习旨在构建一个与训练数据拟合很好，并且复杂度小的决策树。因为从可能的决策树中直接选取最优决策树是NP完全问题。现实中采用启发式方法学习次优的决策树。

决策树学习算法包括3部分：特征选择、树的生成和树的剪枝。常用的算法有ID3、C4.5和CART。

3．特征选择的目的在于选取对训练数据能够分类的特征。特征选择的关键是其准则。常用的准则如下：

（1）样本集合$D$对特征$A$的信息增益（ID3）

$$g(D, A)=H(D)-H(D|A)$$

$$H(D)=-\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|} \log _{2} \frac{\left|C_{k}\right|}{|D|}$$

$$H(D | A)=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} H\left(D_{i}\right)$$

其中，$H(D)$是数据集$D$的熵，$H(D_i)$是数据集$D_i$的熵，$H(D|A)$是数据集$D$对特征$A$的条件熵。	$D_i$是$D$中特征$A$取第$i$个值的样本子集，$C_k$是$D$中属于第$k$类的样本子集。$n$是特征$A$取 值的个数，$K$是类的个数。

（2）样本集合$D$对特征$A$的信息增益比（C4.5）


$$g_{R}(D, A)=\frac{g(D, A)}{H(D)}$$


其中，$g(D,A)$是信息增益，$H(D)$是数据集$D$的熵。

（3）样本集合$D$的基尼指数（CART）

$$\operatorname{Gini}(D)=1-\sum_{k=1}^{K}\left(\frac{\left|C_{k}\right|}{|D|}\right)^{2}$$

特征$A$条件下集合$D$的基尼指数：

 $$\operatorname{Gini}(D, A)=\frac{\left|D_{1}\right|}{|D|} \operatorname{Gini}\left(D_{1}\right)+\frac{\left|D_{2}\right|}{|D|} \operatorname{Gini}\left(D_{2}\right)$$

4．决策树的生成。通常使用信息增益最大、信息增益比最大或基尼指数最小作为特征选择的准则。决策树的生成往往通过计算信息增益或其他指标，从根结点开始，递归地产生决策树。这相当于用信息增益或其他准则不断地选取局部最优的特征，或将训练集分割为能够基本正确分类的子集。

5．决策树的剪枝。由于生成的决策树存在过拟合问题，需要对它进行剪枝，以简化学到的决策树。决策树的剪枝，往往从已生成的树上剪掉一些叶结点或叶结点以上的子树，并将其父结点或根结点作为新的叶结点，从而简化生成的决策树。



## 算法实现

- `DicesionTree.ipynb`：notebook 形式，比较直观
- `DecisionTree.py`：决策树实现 py 代码

参考：https://github.com/fengdu78/lihang-code



## Scikit-learn 实践

`practice` 目录下

1. 决策树分类问题简单练习_sklearn 中的决策树对类别特征的处理_

2. 连续值特征的分类_iris
3. 回归树_Boston 房价预测
4. 随机森林_Boston房价预测



## 注意事项

决策树需要剪枝避免陷入过拟合。

sklearn 中实现的分类决策树是 CART 树，是二叉树，对于离散值还是按连续值处理的，这样的作为会默认离散值的数值有意义，决策树特征选择时会按照数值排序切分。所以对于类别特征一般可以做处理
- one-hot: 失去数值意义，标记类别
- label encoding：以有意义的数值来代替类别特征，比如以人均消费代替城市类别

参考：
https://scikit-learn.org/stable/modules/tree.html#tree

其中有描述 scikit-learn uses an optimised version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now.



如果按连续值来处理离散值，即是按两个连续值的中点来进行树的切分，模型也可以学出来，但还是会考虑数值含义。比如类别是[1,2,3,4,5]，切分点为 [1.5,2.5,3.5,4.5]， 按1.5切分则会分割为两个区间 <1.5 和 > 1.5。

lightGBM 可以处理离散值。

“If you read how the trees work, they are very different. XGB doesn't use categories and builds depth wise. LGBM uses categories and builds leaf wise. And CatBoost uses categories and make interaction terms for you. So you see that they all build different interactions. (The shapes of trees control interactions).”

