# DCN 算法

- `dcn.py`: 模型代码
- `dcn_train.py` ：训练样例

模型代码没有多余的算法库依赖，方便理解模型实现逻辑，与本地实验。数据样例使用 `criteo` 采样的 1万条数据。

## 介绍

在 Wide&Deep 模型提出之后，越来越多的工作基于其中的 Wide 或者 Deep 部分进行改进，而 DCN（Deep & Cross Network） 就是改进了 Wide 部分。这里首先简单介绍一下 Wide&Deep 的思路。

- **Wide 部分**：记忆能力。 简单模型具有强记忆能力，可以直接利用历史数据中的物品或者特征的“共现”能力。如逻辑回归、协同过滤。
- **Deep 部分**： 泛化能力。模型传递特征的相关性，发掘稀疏甚至从未出现过的稀有特征与最终标签相关的能力。比如矩阵分解比协同过滤泛化能力强，通过引入隐向量结构，使数据稀少的item也能生成隐向量，从而将全局数据传递到稀疏物品，提高泛化能力。深度神经忘了，通过特征多次自动组合，发掘前在模式，即使非常稀疏的特征向量也能得到稳定平滑的推荐概率。

![image-20211026114853660](https://gitee.com/skyexu/images/raw/master/img/Wide_deep_model.png)

DCN 算法主要由 Cross 网络代替原来的 Wide 部分，目的在于**增加特征之间的交互能力**。



## 模型

DCN 算法的模型结构如下图。

![image-20211026143703675](https://gitee.com/skyexu/images/raw/master/img/dcn_model.png)



### Emebeding and stacking layer

通过 embedding 将类别特征（Sparse feature）转化为向量，避免高维 one_hot 的特征稀疏。然后将归一化之后的稠密特征（Dense feature） 合并（Stacking）到一起作为原始的输入。
$$
\mathbf{x}_{0}=\left[\mathbf{x}_{\text {embed, } 1}^{T}, \ldots, \mathbf{x}_{\text {embed, } k}^{T}, \mathbf{x}_{\text {dense }}^{T}\right]
$$


### Cross Network

Cross Network 是DCN 算法的核心，关键在于如何设计高效的特征组合，每一层公式如下。
$$
\mathbf{x}_{l+1}=\mathbf{x}_{0} \mathbf{x}_{l}^{T} \mathbf{w}_{l}+\mathbf{b}_{l}+\mathbf{x}_{l}=f\left(\mathbf{x}_{l}, \mathbf{w}_{l}, \mathbf{b}_{l}\right)+\mathbf{x}_{l}
$$
xl 和xl + 1分别是第 l 层和第 l + 1层cross layer的输出，wl 和 bl 分别是连接参数。式中所有的变量均是列向量，W也是列向量，每一层的输出，都是上一层的输出加上feature crossing f。而f就是在拟合该层输出和上一层输出的残差。一层 cross layer 的图如下

![image-20211026151324950](https://gitee.com/skyexu/images/raw/master/img/dcn_cross_layer.png)

cross netwrok 的特征交叉阶数随着层数增加而增加，第l层的特征交叉为l+1阶交叉。因此 cross network 以一种参数共享的方式，通过对叠加层数的控制，可以高效地学习出低维的特征交叉组合，避免了人工特征工程。cross netwrok 参数量为
$$
d \times L_{c} \times 2
$$
其中 d 为特征维度，Lc 为网络层数。参数是线行增加的。

论文中写到，**因为 cross network 参数量较少限制了模型的复杂性，所以引入并行的 DNN 来提取高维非线性信息。**



### Combination Layer

该层将 Cross Network 和 DNN 的输出拼接，加权求和后，通过 sigmoid 函数得到最终的预测概率，损失函数为 logloss。


$$
p=\sigma\left(\left[\mathbf{x}_{L_{1}}^{T}, \mathbf{h}_{L_{2}}^{T}\right] \mathbf{w}_{\text {logits }}\right)
$$


## 代码实现

DCN 算法核心需要实现 Cross Network ，其中关键点是实现 $\mathbf{x}_{0} \mathbf{x}_{l}^{T} \mathbf{w}_{l}$ 时，为了优化内存的使用，可以先计算 $\mathbf{x}_{l}^{T} \mathbf{w}_{l}$ 得到标量，从而避免 $\mathbf{x}_{0} \mathbf{x}_{l}^{T}$ 得到 dim* dim 的矩阵，减少内存开销。

```python
class CrossNetwork(Layer): 
    def __init__(self, layer_num, reg_w=1e-6, reg_b=1e-6):
        """CrossNetwork
        :param layer_num: A scalar. The depth of cross network
        :param reg_w: A scalar. The regularizer of w
        :param reg_b: A scalar. The regularizer of b
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    # 创建层权重
    def build(self, input_shape):
        dim = int(input_shape[-1])
        # 权重维列向量
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        # 增加维度方便后续 tensorflow 中乘法的实现   dim,1 相当于论文中的列向量
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        # 作为常量复制，不会更新
        # xl+1 = x0 * xl * w + b + xl
        x_l = x_0  # (batch_size, dim, 1)
        for i in range(self.layer_num):
            # xl * w  先做 xl * w   再做 x0 * xl_w 减少内存开销    原来的 x0 * xl * w 的方式 会出现   dim*dim 的维度参数，而这个方式不会出现
            # tensordot 取 x_l 的1维（横） 与 cross_weights 的 0 维（列） 相乘 (batch_size, dim, 1) * ( dim,1) 得到标量 
            xl_w = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, 1, 1 )
            x_l = tf.matmul(x_0, xl_w) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        # 删除维度2
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l
```





## 参考

- Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.

- Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[M]//Proceedings of the ADKDD'17. 2017: 1-7.
- 深度学习推荐系统. 王喆
- https://mp.weixin.qq.com/s/DkoaMaXhlgQv1NhZHF-7og
- https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0