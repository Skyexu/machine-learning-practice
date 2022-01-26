#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2021/10/26 17:02
# Description: DCN 算法
# Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[M]//Proceedings of the ADKDD'17. 2017: 1-7.



import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Input, Layer, Dropout

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
        # 增加维度方便后续计算乘法   dim,1 相当于论文中的列向量
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

class CrossNetworkTmp(Layer):
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
                            shape=(dim, ),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, ),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        # 增加维度方便后续计算乘法   dim,1 相当于论文中的列向量
        #x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        # 作为常量复制，不会更新
        # xl+1 = x0 * xl * w + b + xl
        embed_dim = inputs.shape[-1]
        x_0 = inputs
        x_l = x_0  # (batch_size, dim, 1)
        for i in range(self.layer_num):
            # xl * w  先做 xl * w   再做 x0 * xl_w 减少内存开销    原来的 x0 * xl * w 的方式 会出现   dim*dim 的维度参数，而这个方式不会出现
            # tensordot 取 x_l 的1维（横） 与 cross_weights 的 0 维（列） 相乘 (batch_size, dim, 1) * ( dim,1) 得到标量
            x1_T = tf.reshape(x_l, [-1, 1, embed_dim])
            xl_w = tf.tensordot(x1_T, self.cross_weights[i], axes=1)  # (batch_size, 1, 1 )
            x_l = x_0 * xl_w + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        # 删除维度2
       # x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l

class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """Deep Neural Network
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dropout: A scalar. Dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)

        # 最后一层结束才 dropout
            x = self.dropout(x)
        return x


class DCN(Model):
    def __init__(self, sparse_feature_columns, dense_feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-6, cross_w_reg=1e-6, cross_b_reg=1e-6):
        """
        Deep&Cross Network
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cross_w_reg: A scalar. The regularizer of cross network.
        :param cross_b_reg: A scalar. The regularizer of cross network.
        """
        super(DCN, self).__init__()

        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.layer_num = len(hidden_units)
        # N个特征建立N个embedding向量
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def call(self, inputs, **kwargs):

        # 把所有 embedding concat 把同一个样本的输入向量的 embedding 向量 concat 到同一个向量中
        # sparse_embed = tf.concat([self.embed_layers[feat['feat_name']](sparse_data[feat['feat_name']])
        #                           for feat in self.sparse_feature_columns], axis=-1)

        # 输入会由 ndarray 转化为 tensor input ，tensor 需要以 index 来切分输入
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs[:, i])
                                  for i in range(len(self.sparse_feature_columns))], axis=-1)

        dense_inputs = inputs[:, len(self.sparse_feature_columns):]

        x = tf.keras.layers.Concatenate(axis=-1)([sparse_embed, dense_inputs])
        # Cross Network
        cross_x = self.cross_network(x)
        # DNN
        dnn_x = self.dnn_network(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        inputs = Input(shape=(len(self.sparse_feature_columns+self.dense_feature_columns),), dtype=tf.float32)
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


