#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# Datetime:2021/10/26 17:05
# Description: 

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from dcn import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.model_selection import train_test_split

def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat_name': feat}



if __name__ == '__main__':
    # ========================= Hyper Parameters =======================

    file = '../../data/criteo/criteo_example.txt'

    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 128
    epochs = 10
    # ========================== Create dataset =======================
    # 读取 criteo 数据
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
    data_df = pd.read_csv(file, sep='\t', header=None, names=names)
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    # 缺失值填充
    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)
    # Bin continuous data into intervals.
    # 离散等宽分桶，按桶编号编码
    # est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    # data_df[dense_features] = est.fit_transform(data_df[dense_features])

    # 稀疏类别特征重新编码 0 - n_classes-1
    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 数值型特征 minmax 归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])
    # ==============Feature Engineering===================
    # feature_columns 特征名，特征最大值（index），特征embedding维度
    # ====================================================
    sparse_feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                              for feat in sparse_features]
    dense_feature_columns = [denseFeature(feat) for feat in dense_features]
    # 将 sparse_features 放在前面
    data_df_new = data_df[sparse_features + dense_features + ['label']]
    train, test = train_test_split(data_df_new, test_size=test_size)

    train_X = train[sparse_features + dense_features].values.astype('float32')
    train_y = train['label'].values.astype('int32')
    test_X = test[sparse_features + dense_features].values.astype('float32')
    test_y = test['label'].values.astype('int32')


    model = DCN(sparse_feature_columns, dense_feature_columns, hidden_units, dnn_dropout=dnn_dropout)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = 'save/dcn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ===========================Fit==============================
    # earlystopping 含义 https://subaochen.github.io/tensorflow/2019/07/21/tensorflow-earlystopping/
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True), tf.keras.callbacks.TensorBoard(log_dir='.\logs')]
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=callbacks,  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
