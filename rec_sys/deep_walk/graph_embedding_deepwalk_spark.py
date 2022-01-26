#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# 2020.11


# https://spark.apache.org/docs/2.4.0/api/python/pyspark.ml.html
# https://spark.apache.org/docs/2.4.0/ml-features.html#word2vec

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pyspark.ml.feature import Word2Vec
import pandas as pd
import math
import random
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession 
from pyspark.sql.functions import *
from pyspark.sql import Row
import  time


def random_choice(line):
    """
    从 adjacency_list 中选择一个 item, 并添加到路径 path 中
    
    :return: (next_node, path_list)
    """
    now_node = line[0]
    path_list = line[1][0]
    adjacency_list = line[1][1]
    # random choice a item from adjacency_list
    next_node = random.choice(adjacency_list)
    
    # 将选择的节点加入路径
    path_list.append(next_node)
    
    # 当前路径的 join 节点为 next_node
    return (next_node, path_list)
    

def generate_random_walks(item_ids, adjacency_list, num_walks=10, len_walks=30):
    """
    convenience method to generate a list of numWalks random walks. This saves a random walk in targetPath.
    :param item_ids: an RDD of item_ids for which the random walks should be generated.
    :param adjacency_list: a simple RDD with tuples of the form (item_id, [list(id)]).
    :param num_walks: optional. The number of walks, which are to be generated for each item id.
    :param len_walks: optional. The maximum length of each walk.
    :return: a RDD of random walks
    """
    # 初始化 walker 列表 rdd，列表中为键值对 RDD  item_id,  random_walk_path:  []
    walkers = item_ids.flatMap(lambda item_id: [(item_id, [item_id])] * num_walks)
    # 去掉自己 （长度为 walks - 1 ）
    # coalesce(numPartitions) 将RDD中的分区数量减少到numPartitions
    for _ in range(len_walks - 1):
        walkers = walkers \
            .leftOuterJoin(adjacency_list) \
            .map(random_choice) \
            .coalesce(50)
        
    # 返回所有随机游走的路径  random_walk_path:  []
    return walkers.map(lambda x: x[1])




if __name__ == "__main__":

    # 读取数据
    # input:   item1, item2 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------reading data----------------------------------')
    # 获取 spark session
    spark = SparkSession.builder.config('spark.driver.extraClassPath','/home/hadoop/gdcconf/hive/formal/us').config('spark.driver.maxResultSize','32g').config("spark.driver.memory", '32g').appName('deepwalk spark').enableHiveSupport().getOrCreate()
    sc=spark.sparkContext
    df_social =spark.sql("select * from ${t1}")
    social_rdd = df_social.rdd

    # 输入数据是好友关系，并且两两之间的好友关系只有一个记录
    # 1. 需要将好友关系复制一份，变为 A,B  变为  A,B  B,A
    # 注： 如果输入数据是 <节点：邻接节点列表的形式>  =>  <item1, [item2,item3,item4....]>，则无需此操作
    social_rdd = social_rdd.map(lambda x: (str(x[0]), str(x[1]))).flatMap(lambda p: [(p[0], p[1]), (p[1], p[0])])
    
    # 2. 计算得到所有图中的节点
    all_user_nodes = social_rdd.groupByKey().keys()
    # 3. 得到节点和邻接节点
    nodes_adjacency_map = social_rdd.groupByKey().map(lambda line: (line[0] , [x for x in line[1]]))


    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------start random walking----------------------------------')

    walks_sentences = generate_random_walks(all_user_nodes, nodes_adjacency_map, num_walks=8, len_walks=30)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'nodes num:', all_user_nodes.count())
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'train sentence length:', walks_sentences.count())

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------start convert rdd to dataframe----------------------------------')
    # 推断并创建 dataframe， 并注册成为 table
    # rdd 2 dataframe
    walks_sentences_row = walks_sentences.map(lambda p:Row(sentence=p))
    tran_df = spark.createDataFrame(walks_sentences_row)
    tran_df.printSchema()
    tran_df.show()
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------start traning----------------------------------')
    # 迭代次数小于分区数
    # Sets number of iterations (default: 1), which should be smaller than or equal to number of partitions.
    # Sets number of partitions (default: 1). Use a small number for accuracy.
    # To make our implementation more scalable, we train each partition separately and merge the model of each partition after each iteration. To make the model more accurate, multiple iterations may be needed.
    word2Vec = Word2Vec(vectorSize=128, minCount=0, numPartitions=10, stepSize=0.025, maxIter=3, seed=2018, inputCol="sentence")
    model = word2Vec.fit(tran_df)
    result_model = model.getVectors()
    result_model.show()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------save embeddings----------------------------------')


    # 直接通过 spark sql 写hive 表
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------start insert to table----------------------------------')

    alg_name = 'friend_graph_embedding_deepwalk'
    result_model.createOrReplaceTempView('tmptb_pro')
    dim_num = 128
    spark.sql("SET hive.hadoop.supports.splittable.combineputformat=false;")
    insert_sql="insert overwrite table xxxxxx partition(dt="+str(${vDate})+",alg_name='"+alg_name+"') select word , vector, "+str(dim_num)+" from tmptb_pro"
    spark.sql(insert_sql)