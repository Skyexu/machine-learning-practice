#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:xukaihui
# 2020.11


# pyspark==2.4.0,pandas,gensim==3.6.0,networkx==2.1,joblib==0.13.0,tqdm,numpy

from gensim.models import Word2Vec
import pandas as pd
import itertools
import math
import random
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyspark.sql import SparkSession 
from pyspark.sql.functions import *
import  time


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

class RandomWalker:
    def __init__(self, G):
        """
        :param G: graph
        """
        self.G = G

    def deepwalk_walk(self, walk_length, start_node):
        """
        以 start_node 出发随机游走 walk_length 的长度
        """
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def partition_num(self, num, workers):
        """
        用于并行任务切分
        """
        if num % workers == 0:
            return [num//workers]*workers
        else:
            return [num//workers]*workers + [num % workers]


    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        """
        并行任务进行随游走
        """
        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            self.partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        """
        某一任务随机游走 num_walks 次
        """
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=v))
        return walks


if __name__ == "__main__":
    # 获取spark session
    spark = SparkSession.builder.config('spark.driver.extraClassPath','/home/hadoop/gdcconf/hive/formal/us').config('spark.driver.maxResultSize','16g').config("spark.driver.memory", '16g').appName('deepwalk local').enableHiveSupport().getOrCreate()
    sc=spark.sparkContext

    # 1. 读取数据构建图   
    # input:   item1 item2 
    # 输入为 item1 连接到 item2 的边
    G_social = nx.DiGraph()
 
    df_social =spark.sql("select * from ${t1}")
    social_pair = df_social.collect()
    for line in social_pair:
         G_social.add_edge(str(line[0]), str(line[1]))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------reading graph----------------------------------')

    # for line in sys.stdin:
    #     role_id, friend_id = line.strip().split('\t')
    
    # for row in pd_social.iterrows():
    #     G_social.add_edge(role_id, friend_id)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'graph nodes:', len(G_social.nodes))

    print('---------------------------start random walking----------------------------------')

    # 2. 随机游走构建序列
    model_social = DeepWalk(G_social, walk_length=30, num_walks=8, workers=4)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'train sentence length:', len(model_social.sentences))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------start traning----------------------------------')

    # 3. word2vec 模型训练
    model_social.train(embed_size=128, window_size=10, iter=5)

    count = 0
    for sent in model_social.sentences:
        # 将句子去重，计算节点数量
        if len(set(sent)) == 2:
            count += 1
            #print(sent)
    print('equal 2 sentence num:', count)
    print('equal 2 sentence rate:', count / len(model_social.sentences))
    print('sentences length :',  len(model_social.sentences))
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '---------------------------save embeddings----------------------------------')

    # 4. 写 hive 表
    # 用日志输出的形式传到算子输出
    # for role in G_social.nodes():
    #     embeddings = model_social.w2v_model.wv[role]
    #     print ('data:'+'\t'.join([role, str(embeddings.tolist())]), '128')


    # 直接通过 spark sql 写hive 表

    list_df=[]
    for role in G_social.nodes():
        embeddings = model_social.w2v_model.wv[role]
        list_df.append((role, str(embeddings.tolist()), 128))

    df_list = spark.createDataFrame(list_df,["role_id", "embedding", "dim_num"])
    df_list.show()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'start insert to table')
    alg_name = 'friend_graph_embedding_deepwalk'
    df_list.createOrReplaceTempView('tmptb_pro')
    spark.sql("SET hive.hadoop.supports.splittable.combineputformat=false;")
    insert_sql="insert overwrite table xxxx partition(dt="+str(${vDate})+",alg_name='"+alg_name+"') select role_id, embedding, dim_num from tmptb_pro"
    spark.sql(insert_sql)