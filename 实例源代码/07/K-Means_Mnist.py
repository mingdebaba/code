# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:51:27 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import input_data
from random import randint
from collections import Counter


k = 10 # 类别数目
MAX_ITERS = 100 # 最大迭代次数
#num_features = 784  # 每个图片都是28X28，共784个像素

# 导入MNIST数据集
mnist = input_data.read_data_sets('data/', one_hot=True) 
#使用训练集的图片数据为输入数据
X=mnist.train.images # shape:(55000, 784)，注意记住这个55000，理解后面会用到
N = mnist.train.num_examples # 样本点数目
#记录训练集的真实标签数据，为了测试准备率
y_=mnist.train.labels
y = []
for m in range(N):
    for n in range(10):
        if(y_[m][n]==1):
            y.append(n)


#获取初始质心
start_pos = tf.Variable(X[np.random.randint(X.shape[0], size=k),:], dtype=tf.float32)
centroids = tf.Variable(start_pos.initialized_value(), 'S', dtype=tf.float32)
#
# 输入值
points           = tf.Variable(X, 'X', dtype=tf.float32)
ones_like        = tf.ones((points.get_shape()[0], 1))
prev_assignments = tf.Variable(tf.zeros((points.get_shape()[0], ), dtype=tf.int64))

# 获取距离
p1 = tf.matmul(
    tf.expand_dims(tf.reduce_sum(tf.square(points), 1), 1),
    tf.ones(shape=(1, k))
)
p2 = tf.transpose(tf.matmul(
    tf.reshape(tf.reduce_sum(tf.square(centroids), 1), shape=[-1, 1]),
    ones_like,
    transpose_b=True
))
distance = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(points, centroids, transpose_b=True))


point_to_centroid_assignment = tf.argmin(distance, axis=1)

# 计算均值
total = tf.unsorted_segment_sum(points, point_to_centroid_assignment, k)
count = tf.unsorted_segment_sum(ones_like, point_to_centroid_assignment, k)
means = total / count
#中心点是否变化
is_continue = tf.reduce_any(tf.not_equal(point_to_centroid_assignment, prev_assignments))

with tf.control_dependencies([is_continue]):
    loop = tf.group(centroids.assign(means), prev_assignments.assign(point_to_centroid_assignment))



sess = tf.Session()
sess.run(tf.global_variables_initializer())
changed = True
iterNum = 0
while changed and iterNum < MAX_ITERS:
    iterNum += 1
    # 运行graph
    [changed, _] = sess.run([is_continue, loop])
    res = sess.run(point_to_centroid_assignment)
    print(iterNum)
print("train finished")

   
# 评估.获取每个簇所有的点，按照其真实标签的前三数量
nums_in_clusters = [[] for i in range(10)]
for i in range(N):
    nums_in_clusters[res[i]].append(y[i])
for i in range(10):
    print (Counter(nums_in_clusters[i]).most_common(3))