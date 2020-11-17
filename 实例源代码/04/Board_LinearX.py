# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:32:17 2018

@author: Administrator
"""




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

learning_rate=0.5  #学习率
training_epochs=1000 #训练次数

np.set_printoptions(threshold='nan')                    #打印内容不限制长度


x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot(np.float32([0.100, 0.200]), x_data) + 0.300

#绘制三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # 创建一个三维的绘图工程
ax.scatter(x_data[0][:99],x_data[1][:99], y_data[:99], c='r')  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
plt.close()

x = tf.placeholder(tf.float32,[None,None],name='x')
y = tf.placeholder(tf.float32,[None,None],name='y')

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(W, x_data) + b 
loss = tf.reduce_mean(tf.square(y - y_data))

# 优化目标函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# 初始化所有变量
init =tf.global_variables_initializer()
 
with tf.Session() as sess:
    sess.run(init)
    for step in range(1,training_epochs):
        sess.run(train)        
        preW= sess.run(W)        
        preb= sess.run(b)
        if step % 20 == 0:
            print (step,'\n',preW[0][0],preW[0][1],'\n',preb[0])
sess.close()        

