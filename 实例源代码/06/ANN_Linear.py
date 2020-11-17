# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:11:31 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#构建神经网络层
def add_layer(input,in_size,out_size,activation_function):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]))
    Wx_plus_b=tf.matmul(input,Weight)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#构造数据
t_x = np.linspace(-1,1,50,dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0 , 0.05 ,t_x.shape)
noise=noise.astype(np.float32)
t_y = t_x * 3.0+5.0+noise

plt.plot(t_x,t_y,'k.')
plt.show()

#构造模型
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#构建隐藏层，假设有十个神经元
l1=add_layer(x,1,10,activation_function=tf.nn.relu)

#构建输出层
predition=add_layer(l1,10,1,activation_function=None)
#损失函数
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-predition)
                                         ,reduction_indices=[1]
                                                ))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#训练模型
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={x:t_x,y:t_y})
    if (i+1)%50==0:
        print(i,sess.run([loss],feed_dict={x:t_x, y:t_y}))
      

#获取预测值
y_pre = sess.run(predition, feed_dict={x:t_x, y:t_y})

    
      
