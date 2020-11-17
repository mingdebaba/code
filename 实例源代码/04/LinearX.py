# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:32:17 2018

@author: Administrator
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

learning_rate=0.001  #学习率
training_epochs=1000 #训练次数

np.set_printoptions(threshold='nan')                    #打印内容不限制长度
t_x = np.linspace(-1,1,50,dtype = np.float32)
noise = np.random.normal(0 , 0.05 ,t_x.shape)
noise=noise.astype(np.float32)
t_y = t_x * 3.0+5.0+noise


plt.plot(t_x,t_y,'k.')
plt.show()


x = tf.placeholder(tf.float32,name='x')
y = tf.placeholder(tf.float32,name='y')

a = tf.Variable(0.0)
b = tf.Variable(0.0)

curr_y = x * a +  b
loss =   tf.reduce_sum(tf.square(curr_y - y) )    #损失函数，实际输出数据和训练输出数据的方差
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)       #训练的结果是使得损失函数最小

sess = tf.Session()                                     #创建 Session
sess.run(tf.global_variables_initializer())             #变量初始化
for i in range(training_epochs):     

        sess.run(train, {x:t_x, y:t_y})
        if (i+1) % 50==0:        
            print (i,sess.run([a,b,loss],{x:t_x, y:t_y}))
           

a_val=sess.run(a)
b_val=sess.run(b)
print("this model is y=",a_val," * x +",b_val)
sess.close()

y_learned=t_x*a_val+b_val
plt.plot(t_x,t_y,'k.')
plt.plot(t_x,y_learned,'g-')
plt.show()
plt.close()


    
   