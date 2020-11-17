# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:35:52 2018

@author: Administrator
"""

import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
  
data=[]  
label=[]
loss_vec=[] 
 
np.random.seed(0) 
  
##随机产生训练集  
for i in range(150):  
    x1=np.random.uniform(-1,1)  
    x2=np.random.uniform(0,2)  
    if x1*2+ x2<=1:   
        data.append([np.random.normal(x1,0.1),np.random.normal(x2,0.1)])  
        label.append(0)
        plt.plot(data[i][0],data[i][1],'go')
    else:  
        data.append([np.random.normal(x1,0.1),np.random.normal(x2,0.1)])  
        label.append(1) 
        plt.plot(data[i][0],data[i][1],'r*') 
##绘制图形
data=np.hstack(data).reshape(-1,2)  
label=np.hstack(label).reshape(-1,1)
plt.show()  
  

#定义变量
x=tf.placeholder(tf.float32,shape=(150,2))  
y_=tf.placeholder(tf.float32,shape=(None,1)) 

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))



y = (tf.matmul(x, W) + b)
#计算L2范数
l2_norm = tf.reduce_sum(tf.square(W))
#损失函数
alpha = tf.constant([0.1])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y, y_))))
cross_entropy = tf.add(classification_term, tf.multiply(alpha, l2_norm))

#训练模型
#优化器使用梯度下降
learning_rate = 0.01    #学习率
cost_prev=0
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(4000):
   sess.run(train_step, feed_dict={x:data, y_:label})
   train_cost=sess.run(cross_entropy, feed_dict={x:data, y_:label})
   loss_vec.append(train_cost)
   if i % 200==0:        
       print (i,sess.run([W,b,cross_entropy],{x:data, y_:label}))
   if np.abs(cost_prev-train_cost)<1e-6:
       print ('the step:',i,sess.run([W,b,cross_entropy],{x:data, y_:label}))
       break
   else:
       cost_prev=train_cost
#记录最终的w、b值    
W_val=sess.run(W)
b_val=sess.run(b)
sess.close()

#绘制直线和散点图
w1=W_val[0,0]
w2=W_val[1,0]
k=-w1/w2
b=b_val
xx=np.linspace(-1,1.2,100)
yy=k*xx+b
plt.plot(xx,yy)
for i in range(150):  
    if( label[i]==0):        
        plt.plot(data[i][0],data[i][1],'go')
    else:        
        plt.plot(data[i][0],data[i][1],'r*') 
    

plt.show() 