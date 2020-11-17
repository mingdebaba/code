# -*- coding: utf-8 -*-
"""
Created on Tue May 15 19:56:23 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt

leraning_rate=0.01 #学习率
training_epochs=20
batch_size=256
display_step=1
#测试数据
examples_to_show=10

#无监督学习，只使用训练图片
mnist = input_data.read_data_sets('data/', one_hot=True) 

#网络模型，两个隐藏层，
n_hidden1=256
n_hidden2=128
n_input=784

weights={
        'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden1]) ),
        'encoder_h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2]) ),
        'decoder_h1':tf.Variable(tf.random_normal([n_hidden2,n_hidden1]) ),
        'decoder_h2':tf.Variable(tf.random_normal([n_hidden1,n_input]) ),       
        
        }

biases={
        'encoder_b1':tf.Variable(tf.random_normal([n_hidden1]) ),
        'encoder_b2':tf.Variable(tf.random_normal([n_hidden2]) ),
        'decoder_b1':tf.Variable(tf.random_normal([n_hidden1]) ),
        'decoder_b2':tf.Variable(tf.random_normal([n_input]) ),       
        
        }
#压缩函数
def encoder(x):
    layer1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encoder_h2']),biases['encoder_b2']))
    return layer2

#解压函数
def decoder(x):
    layer1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['decoder_h2']),biases['decoder_b2']))
    return layer2

 

x_ = tf.placeholder(tf.float32, [None, 784])

#构建模型
encoder_op=encoder(x_)
decoder_op=decoder(encoder_op)

#预测值
y_pred=decoder_op
#真实值
y_true=x_
#损失函数
cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.RMSPropOptimizer(leraning_rate).minimize(cost)

#训练
init = tf.global_variables_initializer() # 全局参数初始化器
sess = tf.Session() 
sess.run(init) 
total_batch=int(mnist.train.num_examples/batch_size)
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    for i in range(total_batch):
         batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
         _,c=sess.run([optimizer,cost],feed_dict={x_: batch_xs})
    if epoch % display_step==0:
         print(epoch,"cost=",c)
print("train Finished")
#从Mnist测试集中选择进行测试
encoder_decode=sess.run(y_pred,feed_dict={x_: mnist.test.images[:examples_to_show] })
sess.close()
#比较结果
f,a=plt.subplots(2,10,figsize=(10,2))

for i in range(examples_to_show):
    #绘制测试集本身
    a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
    a[1][i].imshow(np.reshape(encoder_decode[i],(28,28)))
f.show()
plt.draw()
plt.waitforbuttonpress()

    
    
    

