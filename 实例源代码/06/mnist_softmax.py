# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:15:47 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import input_data

#加载Mnist数据
print('Download and Extract MNIST dataset')
mnist = input_data.read_data_sets('data/', one_hot=True) # one_hot=True意思是编码格式为01编码
print("number of train data is %d" % (mnist.train.num_examples))
print("number of test data is %d" % (mnist.test.num_examples))
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

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

x_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) 

#没有隐藏层，构建输出层
predition=add_layer(x_,784,10,activation_function=None)

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=predition))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer() # 全局参数初始化器

training_epochs = 101 # 所有样本迭代100次
batch_size = 100 # 每进行一次迭代选择100个样本
display_step = 5
# SESSION
sess = tf.Session() 
sess.run(init) 
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    #训练过程
    avg_cost = 0. 
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch): # 每一个batch进行选择
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 通过next_batch()就可以一个一个batch的拿数据，
        sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys}) # run一下用梯度下降进行求解，通过placeholder把x，y传进来
        avg_cost += sess.run(cross_entropy, feed_dict={x_: batch_xs, y_:batch_ys})/num_batch
    #训练一定程度后，用模型去预测测试数据
    if epoch % display_step == 0: # 
        correct_prediction = tf.equal(tf.argmax(predition, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_acc=sess.run(accuracy, feed_dict={x_: mnist.test.images,
                                      y_: mnist.test.labels})
        print("Epoch: %d/%d cost: %f  TEST ACCURACY: %f"
              % (epoch, training_epochs, avg_cost, test_acc))
sess.close()