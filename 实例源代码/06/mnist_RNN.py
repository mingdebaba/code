# -*- coding: utf-8 -*-
"""
Created on Sun May 13 08:55:50 2018

@author: Administrator
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
#在这里做数据加载，
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# RNN学习时使用的参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 20

# 神经网络的参数
n_input = 28  # 输入层的n
n_steps = 28  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 10  # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

# 构建tensorflow的输入X的placeholder
x = tf.placeholder("float32", [None, n_steps, n_input])
# 输出Y
y = tf.placeholder("float32", [None, n_classes])

# 随机初始化每一层的权值和偏置
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.constant(0.1,shape=([n_hidden,]))),
    'out': tf.Variable(tf.constant(0.1,shape=([n_classes,])))
}

'''
构建RNN
'''
def RNN(_X,  _weights, _biases):   

    _X = tf.reshape(_X, [-1, n_input])  
    # 输入层到隐含层，第一次是直接运算
    X_in = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    #规则数据
    X_in =tf.reshape(X_in,[-1,n_steps,n_hidden])
    #之后使用LSTM
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
    #初始化
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # 开始跑RNN那部分
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state,time_major=False)
    # 输出层
    results=tf.matmul(final_state[1], _weights['out']) + _biases['out']
    return results


pred = RNN(x,  weights, biases)

# 定义损失和优化方法，其中算是为softmax交叉熵，优化方法为Adam
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=pred, labels=y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # Adam Optimizer

# 进行模型的评估，argmax是取出取值最大的那一个的标签作为输出
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()
# 开始运行
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # 持续迭代
    while step * batch_size < training_iters:
        # 随机抽出这一次迭代训练时用的数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 对数据进行处理，使得其符合输入
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       })
        # 在特定的迭代回合进行数据的输出
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, })
            print('step %d, training accuracy %g' % (step, acc))
        step += 1
   