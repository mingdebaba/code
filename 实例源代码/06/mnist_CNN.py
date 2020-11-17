# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:33:05 2018

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
import sys
import input_data
import tempfile

FLAGS = None



#构造神经网络，网络结构为：数据输入层–卷积层1–池化层1–卷积层2–池化层2–全连接层1–全连接层2（输出层），
def CNN_mnist(x):
 
  with tf.name_scope('reshape'):      
# 对图像做预处理，将1D的输入向量转为2D的图片结构，即1*784到28*28的结构,-1代表样本数量不固定，1代表颜色通道数量
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # # 定义第一个卷积层以及池化层
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层以及池化层
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

# fc1，将两次池化后的特征图转换为1D向量，隐含节点1024由自己定义
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减轻过拟合，使用Dropout层
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Dropout层输出连接一个输出层
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print("CNN READY")
  return y_conv, keep_prob

# 定义好初始化函数以便重复使用。给权重制造一些随机噪声来打破完全对称，使用截断的正态分布，标准差设为0.1，
# 同时因为使用relu，也给偏执增加一些小的正值(0.1)用来避免死亡节点(dead neurons)
def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # 参数分别指定了卷积核的尺寸、多少个channel、filter的个数即产生特征图的个数

# 2x2最大池化，即将一个2x2的像素块降为1x1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征。
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    
#加载Mnist数据
    mnist = input_data.read_data_sets('data/', one_hot=True)
    trainimg = mnist.train.images
    trainlabel = mnist.train.labels
    testimg = mnist.test.images
    testlabel = mnist.test.labels
    print("MNIST ready")

    sess = tf.InteractiveSession()
    n_input  = 784 # 28*28的灰度图，像素个数784
    n_output = 10  # 是10分类问题

# 在设计网络结构前，先定义输入的placeholder，x是特征，y是真实的label
    x = tf.placeholder(tf.float32, [None, n_input]) 
    y_ = tf.placeholder(tf.float32, [None, n_output]) 
#
    y_conv, keep_prob = CNN_mnist(x)
# 定义损失函数
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
# 优化器
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 定义评测准确率的操作
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    training_epochs = 1001 # 所有样本迭代1000次
    batch_size = 100 # 每进行一次迭代选择100个样本
    display_step = 100
# 初始化所有参数
    init=tf.global_variables_initializer()
    sess = tf.Session() 
    sess.run(init) 

    for i in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.7})
               
        
        if i % display_step ==0: # 每100次训练，对准确率进行一次测试
            train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))        
            test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, 
                                                 y_:mnist.test.labels, keep_prob:1.0})
            print('test accuracy %g' % (test_accuracy))
    sess.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)