# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:51:53 2018

@author: Administrator
"""

import tensorflow as tf


def Save_model():
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1= tf.Variable(5.0,name="b1")
    w3 = tf.add(w1,w2)
    w4 = tf.multiply(w3,b1,name="op_to_restore") #计算(w1+w2)*5
    feed_dict ={w1:1,w2:2}
    
    sess = tf.Session() 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print (sess.run(w4,feed_dict))
    saver.save(sess, './my_test_model',global_step=1000)
    sess.close()
    print('--------------')


def Load_model():
    sess=tf.Session()    
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict ={w1:13.0,w2:17.0}
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    print (sess.run(op_to_restore,feed_dict))
    sess.close()
    print('****************')




if __name__ == '__main__':
     Save_model()
     Load_model()