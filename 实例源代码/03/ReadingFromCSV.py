# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:56:57 2018

@author: Administrator
"""

import tensorflow as tf
file_name_string="airline.csv"#要读取的csv格式的文件名
filename_queue = tf.train.string_input_producer([file_name_string])

#每次一行
reader = tf.TextLineReader()
key,value = reader.read(filename_queue)
record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]] # 这里的数据类型决定了读取的数据类型，而且必须是list形式
col1, col2, col3, col4, col5,col6 = tf.decode_csv(value, record_defaults=record_defaults) # 解析出的每一个属性都是rank为0的标量

with tf.Session() as sess:
    #线程协调器
    coord = tf.train.Coordinator()
    #启动线程
    threads = tf.train.start_queue_runners(coord=coord)
    is_second_read=0
    line1_name=bytes('%s:1' % file_name_string, encoding='utf8')
    print (line1_name)
    while True:
       #x1第一个数据，x2第二个数据，line_label中保存当前读取的行号
        x1,x2,x3,x4,x5,x6,line_label = sess.run([col1, col2, col3, col4, col5,col6, key])
        #若当前line_label第二次等于第一行的label(即line1_name)则说明读取完，跳出循环
        if is_second_read==0 and line_label==line1_name:
            is_second_read=1
        elif is_second_read==1 and line_label==line1_name:
            break
        print ( x1,x2,x3,x4,x5,x6,line_label)
    coord.request_stop()
    coord.join(threads)#循环结束后，请求关闭所有线程
    sess.close()