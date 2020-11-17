# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import dataset
import utils


HIDDEN_SIZE = 128  # LSTM隐藏节点个数
NUM_LAYERS = 2  # RNN深度
TRAIN_TIMES = 30000  # 迭代总次数（没有计算epoch）
SHOW_STEP = 1  # 显示loss频率
SAVE_STEP = 100  # 保存模型参数频率
VOCAB_SIZE = 6272  # 词汇表大小
MAX_GRAD = 5.0  # 最大梯度，防止梯度爆炸
LEARN_RATE = 0.0005  # 初始学习率
LR_DECAY = 0.92  # 学习率衰减
LR_DECAY_STEP = 600  # 衰减步数
BATCH_SIZE = 64  # batch大小
CKPT_PATH = 'ckpt/model_ckpt'  # 模型保存路径
VOCAB_PATH = 'vocab/poetry.vocab'  # 词表路径
EMB_KEEP = 0.5  # embedding层dropout保留率
RNN_KEEP = 0.5  # lstm层dropout保留率


class TrainModel(object):
   
    def train(self):
        tf.reset_default_graph()
        x_data = tf.placeholder(tf.int32, [BATCH_SIZE, None])  # 输入数据
        y_data = tf.placeholder(tf.int32, [BATCH_SIZE, None])  # 标签
        emb_keep = tf.placeholder(tf.float32)  # embedding层dropout保留率
        rnn_keep = tf.placeholder(tf.float32)  # lstm层dropout保留率

        data = dataset.Dataset(BATCH_SIZE)  # 创建数据集
        global_step = tf.Variable(0, trainable=False)
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=rnn_keep) for
            _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        #RNN
        # 创建词嵌入矩阵权重
        embedding = tf.get_variable('embedding', shape=[VOCAB_SIZE, HIDDEN_SIZE])
        # 创建softmax层参数
        softmax_weights = tf.get_variable('softmaweights', shape=[HIDDEN_SIZE, VOCAB_SIZE])
        softmax_bais = tf.get_variable('softmax_bais', shape=[VOCAB_SIZE])
        # 进行词嵌入
        emb = tf.nn.embedding_lookup(embedding, x_data)
        # dropout
        emb_dropout = tf.nn.dropout(emb, emb_keep)
        # 计算循环神经网络的输出
        init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, emb_dropout, scope='d_rnn', dtype=tf.float32,
                                                initial_state=init_state)
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])
        # 计算logits
        logits = tf.matmul(outputs, softmax_weights) + softmax_bais
        #损失函数
        # 计算交叉熵
        outputs_target = tf.reshape(y_data, [-1])
        coss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs_target, )
        # 平均值
        loss = tf.reduce_mean(coss)
        # 学习率衰减
        learn_rate = tf.train.exponential_decay(LEARN_RATE, global_step, LR_DECAY_STEP,
                                                LR_DECAY)
        # 计算梯度，并防止梯度爆炸
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_variables), MAX_GRAD)
        # 创建优化器，进行反向传播
        optimizer = tf.train.AdamOptimizer(learn_rate)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step)
        #开始训练
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 初始化
            for step in range(TRAIN_TIMES):
                # 获取训练batch
                x, y = data.next_batch()
                # 计算loss
                Loss, _ = sess.run([loss, train_op],
                           feed_dict={x_data: x, y_data:y, emb_keep:EMB_KEEP,
                            rnn_keep:RNN_KEEP})
                if step % SHOW_STEP == 0:
                    print ('step {}, loss is {}'.format(step, Loss))
                # 保存模型
                if step % SAVE_STEP == 0:
                    saver.save(sess, CKPT_PATH, global_step=global_step)
        


class EvalModel(object):
    """
    验证模型
    """
    
    def get_poem(self,poemtype,poemstr):
        
        print(poemtype,poemstr)
        #生成诗歌，poemtype=poem随机生成，head藏头诗
        x_data = tf.placeholder(tf.int32, [1, None])

        emb_keep = tf.placeholder(tf.float32)

        rnn_keep = tf.placeholder(tf.float32)

        saver = tf.train.Saver()
        # 单词到id的映射
        word2id_dict = utils.read_word_to_id_dict()                
        # id到单词的映射
        id2word_dict = utils.read_id_to_word_dict()
        # 验证用模型
        embedding = tf.get_variable('embedding', shape=[VOCAB_SIZE, HIDDEN_SIZE])

        softmax_weights = tf.get_variable('softmaweights', shape=[HIDDEN_SIZE, VOCAB_SIZE])
        softmax_bais = tf.get_variable('softmax_bais', shape=[VOCAB_SIZE])

        emb = tf.nn.embedding_lookup(embedding, x_data)
        emb_dropout = tf.nn.dropout(emb, emb_keep)
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=rnn_keep) for
            _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        # 与训练模型不同，这里只要生成一首古体诗，所以batch_size=1
        init_state = cell.zero_state(1, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, emb_dropout, scope='d_rnn', dtype=tf.float32,
                                                initial_state=init_state)
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])

        logits = tf.matmul(outputs, softmax_weights) + softmax_bais
        probs = tf.nn.softmax(logits)        
      
        with tf.Session() as sess:
            # 加载最新的模型
            ckpt = tf.train.get_checkpoint_state('ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)
            if poemtype=='poem':
            #随机生成一首诗歌
            #预测第一个词
                rnn_state = sess.run(cell.zero_state(1, tf.float32))
                x = np.array([[word2id_dict['s']]], np.int32)                
                #与训练模型不同，这里要记录最后的状态，以此来循环生成字，直到完成一首诗
                prob, rnn_state = sess.run([probs, last_state],
                                   {x_data: x, init_state: rnn_state, emb_keep: 1.0,
                                    rnn_keep: 1.0})
                idword = sorted(prob, reverse=True)[:100]
                index = np.searchsorted(np.cumsum(idword), np.random.rand(1) * np.sum(idword))                
                word = id2word_dict[int(index)]
                poem = ''
                # 循环操作，直到预测出结束符号‘e’
                while word != 'e':
                    poem += word
                    x = np.array([[word2id_dict[word]]])
                    prob, rnn_state = sess.run([probs, last_state],
                                       {x_data: x, init_state: rnn_state, emb_keep: 1.0,
                                        rnn_keep: 1.0})                    
                    idword = sorted(prob, reverse=True)[:100]
                    index = np.searchsorted(np.cumsum(idword), np.random.rand(1) * np.sum(idword))                
                    word = id2word_dict[int(index)]
                # 打印生成的诗歌
                print (poem)
            if poemtype=='head':
                #生成藏头诗，进行预测
                rnn_state = sess.run(cell.zero_state(1, tf.float32))
                poem = ''
                cnt = 1
                # 一句句生成诗歌
                for x in poemstr:
                    word = x
                    while word != '，' and word != '。':
                        poem += word
                        x = np.array([[word2id_dict[word]]])
                        prob, rnn_state = sess.run([probs, last_state],
                                           {x_data: x, init_state: rnn_state, emb_keep: 1.0,
                                            rnn_keep: 1.0})                                           
                        idword = sorted(prob, reverse=True)[:100]
                        index = np.searchsorted(np.cumsum(idword), np.random.rand(1) * np.sum(idword))                
                        word = id2word_dict[int(index)]
                        if len(poem) > 25:
                            print ('bad.')
                            break
                    # 根据单双句添加标点符号
                    if cnt & 1:
                        poem += '，'
                    else:
                        poem += '。'
                    cnt += 1
                # 打印生成的诗歌
                print (poem)
                
        


   
