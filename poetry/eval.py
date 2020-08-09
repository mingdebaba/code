# -*- coding: utf-8 -*-
import sys

import tensorflow as tf
import numpy as np
from rnn_model import EvalModel



if __name__ == '__main__':
    model = EvalModel()  # 创建训练模型
    #生成诗歌，poemtype=poem随机生成，head藏头诗
    model.get_poem('poem','poem')
    model.get_poem('head','生日快乐')
    

