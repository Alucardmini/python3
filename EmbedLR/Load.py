# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/17 8:34 AM'


import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

path = "/Users/wuxikun/Downloads/word2vec-master/vec.txt"
word2vecModel = KeyedVectors.load_word2vec_format(path)
print(word2vecModel.wv['zero'])
res = word2vecModel.most_similar(['zero', 'one'], topn=3)
print(res)