import jieba
import re
import numpy as np
from gensim.models import word2vec

zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

def cut():
    input_file = open('../data/wiki_simplify', 'r')
    out_file = open('../data/wiki_cut.txt', 'w')
    count = 0

    while True:
        line = input_file.readline()
        if not line:
            break
        if line.startswith('<') or line.startswith('\n'):
            continue
        cut = jieba.cut(line)
        
        for word in cut:
            match = zhPattern.search(word)
            if match or word  == '\n':
                if word != '\n':
                    out_file.write(word + ' ')
                else:
                    out_file.write('\n')

        count += 1
        print(count)

def embedding():
    sentence = word2vec.LineSentence('../data/wiki_cut.txt')
    model = word2vec.Word2Vec(sentences=sentence, min_count=5, window=5, iter=3, size=300)
    model.save('../model/word2vec.model')

def inference():
    model = word2vec.Word2Vec.load('../model/word2vec.model')

    req_count = 5
    for key in model.wv.similar_by_word('语文'):
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

if __name__ == "__main__":
    #cut()
    #embedding()
    inference()
