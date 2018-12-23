import jieba
import re
import numpy as np
from gensim.models import word2vec

zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

def process():
    
    #for phase in ['train', 'validation']:
    for phase in ['test']: 
        input_name = '../raw_data/' + phase + '-set.data'
        output_Q_name = '../data/split_data/' + phase + '_Q.txt'
        output_A_name = '../data/split_data/' + phase + '_A.txt'
        #output_L_name = '../data/numpy_array/' + phase + '_label.npy'
        input_file =  open(input_name, 'r')
        output_Q = open(output_Q_name, 'w')
        output_A = open(output_A_name, 'w')
        #output_L = []
        sum_count = 0
        count = 0

        while True:
            line = input_file.readline()
            if not line:
                break
            sum_count += 1
            if (len(line.split('\t')) != 2):
                print(line)
                continue
            count += 1
            #print(count)
            Q = line.split('\t')[0]
            A = line.split('\t')[1]    
            #L = line.split('\t')[2]
            #if L.endswith('%'):
            #    L = L[0]
            #L = int(L)

            Q_cut = jieba.cut(Q)
            for word in Q_cut:
                match = zhPattern.search(word)
                if match:
                    output_Q.write(word + ' ')
            output_Q.write('\n')

            A_cut = jieba.cut(A)
            for word in A_cut:
                match = zhPattern.search(word)
                if match:
                    output_A.write(word + ' ')
            output_A.write('\n')

            #output_L.append(L)
        
        #output_L = np.array(output_L)
        #print(output_L.shape)
        #np.save(output_L_name, output_L)
        input_file.close()
        output_Q.close()
        output_A.close()
        print(count)
        print(sum_count)

def cut():
    input_file = open('../data/raw_data/wiki_simplify', 'r')
    output_file = open('../data/split_data/wiki_cut.txt', 'w')
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
                    output_file.write(word + ' ')
                else:
                    output_file.write('\n')

        count += 1
        print(count)

    input_file.close()
    output_file.close()

def cut_ori():
    input_file_name = ['../data/split_data/train_A.txt', '../data/split_data/train_Q.txt', '../data/split_data/validation_A.txt', '../data/split_data/validation_Q.txt']
    output_file = open('../data/split_data/tv_cut.txt', 'w')

    for name in input_file_name:
        input_file = open(name, 'r')
        while True:
            line = input_file.readline()
            if not line:
                break
            
            cut = jieba.cut(line)
            for word in cut:
                match = zhPattern.search(word)
                if match or word  == '\n':
                    if word != '\n':
                        output_file.write(word + ' ')
                    else:
                        output_file.write('\n')
    
    input_file.close()
    output_file.close()
    

def embedding():
    sentence = word2vec.LineSentence('../data/split_data/tv_cut.txt')
    model = word2vec.Word2Vec(sentences=sentence, min_count=5, window=5, iter=3, size=300)
    model.save('../model/new_word2vec.model')

def inference():
    model = word2vec.Word2Vec.load('../model/new_ord2vec.model')
    outfile = open('../data/split_data/new_words.txt', 'w')
    d = []
    maxlen = 0
    word_vector = []
    word_vector.append(np.zeros((300)))
    for phase in ['train', 'validation']:
        for mode in ['Q', 'A']:
            count = 0
            filename = '../data/split_data/' + phase + '_' + mode + '.txt'
            print(filename)
            file = open(filename, 'r')
            while True:
                line = file.readline()
                count += 1
                print(count)
                if not line:
                    break
                words = line.split()
                if len(words) > maxlen:
                    maxlen = len(words)
                for word in words:
                    if word in model.wv.vocab.keys() and word not in d:
                        d.append(word)
                        outfile.write(word + '\n')
                        word_vector.append(model.wv[word])
            file.close()
    outfile.close()
    np.save('../data/numpy_array/new_word_vector.npy', word_vector)
    print(maxlen)

def build_matrix():
    file = open('../data/split_data/words.txt', 'r')
    d = []
    d.append('')
    while True:
        line = file.readline()
        if not line:
            break
        d.append(line[:-1])
    file.close()

    #for phase in ['train', 'validation']:
    for phase in ['test']:
        for mode in  ['Q', 'A']:
            index = []
            filename = '../data/split_data/' + phase + '_' + mode + '.txt'
            line_count = 0
            file = open(filename, 'r')
            while True:
                line = file.readline()
                line_count += 1
                print(filename, line_count)
                if not line:
                    break
                tmp = np.zeros((200))
                word_count = 0
                words = line.split()
                for word in words:
                    if word in d:
                        if word_count >= 200:
                            break
                        tmp[word_count] = d.index(word)
                        word_count += 1
                index.append(tmp)
            index = np.array(index)
            savename = '../data/numpy_array/' + phase + '_' + mode + '_index.npy'
            np.save(savename, index)

def test():
    model = word2vec.Word2Vec.load('../model/word2vec.model')
    print(model.wv['年'])

if __name__ == "__main__":
    process()
    #cut()
    #cut_ori()
    #embedding()
    #inference()
    build_matrix()
    #test()
