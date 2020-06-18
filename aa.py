#  ! / user/ bin/ python
#  -*-  coding utf-8 -*-
# author : entang   
# time : 2020/4/27



import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from tensorflow import *
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input, Lambda
from keras.models import Model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import  jieba

word2vec_model = Word2Vec.load('./model/word2vec_model')


def stopwordslist(file):
    stopwords = [line.strip() for line in open(file, encoding='UTF-8').readlines()]
    return stopwords
def not_empty(s):
    return s and s.strip()
def process_text(dataset, file):
    cw = lambda x: list(jieba.cut(x))
    dataset['words_list'] = dataset['review'].apply(cw)
    stopwords = stopwordslist(file)
    sw = lambda x: x not in stopwords
    nw = lambda x: len(x) > 1
    sentences = []
    for sentence in dataset['words_list']:
        sentence = filter(sw, sentence)
        sentence = filter(not_empty, list(sentence)) #去空字符、none和空格
        sentence = filter(nw, list(sentence))
        sentences.append(list(sentence))
    dataset['words'] = sentences

    # tokenizer=Tokenizer()  #创建一个Tokenizer对象
    # tokenizer.fit_on_texts(dataset['words']) #fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    # vocab=tokenizer.word_index #得到每个词的编号
    # x_train, x_test, y_train, y_test = train_test_split(dataset['words'], dataset['label'], test_size=0.1)    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    # x_train_word_ids=tokenizer.texts_to_sequences(x_train)
    # x_test_word_ids = tokenizer.texts_to_sequences(x_test)    #序列模式    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    # x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=50) #将超过固定值的部分截掉，不足的在最前面用0填充
    # x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=50)

#
# # 预训练的词向量中没有出现的词用0向量表示
# # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# embedding_matrix = np.zeros((len(vocab) + 1, 300))
# for word, i in vocab.items():
#     try:
#         embedding_vector = word2vec_model[str(word)]
#         embedding_matrix[i] = embedding_vector
#     except KeyError:
#         continue
#
# # # 构建TextCNN模型
# def TextCNN_model_2(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix):
#     # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
#     main_input = Input(shape=(50,), dtype='float64')
#     # 词嵌入（使用预训练的词向量）
#     embedder = Embedding(len(vocab) + 1, 300, input_length=50, weights=[embedding_matrix], trainable=False)
#     # embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
#     embed = embedder(main_input)
#     # 词窗大小分别为3,4,5
#     cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
#     cnn1 = MaxPooling1D(pool_size=38)(cnn1)
#     cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
#     cnn2 = MaxPooling1D(pool_size=37)(cnn2)
#     cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
#     cnn3 = MaxPooling1D(pool_size=36)(cnn3)
#     # 合并三个模型的输出向量
#     cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
#     flat = Flatten()(cnn)
#     drop = Dropout(0.2)(float)
#     main_output = Dense(3, activation='softmax')(drop)
#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     one_hot_labels = to_categorical(y_train, num_classes=2)  # 将标签转换为one-hot编码
#     model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=20)
#     # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
#     result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
#     result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
#     y_predict = list(map(str, result_labels))
#     print('准确率', accuracy_score(y_test, y_predict))
#     print('平均f1-score:', f1_score(y_test, y_predict, average='weighted'))
if __name__ == '__main__':
    df = pd.read_csv(r'./data/online_shopping_10_cats.csv',encoding='utf-8-sig').astype(str)
    df.dropna(subset=['review'], inplace=True)
    process_text(df,file='./data/cn_stopwords.txt')