import re
import numpy as np
import h5py
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
import numpy as np
from keras.models import model_from_json#load_model
from keras.callbacks import ModelCheckpoint
import os
import argparse
from models import *
from prepare_data import *
from constants import *
from keras.models import Sequential
from keras.layers import Dense, Activation,Merge,RepeatVector,concatenate
from keras.layers import LSTM
from keras import backend as tf
from resnet1 import *
from load_glove_embeddings import load_glove_embeddings


def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,weights=[embedding_matrix], input_length=seq_length, trainable=False))
    left = Sequential()
    left.add(LSTM(1024, input_shape = (30,300),return_sequences=True))
    #right = Sequential()
    model.add(left)
    model.add(LSTM(1024,return_sequences=False))
    #print(tf.concatenate([tf.reshape(left.outputs[0][-1][-1],(1024,)),tf.reshape(model.outputs[0],(1024,))],axis=-1))
    
    return model


def Lstm(word_feature_size = 300,number_of_hidden_units = 1024,max_length = 30):
    m,l = Lstm()
#print(l)
#print(m.outputs)

def vqa(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    lstm = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    resnet = ResNet152()
    fc_model = Sequential()
    print("lstm=",lstm.output_shape)
    print("resnet=",resnet.output_shape)
    fc_model.add(Merge([lstm, resnet], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    return fc_model

    #print("----------------------------------resnet.shape",resnet.output_shape)
    #print("----------------------------------lstm.shape",lstm.output_shape)
    #print("---------------------------",tf.multiply(lstm,resnet))
    

if __name__=="__main__":
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())
    #print("----------numwords:",num_words)
    word2index, embedding_matrix = load_glove_embeddings('data/glove.6B.300d.txt', embedding_dim=300)
    print(embedding_matrix.shape)
    #model = Word2VecModel(embedding_matrix,num_words,embedding_dim,seq_length,0.5)
    vqa_model = vqa(embedding_matrix,num_words,embedding_dim,seq_length,0.5)
    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    data = json.load(open("data/data_prepro.json"))
    idx2word = data['ix_to_word']
    que = [0 for x in range(0,19)]
    ques = [x.lower() for x in word_tokenize("What is the color of cat?")]
    ques_len = []
    ques_len.append(len(ques))
    for j in ques:
	for i in idx2word.keys():
		if idx2word[i] == j:
			que.append(i)
			break
    que = np.array(que)
    que_len = np.array(ques_len)
    que_check = move_right(que, que_len)
    print("----------------------->",que_check.shape)
    vqa_model.predict(x,que_check)
