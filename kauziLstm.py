from keras.models import Sequential,Model
from keras.layers import Dense,Dot,Lambda,Input, Activation,Merge,RepeatVector,Concatenate,Dropout,Reshape, Embedding
from keras.layers import LSTM
from keras import backend as tf
import os, h5py, json
import numpy as np
def Lstm(num_words , embedding_dim, word_feature_size = 300,number_of_hidden_units = 1024,que_length = 26):
    inputs = Input(shape = (26,))
    if os.path.exists('data/ckpts/embeddings_300.h5'):
        with h5py.File('data/ckpts/embeddings_300.h5') as f:
            embedding_matrix = np.array(f['embedding_matrix'])
    inpu = Embedding(num_words, embedding_dim, 
        weights=[embedding_matrix], input_length=que_length, trainable=False)(inputs)
#    print(inputs)
    left = LSTM(1024, input_shape = (que_length,word_feature_size),return_sequences=True)(inpu)
    p = Lambda(lambda x:x[:,-1,:])(left)
    left = (Dropout(rate=0.4))(left)
#    print(p)
    left = LSTM(1024,return_sequences=False)(left)
    left = Lambda(lambda x:x[:,:])(left)
    left = Dropout(rate = 0.4)(left)
#    print(left)
    con = Concatenate(axis = 1)([p,left])
    model = Model(input= inputs, output=con)
#    print(model.output_shape)
    return model
"""
meta_data = json.load(open('data_prepro.json', 'r'))
meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
num_words = len(meta_data['ix_to_word'])
m = Lstm(num_words = num_words, embedding_dim=300)"""
