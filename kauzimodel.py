from Lstm import *
from Resnet152 import *
import keras
from keras.models import Sequential, Model
from keras.layers import Multiply, Dot, Merge, Input, multiply, dot

def concat_test(input):
	a = input[0]
	b = input[1]
	return multiply([a, b])
def model(num_words, embedding_dim):
	lstm = Lstm(num_words = num_words, embedding_dim = embedding_dim)
	img_mod = ResNet152(include_top = False, weights='imagenet')
	inpu1 = Input(shape=img_mod.output_shape)
	inpu2 = Input(shape=lstm.output_shape)
	print("img_mod: {}, lstm: {}".format(img_mod.output_shape, lstm.output_shape))
	vqa = Lambda(concat_test, name='dot')([inpu1, inpu2])
	vqa = Dense(1000,activation='tanh')(vqa)
	vqa = Dense(1000,activation='tanh')(vqa)
	model = Model(inputs = [inpu1, inpu2], outputs = vqa)
	print(model.output_shape)
	return model
meta_data = json.load(open('data_prepro.json', 'r'))
meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
num_words = len(meta_data['ix_to_word'])
vqa = model(num_words, 300)
