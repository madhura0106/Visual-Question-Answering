from Lstmfinal import *
from Resnet152final import *
import keras
from keras.models import Sequential, Model
from keras.layers import Multiply, Dot, Merge, Input, multiply, dot, merge

ldata = {"cat_1.jpg":{"questions":["what is the color of the cat?", "which animal is on grass?","is the cat on grass?"], "answers" : ["white", "cat", "yes"]},"cat_2.jpg":{"questions":["what is the color of the cat?", "which animal is on the floor?","is the cat on grass?", "is the cat having food?"], "answers" : ["black", "cat", "no", "yes"]},"dog_1.jpg":{"questions":["what is the color of the dog?", "which animal is on grass?","is the dog on grass?", "what is the dog bitting?"], "answers" : ["white", "dog", "yes", "shoe"]},"dog_1.jpg":{"questions":["what is the color of the dog?", "which animal is on floor?","is the dog on grass?", "how many animals are on the floor?"], "answers" : ["black", "dog", "no","2"]}}

def concat_test(input):
	a = input[0]
	b = input[1]
	return multiply([a, b])

def model(num_words, embedding_dim, num_classes):
	img_mod = ResNet152(include_top = False, weights='imagenet', input_shape = (224, 224, 3))
	for layer in img_mod.layers:
		layer.trainable=False
	lstm = Lstm(num_words = num_words, embedding_dim = embedding_dim)
#	inpu1 = Input(shape=img_mod.output_shape)
#	inpu2 = Input(shape=lstm.output_shape)
	vqa = Lambda(concat_test, name='dot')([img_mod.output, lstm.output])
	vqa = Dense(1000, activation='tanh', name = 'fin_fc_1')(vqa)
	vqa = Dense(num_classes,activation='softmax', name = 'fin_fc_2')(vqa)
	model = Model(inputs = [img_mod.input, lstm.input], outputs = vqa)
	model.summary()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#	print(model.output_shape)
	return model

def move_right(array, length):
    vec = np.zeros(np.shape(array))
    n = array.shape[0]
    for i in range(0,length):
    	vec[n - length + i] = array[i]
    return vec
"""
def cust_data():
	img = np.array(image.load_img('cat.jpg', target_size=(224,224)))
	#img_path = 'cat.jpg'
#	x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#x = preprocess_input(x)
	#model = VGG16(include_top = False, weights='imagenet')
	#img = model.predict(x)
	#t = np.sqrt(np.sum(np.multiply(img, img), axis = 1))
	#img = np.divide(img, np.transpose(np.tile(t, (4096, 1)))) 
	ques = [x.lower() for x in word_tokenize("What is the color of cat?")]
	que = [0 for x in range(0,26)]
	que_len = []
	que_len.append(len(ques))
	loc = 0
	data = json.load(open("data/data_prepro.json"))
	idx2word = data['ix_to_word']
	for j in ques:
		for i in idx2word.keys():
			if idx2word[i] == j:
				que[loc] = int(i)
				loc+=1
				break
	que = np.array(que)
	que_len = np.array(que_len)
	que_check = move_right(que, que_len)
	model = get_model(0.0, model_weights_filename)
	print("Img dim: {}, que_check dim: {}".format(img.shape, que_check.shape))
	value = model.predict([img, que_check])
	print(value)
"""
if __name__ == "__main__":
	meta_data = json.load(open('data_prepro.json', 'r'))
	meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
	num_words = len(meta_data['ix_to_word'])
	num_classes = len(meta_data['ix_to_ans'])
	vqa = model(num_words, 300, num_classes)
	
