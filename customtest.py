from modelfinal import *
from keras.utils.np_utils import to_categorical
from nltk.tokenize import word_tokenize
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from constants import *

def cust_data():
	imgin = raw_input("Image to test: ")
	img = np.array(image.load_img(imgin, target_size=(224,224)))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	img = preprocess_input(x)
	question_in = raw_input("Question:")
	ques = [x.lower() for x in word_tokenize(question_in)]
	que = [0 for x in range(0,26)]
	que_len = []
	que_len.append(len(ques))
	loc = 0
	data = json.load(open(data_prepo_meta))
	idx2word = data['ix_to_word']
	for j in ques:
		for k in idx2word.keys():
			if idx2word[k] == j:
				que[loc] = int(k)
				loc+=1
				break
	que = np.array(que)
	que_len = np.array(que_len)
	que_check = move_right(que, que_len)
	que_check = np.reshape(que_check,(1,26))
	#print("que_check: {}".format(que_check))
	meta_data = json.load(open(data_prepo_meta, 'r'))
	meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
	num_words = len(meta_data['ix_to_word'])
	num_classes = len(meta_data['ix_to_ans'])
	vqa = model(num_words, 300, num_classes)
	vqa.load_weights('weights-22.hdf5')
	value = vqa.predict([img, que_check])
	#print(value)
	th = max(value[0])
	#sprint(th)
	ans = []
	data = json.load(open(data_prepo_meta))
	idx2ans = data['ix_to_ans']
	#print(idx2ans)
	#print(value)
	for i in range(0,1000):
		if value[0][i] == th:
#			print("here?", i)
			cnt = 0
			for k in idx2ans.keys():
				if cnt == i:
					ans = idx2ans[k]
				cnt+=1
	img=mpimg.imread(imgin)
	imgplot = plt.imshow(img)
	print("Question: {}".format(question_in))
	print("Answer: {}".format(ans))
	plt.show()
if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	warnings.filterwarnings("ignore")
	cust_data()
