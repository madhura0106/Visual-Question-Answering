from modelfinal import *
from keras.utils.np_utils import to_categorical
from nltk.tokenize import word_tokenize
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from constants import *
ldata = {"cat1.jpg":{"questions":["what is the color of the cat?", "what animals are in the picture?","is the cat on grass?","How many people are watching the animals?","What is the color of eye of cat?","What is the color of mouse","What is the color of bird"], "answers" : ["white", "a cat, a mouse and a bird ", "no","six","brown","white","yellow"]},"img1.jpg":{"questions":["who is playing?", "what is the boy holding in his hands?", "what is the boy playing?", "What is the color of the shirt of boy?", "What is the color of ball?" , "what is the hair color of the boy?", "What is the girl playing"],"answers" : ["a boy", "racket", "tennis", "blue", "green", "blonde", "Invalid question"]}, "img4.jpg":{"questions":["Which animals are there?", "What are the animals playing?", "Where are the animals playing", "what is the color of rabbit?", "what is the color of the tortoise?", "What is the color of the grass?", "What is the color of the ground?", "What are the animals doing?"], "answers":["rabbit and tortoise", "running race", "ground", "white", "green and yellow", "green", "brown", "running"]}, "img5.jpg":{"questions":["what is on the grass", "what type of car is on shown?", "what is the color of the car", "what is the number on the number plate of car?", "what is the color of the grass?", "Is the front light of the car on?", "what is written in the blue color on the number plate?","which state passing the car has?"],"answers":["car", "audi", "black", "MH 02BY5312", "green", "yes", "IND","Maharashtra"]},"img6.jpg":{"questions":["Which animal is standing on the grass?", "What is the color of the cat?","What is the color of the grass?", "How many cats are there on the grass?", "What is the color of eyes of cat?", "What is the color of dog?", "what is dog doing?"],"answers":["cat", "white", "green", "one", "blue", "Invalid question", "Invalid question"]}}
"""
def read_data(data_limit):
    print "Reading Data..."
    img_data = h5py.File(data_img)
    ques_data = h5py.File(data_prepo)
  
    img_data = np.array(img_data['images_train'])
    img_pos_train = ques_data['img_pos_train'][:data_limit]
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))

    #shifting padding to left side
    ques_train = np.array(ques_data['ques_train'])[:data_limit, :]
    ques_length_train = np.array(ques_data['ques_length_train'])[:data_limit]
    ques_train = right_align(ques_train, ques_length_train)

    train_X = [train_img_data, ques_train]
    # NOTE should've consturcted one-hots using exhausitve list of answers, cause some answers may not be in dataset
    # To temporarily rectify this, all those answer indices is set to 1 in validation set
    train_y = to_categorical(ques_data['answers'])[:data_limit, :]

    return train_X, train_y
"""

def read_data():
	print("Reading Data")
	imgs = []
	ques = []
	ans_f = []
	data1 = json.load(open(data_prepo_meta))
	idx2word = data1['ix_to_word']
	print(idx2word)
	for i in ldata.keys():
	#	print("img: {}".format(i))
		img = np.array(image.load_img('/home/madhura/study material/mini project/4th project/VQA-Keras-Visual-Question-Answering-master/'+str(i), target_size=(224,224)))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		img = preprocess_input(x)
		#print(data[i])
		for j in ldata[i]['questions']:
			q = [x.lower() for x in word_tokenize(j)]
			que = [0 for x in range(0,26)]
			que_len = len(q)
			print(j)
			loc = 0
			for k in q:
				for l in idx2word.keys():
					if idx2word[l] == k:
						que[loc] = int(l)
						loc+=1
						break
			print(que)	
			que = np.array(que)
			que_len = np.array(que_len)
			que_check = move_right(que, que_len)
			que_check = np.reshape(que_check,(1,26))
			ques.append(que_check)
			imgs.append(img)
		ar = ldata[i]['answers']
		data = json.load(open(data_prepo_meta))
		idx2ans = data['ix_to_ans']
		#print('\n\n\n\n',idx2ans,'\n\n\n\n')
		for j in ar:
			print(j)
		#	print("answers: {}".format(j))
			train_y = [0 for i in range(0,1000)]
			ans = [x.lower() for x in word_tokenize(j)]
			cnt = 0
			print("ans: {}".format(ans))
			for k in idx2ans.keys():
				if idx2ans[k] == ans[0]:
					save = cnt
					break
				cnt+=1
			print("save: {}".format(save))
			train_y[save] = 1
			train_y = np.array(train_y).reshape((1, 1000))
			ans_f.append(train_y)
#	print(ans_f)
#	ans = to_categorical(ans)
	#print("Img shape: {}".format(img.shape))
	imgs = np.array(imgs)
	imgs = imgs.reshape(imgs.shape[0],imgs.shape[2],imgs.shape[3],imgs.shape[4])
	ques = np.array(ques)
	ques = ques.reshape(ques.shape[0], ques.shape[2])
	train_X = [imgs, ques]
	train_y = np.array(ans_f)
	train_y = train_y.reshape(train_y.shape[0], train_y.shape[2])
	#print("Y shape: {}, Y: {}".format(train_y.shape, train_y))
	return train_X, train_y

def train():
	train_X, train_y = read_data()
	meta_data = json.load(open(data_prepo_meta, 'r'))
	meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
	num_words = len(meta_data['ix_to_word'])
	num_classes = len(meta_data['ix_to_ans'])
	vqa = model(num_words, 300, num_classes)
	file_name = 'weights-{epoch:02d}.hdf5'
	checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
	vqa.fit(train_X, train_y, epochs = 40, callbacks = [checkpoint], verbose = 1)
	return vqa

if __name__ == "__main__":
	train()
	#train_X, train_y = read_data()
	#print(train_X[1].shape, train_X[0].shape)
	#print(train_y.shape)
#	cust_data()
	#train_X, train_y = read_data()
