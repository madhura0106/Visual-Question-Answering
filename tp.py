import h5py
import json
import numpy as np
from nltk.tokenize import word_tokenize
def move_right(array, length):
    vec = np.zeros(np.shape(array))
    n = array.shape[0]
    for i in range(0,length):
    	vec[n - length + i] = array[i]
    return vec

data = json.load(open("data/data_prepro.json"))
idx2word = data['ix_to_word']
que = [0 for x in range(0,26)]
ques = [x.lower() for x in word_tokenize("What is the color of cat?")]
que_len = []
que_len.append(len(ques))
loc = 0
for j in ques:
	for i in idx2word.keys():
		if idx2word[i] == j:
			que[loc] = int(i)
			loc+=1
			break
que = np.array(list(que))
print(que)
que_len = np.array(que_len)
print(que_len)
print(move_right(que, que_len))
