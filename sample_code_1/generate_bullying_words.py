import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
def load_data():
        train_data,dev_data,test_data = tr.simplified_data(700, 100, 200)
        vocab = Vocab()
        train_sents = [t.get_words() for t in train_data]
        vocab.construct(list(itertools.chain.from_iterable(train_sents)))
        return train_sents 
def extend_dropout_words(wv,wordVocab,model):
	file=open("non_dropout_words.txt","r")
	non_dropout_words=[]
	for lines in file:
		words=lines.split()
		flag=0
		for word in words:
			if word in wordVocab: 
				non_dropout_words.append(word)
			else:
				continue
	file.close()
	file1=open("non_dropout_words_new.txt","w")
	t=len(non_dropout_words)
	for word in wordVocab:
		word_vec=model[word]
		maxVal=0
		for i in range(0,t):
			word_vec1=model[non_dropout_words[i]]
			temp=cosine_similarity(word_vec.reshape(1,-1),word_vec1.reshape(1,-1))
			if(temp[0][0] > maxVal):
				maxVal=temp[0][0]
		if(maxVal > 0.99):
			file1.write(word.encode('utf-8'))
			file1.write("\n")
	file1.close()

train_sents=load_data()
wordVocab=[]
wv=[]
for sentence in train_sents:
	for words in sentence:
		if words in wordVocab:
			continue
		wordVocab.append(words)
model = gensim.models.Word2Vec(train_sents,size=64,min_count=1)
for word in wordVocab:
	wv.append(model[word])
print wordVocab[0:5]
extend_dropout_words(wv,wordVocab,model)