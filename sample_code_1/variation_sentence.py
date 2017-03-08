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
from random import choice
def variation(sentence):
	file=open("non_dropout_words_new.txt","r")
	non_dropout_words=[]
	for lines in file:
		words=lines.split()
		for word in words:
			non_dropout_words.append(word)
	file.close()
	possible_dropout=[]
	for word in sentence:
		if word in non_dropout_words:
			continue
		possible_dropout.append(word)
	possible_dropout_len=len(possible_dropout)
	sentence_len=len(sentence)
	random_choice=np.random.choice(possible_dropout,size=3,replace=False)
	final_sentence=[]
	X='mukilesh'
	for i in range(0,sentence_len):
		if sentence[i] not in random_choice:
			final_sentence.append(sentence[i])
		else:
			final_sentence.append(X)
	return final_sentence