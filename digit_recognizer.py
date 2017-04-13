#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import itertools

from neural_network import Neural_Network

def dight_recognizer(layer_one_size = 500, layer_two_size = 250):

	# load train_data
	print "loading data : "
	train_filename = '/home/fuyan/kaggle/digit_recognizer/data/train.csv'
	all_data = np.array(pd.read_csv(train_filename))
	# featrue change to float
	featrue_data = all_data[ : , 1 : ].astype(float)
	# label change to int
	label_data = all_data[ : , 0].astype(int)

	# average-removed
	train_average = np.mean(featrue_data, axis = 0)
	featrue_data -= train_average

	# get train set and test set
	print "getting train set and test set : "
	train_data, test_data, train_label, test_label = train_test_split(featrue_data, label_data, test_size = 0.2, random_state = 42)

	# set layer, get classifications
	all_data_num, featrue_num = train_data.shape
	layer_1_size = layer_one_size
	layer_2_size = layer_two_size
	classifications_num = len(np.unique(label_data))
	print "total	" + str(classifications_num) + "	types"

	#set main parameter
	learning_rate = [0.002]	# we can put more parameter to compare
	regularization_strengths = [0.02]	# we can put more parameter to compare
	num_iters = 50000
	batch_size = 100
	learning_rate_decay_num = 0.98

	# init best net
	best_net = None
	best_loss_history = None
	best_accuracy = None

	# start
	print "training start : "
	for rate_temp, reg_temp in itertools.product(learning_rate, regularization_strengths):
		print "learning_rate : " + str(rate_temp) + "\t" + "regularization_strengths : " + str(reg_temp)
		net_temp = Neural_Network(featrue_num, layer_1_size, layer_2_size, classifications_num)
	
		loss_history_temp = net_temp.train(	train_set = train_data, 
							label_set = label_data,
							learning_rate = rate_temp, 
							regularization_strengths = reg_temp, 
							iters_number = num_iters, 
							batch_number = batch_size, 
							learning_rate_decay_number = learning_rate_decay_num,
							n = 1000,
							verbose=True	)
		#return 
		'''
		g, p = net_temp.train(	train_set = train_data, 
							label_set = label_data,
							learning_rate = rate_temp, 
							regularization_strengths = reg_temp, 
							iters_number = num_iters, 
							batch_number = batch_size, 
							learning_rate_decay_number = learning_rate_decay_num,
							n = 1000,
							verbose=True	)
		return g, p
		'''
		# output accuracy
		train_data_accuracy = np.mean(net_temp.predict(train_data) == train_label)
		test_data_accuracy = np.mean(net_temp.predict(test_data) == test_label)
		print "\ttrain set accuracy : " + str(train_data_accuracy)
		print "\ttest set accuracy : " + str(test_data_accuracy)

		# update the best net
		if test_data_accuracy > best_accuracy:
			best_accuracy = test_data_accuracy
			best_net = net_temp
			best_loss_history = loss_history_temp

	# output the best net
	print "the best neural network accuracy is : "
	print "\ttrain set accuracy : " + str(np.mean(best_net.predict(train_data) == train_label))
	print "\ttest set accuracy : " + str(np.mean(best_net.predict(test_data) == test_label))

	# plot loss history
	print "plot loss history : "
	plt.plot(best_loss_history)
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.title('loss history')
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

	# # recognizer kaggle test set and write to file
	# print "loading kaggle test set : "
	# kaggle_test_filename = '/home/fuyan/kaggle/dight_recognizer/data/test.csv'
	# kaggle_test_data = pd.read_csv(kaggle_test_filename)
	

if __name__ == '__main__':
	dight_recognizer(500, 250)