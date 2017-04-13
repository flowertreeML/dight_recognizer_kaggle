#!/usr/bin/env python
# -*- coding:utf-8 -*-

# first neural network

from classifier import Classifier

import numpy as np

class Neural_Network(Classifier):

	# 3 - layer neural network with ReLU activation function
	def __init__(self, featrue_num, layer_1_size, layer_2_size, classifications_num):
		self.featrue_number = featrue_num
		self.hidden_1_size = layer_1_size
		self.hidden_2_size = layer_2_size
		self.classifications_number = classifications_num

		# set neural network parameter, Normal distribution
		self.parameter = {}
		self.parameter['W1'] = np.random.randn(self.featrue_number, self.hidden_1_size) * np.sqrt(2.0 / self.featrue_number)
		self.parameter['W2'] = np.random.randn(self.hidden_1_size, self.hidden_2_size) * np.sqrt(2.0 / self.hidden_1_size)
		self.parameter['W3'] = np.random.randn(self.hidden_2_size, self.classifications_number) * np.sqrt(2.0 / self.hidden_2_size)
		self.parameter['B1'] = np.ones(self.hidden_1_size) * 0.01
		self.parameter['B2'] = np.ones(self.hidden_2_size) * 0.01
		self.parameter['B3'] = np.ones(self.classifications_number) * 0.01

	def copy(self):
		nn = Neural_Network(self.featrue_number, self.hidden_1_size, self.hidden_2_size, self.classifications_number)
		return nn

	def train(	self, 
			train_set,
			label_set,
			learning_rate,
			learning_rate_decay_number,
			regularization_strengths,
			iters_number,
			batch_number = -1,
			n = 1000,
			verbose = False	):

		label_set = self._normalize_labels(label_set)

		train_number, featrue_number = train_set.shape

		loss_history = []

		# start train
		print "neural network is training : "
		for it in xrange(1, 2 + 1):

			# get batch to SGD, np.random.choice replace = False means can not be repeated
			if batch_number != -1:
				indices = np.random.choice(train_number, size=batch_number, replace=True)
				train_batch = train_set[indices]
				label_batch = label_set[indices]
			else:
				train_batch = train_set
				label_batch = label_set

			loss, grads = self.loss(train_batch, label_batch, regularization_strengths)
			print loss
			#return
			loss_history.append(loss)

			# update W and B
			for parameter_temp in self.parameter.keys():
				self.parameter[parameter_temp] -= learning_rate * grads[parameter_temp]

			# output loss when 1000, 2000, 3000....
			if verbose and it % 1000 == 0:
				print "the " + str(it) + " loss is :" + str(loss)

			# update learning_rate when n
			if it % n == 0:
				learning_rate *= learning_rate_decay_number

		return np.array(loss_history)

	def predict(self, data):
		data_scores = self.predict_scores(data)

		label_predict = np.argmax(data_scores, axis = 1)
		print label_predict[0 : 20]
		print np.vectorize(self.to_label)(label_predict)[0 : 20]
		return np.vectorize(self.to_label)(label_predict)

	def predict_scores(self, data):
		W1, W2, W3, B1, B2, B3 = (	self.parameter['W1'],
						self.parameter['W2'],
						self.parameter['W3'],
						self.parameter['B1'],
						self.parameter['B2'],
						self.parameter['B3'],	)
		scores = np.maximum(0, 
					np.maximum(0, 
							data.dot(W1) + B1).dot(W2) + B2).dot(W3) + B3
		return scores
	'''
	def loss(self, train_batch, label_batch, regularization_strengths = 0):
		train_number, featrue_number = train_batch.shape
		W1, W2, W3, B1, B2, B3 = (	self.parameter['W1'],
						self.parameter['W2'],
						self.parameter['W3'],
						self.parameter['B1'],
						self.parameter['B2'],
						self.parameter['B3'],	)
		
		# forward
		hidden_1_scores = train_batch.dot(W1) + B1
		hidden_1_relu = np.maximum(0, hidden_1_scores)
		#print hidden_1_scores[0]
		#print hidden_1_relu[0]

		hidden_2_scores = hidden_1_relu.dot(W2) + B2
		hidden_2_relu = np.maximum(0, hidden_2_scores)
		#print hidden_2_scores[0]
		#print hidden_2_relu[0]

		output_scores = hidden_2_relu.dot(W3) + B3
		#print output_scores[0]

		softmax_exp = np.exp(output_scores)
		softmax_scores_sum = np.sum(softmax_exp, axis = 1).reshape(-1, 1)
		softmax_scores = softmax_exp / softmax_scores_sum
		correct_prodict = softmax_scores[np.arange(train_number), label_batch]

		# loss only aim to plot loss function, the last step don't know why, but no problem
		loss = np.sum(-np.log(correct_prodict))
		loss /= train_number
		loss += 0.5 * regularization_strengths * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

		# bp
		# softmax-layer loss function
		softmax_scores[np.arange(train_number), label_batch] -= 1
		softmax_scores /= train_number

		dB3 = np.sum(softmax_scores, axis = 0)
		dW3 = (hidden_2_relu.T / train_number).dot(softmax_scores)
		dW3 += regularization_strengths * W3

		d_hidden2_output = softmax_scores.dot(W3.T)
		d_hidden2_scores = (hidden_2_scores > 0).astype(float) * d_hidden2_output

		dB2 = np.sum(d_hidden2_scores, axis = 0)
		dW2 = (hidden_1_relu.T / train_number).dot(d_hidden2_scores)
		dW2 += regularization_strengths * W2

		d_hidden1_output = d_hidden2_scores.dot(W2.T)
		d_hidden1_scores = (hidden_1_scores > 0).astype(float) * d_hidden1_output

		dB1 = np.sum(d_hidden1_scores, axis = 0)
		dW1 = (train_batch.T / train_number).dot(d_hidden1_scores)
		dW1 += regularization_strengths * W1

		grads = {
			'W1' : dW1,
			'W2' : dW2,
			'W3' : dW3,
			'B1' : dB1,
			'B2' : dB2,
			'B3' : dB3
		}
		return loss, grads
	'''
	def loss(self, X, y, reg = 0):
		N, _ = X.shape

		W1, b1, W2, b2, W3, b3 = (self.parameter['W1'],
		self.parameter['B1'],
		self.parameter['W2'],
		self.parameter['B2'],
		self.parameter['W3'],
		self.parameter['B3'])

		# computing score
		        
		h1_scores = X.dot(W1) + b1
		h1_relu   = np.maximum(0, h1_scores)
		h2_scores = h1_relu.dot(W2) + b2
		h2_relu   = np.maximum(0, h2_scores)
		scores = h2_relu.dot(W3) + b3

		unnormalized_probs = np.exp(scores)
		normalizer = np.sum( unnormalized_probs, axis=1 ).reshape(-1, 1)
		probs = unnormalized_probs / normalizer
		correct_label_probs = probs[np.arange(N), y]

		loss = np.sum( -np.log(correct_label_probs) )
		loss /= N
		loss += 0.5 * reg * ( np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) )

		dscores = probs
		dscores[np.arange(N), y] -= 1
		dscores /= N

		db3 = np.sum(dscores, axis=0)
		dW3 = h2_relu.T.dot(dscores)
		dW3 += reg * W3

		dh2_relu = dscores.dot(W3.T)
		dh2_scores = (h2_scores > 0).astype(float) * dh2_relu

		db2 = np.sum(dh2_scores, axis=0)
		dW2 = h1_relu.T.dot(dh2_scores)
		dW2 += reg * W2

		dh1_relu = dh2_scores.dot(W2.T)
		dh1_scores = (h1_scores > 0).astype(float) * dh1_relu

		db1 = np.sum(dh1_scores, axis=0)
		dW1 = X.T.dot(dh1_scores)
		dW1 += reg * W1

		grads = {'W1' : dW1,
			'W2' : dW2,
			'W3' : dW3,
			'B1' : db1,
		 	'B2' : db2,
			'B3' : db3 }

		return loss, grads