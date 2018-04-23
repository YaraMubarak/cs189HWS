import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 
	
		'''
		X = np.array(X)
		Y = np.array(Y)
		means = [] 
		for i in range(self.NUM_CLASSES):
			boolean = i == Y 
			means.append(np.average(X[boolean,:], axis = 0 ))
		
		#use first cov only 
		# covclass = 1  
		# boolean1 = covclass == Y 
		# X1 = X[boolean1,:]
		# cov = np.zeros((2,2)) 
		# for i in range(len(X1)):
		# 	cov = cov + np.dot(X[i,:], X[i,:].T)
		# cov = cov/float(len(X1)) - (means[covclass])**2 

		cov = np.zeros((2,2)) 
		for i in range(len(X)):
			cov = cov + np.dot(X[i,:], X[i,:].T)
		cov = cov/float(len(X)) - (np.average(X, axis = 0))**2 
		
		
		cov = cov + np.eye(2)*self.reg_cov 

		self.cov = cov 
		self.means = means 


	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		x = np.array(x)
		preds = []
		for i in range(self.NUM_CLASSES): 
			vec  = (x- self.means[i])
			preds.append(np.dot(vec.T, np.dot(np.linalg.inv(self.cov), vec)))   
		preds = np.array(preds)

		return np.argmax(preds)

	