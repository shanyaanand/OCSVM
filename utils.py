import numpy as np 
import os 
import matplotlib.pyplot as plt 
import pywt

def WPT(X):
	y = []
	for x in X:
		wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
		temp = [node.data[0] for node in wp.get_level(wp.maxlevel, 'natural')]
		y.append(temp)
	return np.array(y)

def load(address):
	file = open(address, "r")
	lines = file.readlines()
	arr = []
	for line in lines:
		sensor = list(map(float, line.split(" ")[: -2]))[5:]
		arr.append(sensor)
	return np.array(arr)

def normalize(X):
	mean_vector = X.mean(axis=0)
	std_vector = X.std(axis=0)
	y = np.zeros(X.shape)
	for i in range(X.shape[0]):
		x = X[i, :]
		y[i, :] = np.divide(x - mean_vector,std_vector )
	return y

def Get_training_data():

	X = []
	txt_files = os.listdir("./train")
	for j in range(len(txt_files)):
		txt = txt_files[j]
		address = "./train/"+txt
		datas = load(address).T
		X.append(WPT(normalize(datas)))
		print(txt)
	return X

def Get_testing_data():

	X = []
	txt_files = os.listdir("./test")
	for j in range(len(txt_files)):
		txt = txt_files[j]
		address = "./test/"+txt
		datas = load(address).T
		X.append(WPT(normalize(datas)))
		print(txt)
	return X

