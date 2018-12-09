import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time


def costFunction(loss, lambd, X, Y, theta, m):
	# cost = np.sum(loss ** 2) + np.sum(np.pow(theta[1:][0], 2))
	# print(theta[0][0])
	theta_sq = theta ** 2
	# print(theta)
	# print(theta_sq)
	theta_sq[0][0] = theta[0][0]
	# print(theta_sq)
	# print(theta_sq)
	# print(np.sum(theta_sq[1:]))
	return (np.sum(loss ** 2) + lambd * np.sum(theta_sq)) 
	# print(cost)

def gradientDescent(X, Y, theta):
	threshold = 0.5
	alpha = 0.00000000000001	# LEARNING RATE
	lambd = 10	# REGULARIZATION PARAMETER
	m = 1
	# print(m)
	itr = 0
	itr_list = []
	cost_list = []
	theta_list = []
	start = time.time()
	while(itr != 10000):
		hypothesis = np.dot(X, theta)
		loss = hypothesis - Y
		delta_Cost = (1.0/ m) * np.dot(X.transpose(), loss)

		first_theta_element = theta[0][0]
		regularized_term = (lambd/ m) * theta
		# print(regularized_term)
		regularized_term[0][0] = first_theta_element
		# print(regularized_term)
		theta = theta - alpha * (delta_Cost + regularized_term)
		theta_list.append(theta)
		cost = costFunction(loss, lambd, X, Y, theta, m)
		norm_delta = np.linalg.norm(delta_Cost + regularized_term)
		# cost = np.sum(loss ** 2)/ (2 * len(X)) 
		print("Iteration %d | Cost: %f | Norm: %f" % (itr, cost, norm_delta))
		# print(loss)
		# print(theta)
		itr_list.append(itr)
		cost_list.append(cost)
		itr = itr + 1
		if norm_delta <= threshold:
			break
	stop = time.time()
	duration = stop - start
	print('Duration: %f' % (duration))
	# print(theta)
	
	return theta, norm_delta, cost, itr_list, cost_list, theta_list


def preprocessing(trainData):	
	'''Elimination of ID Column'''
	trainData.drop(['id'], axis = 1, inplace = True)
	# print(trainData)
	# print(trainData.date)

	''' Date Split Function '''
	# trainData['Month'] = [d.split('/')[0]for d in trainData.date]
	# trainData['Date'] = [d.split('/')[1] for d in trainData.date]
	# trainData['Year'] = [d.split('/')[2] for d in trainData.date]
	date = pd.to_datetime(trainData['date'])
	trainData.insert(1, 'Day', date.dt.strftime('%d').astype(float))
	trainData.insert(2, 'Month', date.dt.strftime('%m').astype(float))
	trainData.insert(3, 'Year', date.dt.strftime('%y').astype(float))
	trainData.drop('date', axis = 1, inplace = True)
	# print(trainData)
	
	'''Adding Price column in a different DataFrame'''
	price = pd.DataFrame()
	price = pd.concat([price, trainData['price']], axis = 1)

	'''Deleting 'price' column from trainData'''
	trainData.drop(['price'], axis = 1, inplace = True)
	# print(price)

	# print(trainData.to_string())
	# print(trainData.values)
	# file = open("abc.txt", "w")
	# trainData.values.tofile(file, sep = ',')
	

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NUMERICAL DATA PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	numericalData = pd.DataFrame()
	numericalData = pd.concat([numericalData, trainData[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement','long', 'lat', 'sqft_lot15', 'sqft_living15', 'yr_built', 'yr_renovated']]], axis = 1)
	# print(numericalData.describe())
	# print(numericalData.describe(include='all').to_string())
	df1 = pd.DataFrame()
	
	''' MEAN of Numerical Values '''
	df1["MEAN"] = numericalData.mean()
	''' Standard Deviation of Numerical Values '''
	df1["STD"] = numericalData.std()
	''' Range of Numerical Values '''
	df1["RANGE"] = numericalData.max() - numericalData.min()
	# print(df1)
	
	''' Open a CSV File in Append Mode'''
	file = open("DataStats.csv", "a")
	
	''' Write Stats of Numerical data to CSV file '''
	df1.to_csv(file, sep = ',')

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CATEGORICAL DATA PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	categoricalData = pd.DataFrame()
	categoricalData = pd.concat([categoricalData, trainData[['grade', 'waterfront', 'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']]], axis = 1)
	# print(categoricalData)
	
	list_of_Col = ['grade', 'waterfront', 'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']
	categoricalData[list_of_Col].apply(lambda column: ((column.value_counts()/ 10000) * 100).to_frame().to_csv(file, sep = ','))
	# for column in categoricalData:
	# ''' Write Stats of Categorical to CSV file '''
	# 	((categoricalData[column].value_counts()/ 10000) * 100).to_frame().to_csv(file, sep = ' ')

	

	# t = trainData.corr()
	# #print t
	list_of_Col_norm = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement','long', 'lat', 'sqft_lot15', 'sqft_living15', 'yr_built', 'yr_renovated', 'grade', 'waterfront', 'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']
	return trainData, price.values, list_of_Col_norm

	
def normalize(trainData, list_of_Col_norm):
	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA NORMALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	
	trainData[list_of_Col_norm] = trainData[list_of_Col_norm].apply(lambda col: (col - col.min()) / (col.max() - col.min()))
	# print(trainData)
	# plt.plot(trainData.values, price.values, 'bo')
	# plt.legend()
	# plt.show()
	return trainData.values


def fileRead(filename):
	File = pd.read_csv(filename)
	return pd.DataFrame(File)
	# print(trainData)

def validFunc(theta_list, X, Y):
	# print(theta.shape)
	# print(X.shape)
	# print(Y.shape)
	#prediction = np.dot(X, theta)
	# print(prediction.shape)
	# print(Y.shape)
	#errorMat = Y - prediction
	# print(errorMat)
	# print(theta)
	# theta, norm_delta, cost, itr_list_val, cost_list_val = gradientDescent(X, Y, theta)
	# return itr_list_val, cost_list_val
	# print(theta)
	#SSE_mat_val = errorMat ** 2
	#return np.sum(SSE_mat_val)
	SSE_mat_val = []
	for theta in theta_list:
		prediction = np.dot(X, theta)
		errorMat = Y - prediction
		sq_error = errorMat ** 2
		SSE = np.sum(sq_error)
		SSE_mat_val.append(SSE)
	return SSE_mat_val, SSE

if __name__ == '__main__':
	df = fileRead('PA1_train.csv')
	X, Y, colName = preprocessing(df)
	#print(X)
	#print(Y)
	# X = normalize(X, colName)
	#print(X)
	theta = np.empty(shape = [22, 1])
	theta.fill(0)
	theta, norm_delta, cost, itr_list_train, cost_list_train, train_theta_list = gradientDescent(X, Y, theta)
	print(theta)
	df = fileRead('PA1_dev.csv')
	X, Y, colName = preprocessing(df)
	# X = normalize(X, colName)
	SSE_valid, last_SSE_valid = validFunc(train_theta_list, X, Y)
	print(last_SSE_valid)
	#print(train_theta_list)
	plt.scatter(itr_list_train, cost_list_train, color = 'blue', s = 1)
	plt.title("Normalized | Learning Rate: 10^-14 | Regularization Parameter: 10")
	plt.xlabel("Number of Iterations")
	plt.ylabel("SSE")
	plt.scatter(itr_list_train, SSE_valid, color = 'red', s = 1)
	plt.show()