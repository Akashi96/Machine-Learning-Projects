'''Submitted by:	
				Akash Agarwal
				OSU ID NUMBER: 933-471-097
				Vishnupriya Nochikaduthekkedath Reghunathan
				OSU ID NUMBER: 933-620-571
				Anand P Koshy
				OSU ID NUMBER: 933-617-060'''


import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import time

list_of_Col_norm = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement',
						'long', 'lat', 'sqft_lot15', 'sqft_living15', 'yr_built', 'yr_renovated', 'grade', 'waterfront',
						'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']

def costFunction(loss, lambd, X, Y, theta, m):
	theta_sq = theta ** 2
	theta_sq[0][0] = theta[0][0]
	return (np.sum(loss ** 2) + lambd * np.sum(theta_sq))


# print(cost)

def gradientDescent(X, Y, theta):
	threshold = 0.5
	alpha = 0.00001  # LEARNING RATE
	lambd = 10  # REGULARIZATION PARAMETER
	m = 1
	itr = 0
	itr_list = []
	cost_list = []
	theta_list = []
	start = time.time()
	while (True):
		hypothesis = np.dot(X, theta)
		loss = hypothesis - Y
		delta_Cost = (1.0 / m) * np.dot(X.transpose(), loss)

		first_theta_element = theta[0][0]
		regularized_term = (lambd / m) * theta
		regularized_term[0][0] = first_theta_element

		theta = theta - alpha * (delta_Cost + regularized_term)
		theta_list.append(theta)
		cost = costFunction(loss, lambd, X, Y, theta, m)
		norm_delta = np.linalg.norm(delta_Cost + regularized_term)
		print("Iteration %d | Cost: %f | Norm: %f" % (itr, cost, norm_delta))
		itr_list.append(itr)
		cost_list.append(cost)
		itr = itr + 1
		if norm_delta <= threshold:
			break
	stop = time.time()
	duration = stop - start
	print('Duration: %f' % (duration))

	return theta, norm_delta, cost, itr_list, cost_list, theta_list


def preprocessing(trainData):
	'''Elimination of ID Column'''
	trainData.drop(['id'], axis=1, inplace=True)
	''' Date Split Function '''

	date = pd.to_datetime(trainData['date'])
	trainData.insert(1, 'Day', date.dt.strftime('%d').astype(float))
	trainData.insert(2, 'Month', date.dt.strftime('%m').astype(float))
	trainData.insert(3, 'Year', date.dt.strftime('%y').astype(float))
	trainData.drop('date', axis=1, inplace=True)

	'''Adding Price column in a different DataFrame'''
	price = pd.DataFrame()
	price = pd.concat([price, trainData['price']], axis=1)

	'''Deleting 'price' column from trainData'''
	trainData.drop(['price'], axis=1, inplace=True)

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NUMERICAL DATA PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	numericalData = pd.DataFrame()
	numericalData = pd.concat([numericalData, trainData[
		['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'long', 'lat',
		 'sqft_lot15', 'sqft_living15', 'yr_built', 'yr_renovated']]], axis=1)

	df1 = pd.DataFrame()

	''' MEAN of Numerical Values '''
	df1["MEAN"] = numericalData.mean()
	''' Standard Deviation of Numerical Values '''
	df1["STD"] = numericalData.std()
	''' Range of Numerical Values '''
	df1["RANGE"] = numericalData.max() - numericalData.min()

	''' Open a CSV File in Append Mode'''
	file = open("DataStats.csv", "a")

	''' Write Stats of Numerical data to CSV file '''
	df1.to_csv(file, sep=',')

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CATEGORICAL DATA PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	categoricalData = pd.DataFrame()
	categoricalData = pd.concat([categoricalData, trainData[
		['grade', 'waterfront', 'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']]], axis=1)

	list_of_Col = ['grade', 'waterfront', 'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']
	categoricalData[list_of_Col].apply(
		lambda column: ((column.value_counts() / 10000) * 100).to_frame().to_csv(file, sep=','))

	list_of_Col_norm = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement',
						'long', 'lat', 'sqft_lot15', 'sqft_living15', 'yr_built', 'yr_renovated', 'grade', 'waterfront',
						'condition', 'zipcode', 'view', 'Month', 'Day', 'Year']
	return trainData, price.values, list_of_Col_norm


def normalize(trainData, list_of_Col_norm):
	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA NORMALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	trainData[list_of_Col_norm] = trainData[list_of_Col_norm].apply(
		lambda col: (col - col.min()) / (col.max() - col.min()))

	return trainData.values


def fileRead(filename):
	File = pd.read_csv(filename)
	return pd.DataFrame(File)


def preProcessData(data):
	date = pd.to_datetime(data['date'])
	data.insert(1, 'Day', date.dt.strftime('%d').astype(float))
	data.insert(2, 'Month', date.dt.strftime('%m').astype(float))
	data.insert(3, 'Year', date.dt.strftime('%y').astype(float))
	data.drop('date', axis=1, inplace=True)
	data.drop(['id'], axis=1, inplace=True)
	X = normalize(data, list_of_Col_norm)
	return X

def generateModel():
	df = fileRead('PA1_train.csv')
	X, Y, colName = preprocessing(df)
	X = normalize(X, colName)
	theta = np.empty(shape=[22, 1])
	theta.fill(0)
	theta, norm_delta, cost, itr_list_train, cost_list_train, train_theta_list = gradientDescent(X, Y, theta)
	return theta


def startFinalTest():
	df_test = fileRead('PA1_test.csv')
	X = preProcessData(df_test)
	theta = generateModel()
	hypothesis = np.dot(X, theta)
	np.savetxt('output.txt', hypothesis, delimiter='\n')

if __name__ == '__main__':
	# df = fileRead('PA1_train.csv')
	# X, Y, colName = preprocessing(df)
	# # print(X)
	# # print(Y)
	# X = normalize(X, colName)
	# # print(X)
	# theta = np.empty(shape=[22, 1])
	# theta.fill(0)
	# theta, norm_delta, cost, itr_list_train, cost_list_train, train_theta_list = gradientDescent(X, Y, theta)
	# print(theta)

	startFinalTest()


	# df = fileRead('PA1_dev.csv')
	# X, Y, colName = preprocessing(df)
	# # X = normalize(X, colName)
	# SSE_valid, last_SSE_valid = validFunc(train_theta_list, X, Y)
	# print(last_SSE_valid)
	# print(train_theta_list)
	# plt.scatter(itr_list_train, cost_list_train, color='blue', s=100)
	# plt.title("Normalized | Learning Rate: 1 | Regularization Parameter: 10")
	# plt.xlabel("Number of Iterations")
	# plt.ylabel("SSE")
	# plt.scatter(itr_list_train, SSE_valid, color='red', s=100)
	# plt.show()