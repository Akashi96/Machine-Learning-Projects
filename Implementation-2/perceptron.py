'''Submitted by:	
				Akash Agarwal
				OSU ID NUMBER: 933-471-097
				Vishnupriya Nochikaduthekkedath Reghunathan
				OSU ID NUMBER: 933-620-571
				Anand P Koshy
				OSU ID NUMBER: 933-617-060'''

import numpy as np
# import csv
# from matplotlib import pyplot as plt

	
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def fileRead(fileName):
	data = np.genfromtxt(fileName, delimiter = ',')
	return data

def change_N_split(data):
	'''CHANGE 3 TO 1 AND 5 TO -1'''
	Y = data[:, 0]
	Y = np.where(Y > 3.0, -1, 1)

	'''DELETE OUTPUT FROM THE DATA'''
	data = np.delete(data, 0, 1)
	
	return data, Y

def addBias(X, n):
	bias = np.ones((n, 1))
	X = np.append(X, bias, axis = 1)
	return X

def sign(num):
	if(num <= 0):
		num = -1
	else:
		num = 1
	return num



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ONLINE PERCEPTRON~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def onlinePerceptron_training(X, Y, W, iters):
	n = len(X)
	itr = 0
	weightList = []
	lossList = []
	itrList = []
	accurList = []

	while(itr < iters):
		loss = 0
		t = 0

		for row in range(n):
			y_cap = (np.dot(X[row, :], W.transpose()))
			if((Y[row] * y_cap) <= 0):
				W = W + Y[row] * X[row, :]
				# loss = loss + 1

		for row in range(n):
			y_cap = (np.dot(X[row, :], W.transpose()))
			if((Y[row] * y_cap) <= 0):
				loss = loss + 1

		loss = loss  * 1.0 / n
		accuracy = (1 - loss)* 100
		weightList.append(W)
		lossList.append(loss)
		itrList.append(itr + 1)
		accurList.append(accuracy)
		itr = itr + 1
	return weightList, lossList, itrList, accurList



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~AVERAGED PERCEPTRON~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def	averagePerceptron_training(X, Y, W, iters):
	n = len(X)
	c = 0
	avg_W = np.empty(shape = [1, 785])
	avg_W.fill(0)
	sum_c = 0
	itr = 0
	weightList = []
	lossList = []
	itrList = []
	accurList = []
	while(itr < iters):
		loss = 0
		for row in range(n):
			y_cap = np.sign(np.dot(X[row, :], W.transpose()))
			
			if((Y[row] * y_cap) <= 0):
				if sum_c + c > 0:
					avg_W = (sum_c * avg_W + c * W)/ (sum_c + c)
				
				sum_c = sum_c + c
				W = W + Y[row] * X[row, :]
				c = 0
			
			else:
				c = c + 1
		
		if(c > 0):
			avg_W = (sum_c * avg_W + c * W)/ (sum_c + c)
			sum_c += c
			c = 0
		
		loss = 0
		for row in range(n):
			y_cap = np.sign(np.dot(X[row, :],avg_W.transpose()))
			if((Y[row] * y_cap) <= 0):
				loss = loss + 1

		loss = loss * 1.0 / n
		accuracy = (1 - loss)* 100
		accurList.append(accuracy)
		weightList.append(avg_W)
		lossList.append(loss)
		itrList.append(itr + 1)
		itr = itr + 1

	return weightList, lossList, itrList, accurList



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~KERNEL(POLYNOMIAL) PERCEPTRON~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def generate_K_matrix(X, Y, p):
	K = np.matmul(X, Y.transpose())
	K = (1 + K) ** p
	return K

def kerneledPerceptron(X, Y, iters, power):
	n = len(X)
	alpha = np.zeros((n, ))
	K_matrix = generate_K_matrix(X, X, power)
	itr = 0

	lossList = []
	itrList = []
	accurList = []
	alphaList = []

	while(itr < iters):
		loss = 0
		for row in range(n):
			product = np.matmul(K_matrix[row][:], alpha * Y)
			y_cap = np.sign(product)
			if(Y[row] * y_cap <= 0):
				alpha[row] = alpha[row] + 1

		for row in range(n):
			product = np.matmul(K_matrix[row][:], alpha * Y)
			y_cap = np.sign(product)
			if(Y[row] * y_cap <= 0):
				loss= loss + 1

		loss = loss * 1.0/ n
		accuracy = (1 - loss)* 100
		itrList.append(itr + 1)
		lossList.append(loss)
		accurList.append(accuracy)
		# print("Alpha at p = {0}: {1}".format(power, alpha))
		alphaList.append(alpha * 1)
		itr = itr + 1
	return alphaList, lossList, itrList, accurList, K_matrix



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VALIDATION LOSS Function~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def validLoss_func(X, Y, weightList, iters):
	n = len(X)
	itr = 0
	lossList = []
	itrList = []
	accurList = []
	weightList,
	while(itr < iters):
		loss = 0
		for row in range(n):
			y_cap = (np.dot(X[row, :], weightList[itr][:].transpose()))
			if(Y[row] * y_cap) <= 0:
				loss = loss + 1
			
		loss = loss  * 1.0 / n
		accuracy = (1 - loss) * 100
		lossList.append(loss)
		itrList.append(itr + 1)
		accurList.append(accuracy)
		itr = itr + 1
	return lossList, itrList, accurList

def validLoss_func_kernel(X_train, Y_train, X_val, Y_val, weightList, iters, power):
	n = len(X_val)
	itr = 0
	lossList = []
	itrList = []
	accurList = []
	K_matrix = generate_K_matrix(X_val, X_train, power)

	for alpha in weightList:
		loss = 0
		for row in range(0, n):

			product = np.matmul(K_matrix[row][:], alpha * Y_train)
			y_cap = np.sign(np.sum(product))
			if(Y_val[row] * y_cap <= 0):
				loss= loss + 1

		loss = loss  * 1.0 / n
		accuracy = (1 - loss) * 100
		lossList.append(loss)
		itrList.append(itr + 1)
		accurList.append(accuracy)
		itr = itr + 1
		# print("Itr:{0} | Validation Accuracy at p = {1}: {2}".format(itr, power, accuracy))
	return lossList, itrList, accurList



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TESTING, GENERATE PREDICTED Y~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def saveinFile(data, filename):
	np.savetxt(filename, data, delimiter = ",")

def predictY(X, W):
	n = len(X)
	predicted_Y = []
	for row in range(n):
		y_cap = np.sign(np.dot(X[row, :], W.transpose()))
		predicted_Y.append(y_cap)
	return predicted_Y

def predictY_percept(X_test, X_train, Y_train, alpha, power):
	n = len(X_test)
	K_matrix = generate_K_matrix(X_test, X_train, power)
	predicted_Y = []

	for row in range(0, n):
		product = np.matmul(K_matrix[row][:], alpha * Y_train)
		y_cap = np.sign(np.sum(product))
		predicted_Y.append(y_cap)
	return predicted_Y


def online_Perceptron(X, Y, W):
	
	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Call Online Perceptron Training function:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	weightList, lossList, itrList, accurList = onlinePerceptron_training(X, Y, W, 14)		# Call Online Perceptron Training function:

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Training Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	# plt.scatter(itrList, accurList, color = 'blue', s = 15)
	# blue_line, = plt.plot(itrList, accurList, color = 'blue', label = 'Training Accuracy')
	# plt.title("ACCURACY vs NUMBER of ITERATIONS")
	# plt.xlabel("Number of Iterations")
	# plt.ylabel("Accuracy (in %)")

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VALIDATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	validData = fileRead('pa2_valid.csv')	# Read Validation Examples
	X, Y = change_N_split(validData)
	X = addBias(X, 1629)
	lossList, itrList, accurList = validLoss_func(X, Y, weightList, 14)

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Validation Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	# plt.scatter(itrList, accurList, color = 'red', s = 15)
	# red_line, = plt.plot(itrList, accurList, color = 'red', label = 'Validation Accuracy')
	# plt.legend(handles  = [blue_line, red_line])
	# plt.show()

	return weightList


def average_Perceptron(X, Y, W):
	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Call Averaged Perceptron Training function:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	weightList, lossList, itrList, accurList = averagePerceptron_training(X, Y, W, 15)

	print("Training Accuracy List for 15 iterations: {0}".format(accurList))
	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Training Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	# plt.scatter(itrList, accurList, color = 'blue', s = 15)
	# blue_line, = plt.plot(itrList, accurList, color = 'blue', label = 'Training Accuracy')
	# plt.title("ACCURACY vs NUMBER of ITERATIONS")
	# plt.xlabel("Number of Iterations")
	# plt.ylabel("Accuracy (in %)")
	
	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VALIDATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	validData = fileRead('pa2_valid.csv')	# Read Validation Examples
	X, Y = change_N_split(validData)
	X = addBias(X, 1629)
	lossList, itrList, accurList = validLoss_func(X, Y, weightList, 15)

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Validation Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	# plt.scatter(itrList, accurList, color = 'red', s = 15)
	# red_line, = plt.plot(itrList, accurList, color = 'red', label = 'Validation Accuracy')
	# plt.legend(handles  = [blue_line, red_line])
	# plt.show()
	print("Validation Accuracy List for 15 iterations: {0}".format(accurList))
	return weightList


def kerneled_Perceptron(X_train, Y_train):
	powerList = [1, 2, 3, 7, 15]
	max_accur_per_power = []
	list_alpha_per_power = []

	for i in range(len(powerList)):
		p = powerList[i]
		'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Call Kerneled Perceptron Training function:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
		weightList, lossList, itrList, accurList, K_matrix = kerneledPerceptron(X, Y, 25, p)
		
		'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Training Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
		# plt.scatter(itrList, accurList, color = 'blue', s = 15)
		# blue_line, = plt.plot(itrList, accurList, color = 'blue', label = 'Training Accuracy')
		# plt.title("ACCURACY vs NUMBER of ITERATIONS (at P = {0})".format(p))
		# plt.xlabel("Number of Iterations")
		# plt.ylabel("Accuracy (in %)")

		'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VALIDATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
		validData = fileRead('pa2_valid.csv')	# Read Validation Examples
		X_val, Y_val = change_N_split(validData)
		X_val = addBias(X_val, 1629)
		lossList, itrList, accurList = validLoss_func_kernel(X_train, Y_train, X_val, Y_val, weightList, 25, p)
		max_accur_per_power.append(max(accurList))
		# print('Power = {0} | {1}'.format(powerList[i], max(accurList)) )
		list_alpha_per_power.append(weightList)
		
		'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Validation Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
		# plt.scatter(itrList, accurList, color = 'red', s = 15)
		# red_line, = plt.plot(itrList, accurList, color = 'red', label = 'Validation Accuracy')
		# plt.legend(handles  = [blue_line, red_line])
		# plt.show()

	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting Functions for Validation Accuracy vs Powers~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	# plt.scatter(powerList, max_accur_per_power, color = 'green', s = 15)
	# plt.plot(powerList, max_accur_per_power, color = 'green')
	# plt.title("VALIDATION ACCURACY vs DEGREE")
	# plt.xlabel("Degree")
	# plt.ylabel("Validation Accuracy (in %)")
	# plt.show()
	# print(len(list_alpha_per_power[0][7]))
	return list_alpha_per_power


if __name__ == '__main__':
	
	trainData = fileRead('pa2_train.csv')	# Read Training Examples
	X, Y = change_N_split(trainData)
	X = addBias(X, 4888)
	W = np.empty(shape = [1, 785])
	W.fill(0)

	print("Processing Online Percepetron")
	# weightList = online_Perceptron(X, Y, W)

	print("Processing Average Perceptron")
	# weightList_avg = average_Perceptron(X, Y, W)

	print("Processing Kerneled Perceptron")
	alphaList =  kerneled_Perceptron(X, Y)


	'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
	# testData = fileRead('pa2_test_no_label.csv')	# Read test Data
	# X_test = addBias(testData, 1629)
	
	# ONLINE PERCEPTRON TESTING
	# weightList_len = len(weightList)
	# predicted_Y = predictY(X_test, weightList[weightList_len - 1][:])
	# saveinFile(predicted_Y, 'oplabel.csv')

	# KERNELED PERCEPTRON TESTING
	# predicted_Y = predictY_percept(X_test, X, Y, alphaList[2][3], 3) # weightlist[2][3] means that we get the alpha matrix accquired at 4th iteration from the list of alpha made when p = 3
	print(alphaList[2][3])
	# saveinFile(predicted_Y, 'kplabel.csv')
	np.savetxt('abc.csv', alphaList[2][3], delimiter = ",")