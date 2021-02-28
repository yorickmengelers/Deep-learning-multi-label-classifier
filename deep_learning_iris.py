# Multi label classification deep learning model voor RNAseq cancer data voor Machine learning project
# Volgens https://machinelearningmastery.com/multi-label-classification-with-deep-learning/

# import packages
import pandas as pandas
import numpy as np
from numpy import mean
from numpy import std
np.random.seed(123)  # for reproducibility
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# load data
data = pandas.read_csv("cancer_rnaseq_data_without_first_column.csv") # default header = True
labels = pandas.read_csv("one_hot-labels_without_first_2_columns.csv") # default header = True

# Remove first column of data and first 2 columns of labels
# Ik heb dat even handmatig gedaan, omdat ik het idee had dat er hier een fout zat

# Make data compatible for converting to tensors
data_as_array = np.asarray(data).astype('float32')
labels_as_array = np.asarray(labels).astype('float32')

#print(data_as_array)
#print(labels_as_array)

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	# 20 in the line below means that the hidden layer has 20 nodes
	# Maybe we should try some more values (trial and error)
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# Evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results

n_inputs = 20531
n_outputs = 5
model = get_model(n_inputs, n_outputs)
model.fit(data_as_array,labels_as_array, verbose = 1, epochs = 100) # epochs = 100 ipv 1000

# predict de eerste sample
#row = data_as_array.iloc[0:1,:]
#yhat = model.predict(row)
#print('Predicted: %s' % yhat[0])

# evaluate model
results = evaluate_model(data_as_array, labels_as_array)

# summarize performance
print("Accuracy: %.3f (%.3f)" % (mean(results), std(results)))
