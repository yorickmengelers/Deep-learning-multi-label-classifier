# Multi label classification deep learning model voor RNAseq cancer data voor Machine learning project
# Volgens https://machinelearningmastery.com/multi-label-classification-with-deep-learning/

# import packages
import pandas as pandas
import numpy as np
from numpy import mean
from numpy import std
from numpy import asarray
np.random.seed(123)  # for reproducibility
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense




# Laad data in
data = pandas.read_csv("C:/Users/Mischa/Downloads/data.csv")
labels = pandas.read_csv("C:/Users/Mischa/Downloads/one_hot-labels.csv")

# Verwijder eerste kolom (sample 0:800, staat toch al in rij naam)
del data['Unnamed: 0']
del labels['Unnamed: 0']
del labels['X']

#print(data)
#print(labels)

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# evaluate a model using repeated k-fold cross-validation
# werkt nog niet, wss kan pandas niet gebruikt worden voor input van RepeatedKfold??
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
model.fit(data,labels, verbose = 1, epochs = 1000)
# predict de eerste sample
row = data.iloc[0:1,:]
yhat = model.predict(row)
print('Predicted: %s' % yhat[0])

# als evaluate model werkt
#results = evaluate_model(data, labels)
##p#rint("Accuracy: %.3f (%.3f)" % (mean(results), std(results)))


