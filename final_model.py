# Multi label classification deep learning model voor RNAseq cancer data voor Machine learning project
# Volgens https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

# - hoeveel layers: hoeveelheid RNA expressed is input, daarna wordt dit o.a. (deels) omgezet in eiwitten die vervolgens dingen
# doen voor een cel (hier een groep cellen) wat iets zou kunnen zeggen over welk type kanker. Dus ca. 2 layers (3 in die tutorial, want daar is output ook een dense layer).
#Parameter BRCA COAD KIPAN
#Activation function {Rectifier, Tanh, Maxout}
#Number of hidden layers {2, 3, 4}
#Number of units per layer [10, 200]
#L1 regularization [0.001, 0.1]
#L2 regularization [0.001, 0.1]
#Input dropout ratio [0.001, 0.1]
#Hidden dropout ratios [0.001, 0.1]

# import packages
import pandas as pandas
import numpy as np
import imblearn
from numpy import mean
from numpy import std
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

# define the model
def create_model(hidden_layers=1,activation='relu',neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(80, input_dim=20531, activation=activation))
    for i in range(hidden_layers):
        #Add a hidden layer
        model.add(Dense(neurons, activation=activation))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def apply_pca(data, nr_of_pc):
    if nr_of_pc == 0:
        pca = PCA(.95) # minimal number of components to explain 95% of the variance
    else: pca = PCA(nr_of_pc)

    pca.fit(data)
    nr_of_pc = pca.n_components_
    print("number of principal components: %d" %nr_of_pc)
    result = pca.transform(data)
    return result

def RFE_SVM(data, labels, n):
    # returns a list of the n most important features
    selected_features = []
    model = SVR(kernel='linear')
    rfe = RFE(model, n_features_to_select=n, step=100, verbose=0)
    rfe.fit(data, labels)
    for i in range(data.shape[1]):
        if rfe.ranking_[i] == True:
            selected_features.append(data.columns.values[i])
    return selected_features

# load data
data = pandas.read_csv("data.csv", index_col= 0) # default header = True
labels = pandas.read_csv("one_hot-labels.csv", sep = ";", index_col= 0) # default header = True

#print(data)
#print(labels)
# Make data compatible for converting to tensors
data_as_array = np.asarray(data).astype('float32')
labels_as_array = np.asarray(labels).astype('float32')

#print(data_as_array)
#print(labels_as_array)

# OUD: Maak train en test set
#X_train, X_test, y_train, y_test = train_test_split(data_as_array, labels_as_array, test_size=0.20, random_state=33)

# Split 5 time the data into a test and training set for outer CV
cv_outer = KFold(n_splits=5, shuffle=True)

for train_ix, test_ix in cv_outer.split(data_as_array):

    # split data
    X_train, X_test = data_as_array[train_ix, :], data_as_array[test_ix, :]
    y_train, y_test = labels_as_array[train_ix], labels_as_array[test_ix]

    # Balance data set volgens http://glemaitre.github.io/imbalanced-learn/generated/imblearn.over_sampling.RandomOverSampler.html
    # oversample samples < average
    no_samples = np.count_nonzero(y_train, axis = 0)
    average_samples = int(mean(no_samples))
    weights = []
    for i in range(len(no_samples)):
        if no_samples[i] < average_samples:
            weights.append(average_samples)
        else:
           weights.append(no_samples[i])

    ratio_over = {0:weights[0], 1:weights[1], 2:weights[2], 3:weights[3], 4:weights[4]}
    over = SMOTE(sampling_strategy = ratio_over, random_state = 314)
    X_train,y_train = over.fit_resample(X_train,y_train)
    # undersample samples > average
    ratio_under = {0:average_samples, 1:average_samples, 2:average_samples, 3:average_samples, 4:average_samples}
    under = RandomUnderSampler(sampling_strategy = ratio_under, random_state = 314)
    X_train,y_train = under.fit_resample(X_train,y_train)

    # PCA feature reduction
    X_train = apply_pca(X_train, 0)

    # nr_of_features_to_select_RFE = 5
    # RFE_selected_genes = RFE_SVM(data, labels['Class'], nr_of_features_to_select_RFE)
    # X_train_RFE = X_train[RFE_selected_genes]


    # Configure the cross-validation procedure
    cv_inner = LeaveOneOut()

    #Hyper parameter code
    batch_size = [8,16,32]
    neurons = [30,40,50]
    hidden_layers = [1,2,3]
    epochs = [10,50,100]
    activation = ['relu','tanh','sigmoid','linear']
    param_grid = dict(batch_size=batch_size,neurons=neurons,hidden_layers=hidden_layers,epochs=epochs,activation=activation)
    estimator = KerasClassifier(build_fn=create_model, verbose=0)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-2, cv=cv_inner)
    #Hier gaat het fout, misschien verkeerde x en y data set?
    resultgridsearch = grid.fit(X_train,y_train)
    print("Best grid search score: %f using the following parameters: %s" % (resultgridsearch.best_score_,resultgridsearch.best_params_))