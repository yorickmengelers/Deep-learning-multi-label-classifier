from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
from numpy import array
import numpy as np
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.svm import SVR



def load_data():

    data = pd.read_csv('data_RNAseq.csv', index_col=0)
    # data with on-hot labeling werkt niet
    #labels = pd.read_csv('one_hot-labels2.csv', sep=';', index_col=0)

    #return np.asarray(data).astype("float32"), np.array(labels).astype(int)

    # zonder on-hot-labels
    labels = pd.read_csv('labels_RNAseq.csv', index_col=0)
    return data, labels

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

def apply_pca(data, nr_of_pc):
    if nr_of_pc == 0:
        pca = PCA(.95) # minimal number of components to explain 95% of the variance
    else: pca = PCA(nr_of_pc)

    pca.fit(data)
    nr_of_pc = pca.n_components_
    print("number of principal components: %d" %nr_of_pc)
    result = pca.transform(data)
    # convert to dataframe
    df_result = pd.DataFrame(result)
    return df_result

def logistic_regression(data, labels):
    lr = LogisticRegression(multi_class='ovr')
    # train logistic regression model
    lr.fit(data, labels)
    # evaluate accuracy
    accuracy = lr.score(data, labels)
    return accuracy

def one_hot_labeling(Y):
    #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    y = Y['Class'].unique()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = y.reshape(len(y), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

def convert_to_numeric_labels(labels):
    labels['Class'].unique()
    labels['Class'] = labels['Class'].astype('category').cat.codes


def RFE_SVM(data, labels, n):
    # returns a list of the n most important features
    selected_features = []
    model = SVR(kernel='linear')
    rfe = RFE(model, n_features_to_select=n, step=100, verbose=0)
    rfe.fit(data, labels)
    for i in range(data.shape[1]):
        print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
        if rfe.ranking_[i] == True:
            selected_features.append(data.columns.values[i])
    return selected_features


data, labels = load_data()
#labels = one_hot_labeling(Y)
convert_to_numeric_labels(labels)

# nr of principal components if 0 -. nr components that explains
# 95% of data
nr_of_pc = 10
pca_results = apply_pca(data, nr_of_pc)
nr_of_features_to_select_RFE = 10
RFE_selected_genes = RFE_SVM(data, labels['Class'], nr_of_features_to_select_RFE)

pca_lr_score = logistic_regression(pca_results, labels['Class'])
RFE_score = logistic_regression(data[RFE_selected_genes], labels["Class"])

print("accuracy after pca: %d" %pca_lr_score)
print("accuracy after RFE: %d" %RFE_score)


#labels = np.argmax(labels, axis=1)