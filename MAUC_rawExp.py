#experiments with original features

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from itertools import cycle
from scipy import interp
#from MAUC import MAUC

#importing python utility libraries
import os, sys, random, io, urllib
from datetime import datetime
import time #elapsed time

# importing data science libraries
import pandas as pd
import random as rd
import numpy as np

# importing python plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from IPython.display import Image, display


# load the synthetic ERP dataset
#ori_dataset = pd.read_csv('./data/fraud_dataset_v2.csv')

# load the synthetic ERP dataset
url = 'https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv'
ori_dataset = pd.read_csv(url)

# remove the "ground-truth" label information for the following steps of the lab
label = ori_dataset.pop('label')

#### REMOVE THIS BLOCK = 1 ################################################################33
'''
# prepare to plot posting key and general ledger account side by side
fig, ax = plt.subplots(1,2)
fig.set_figwidth(12)

# plot the distribution of the posting key attribute
g = sns.countplot(x=ori_dataset.loc[label=='regular', 'BSCHL'], ax=ax[0])
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_title('Distribution of BSCHL attribute values')

# plot the distribution of the general ledger account attribute
g = sns.countplot(x=ori_dataset.loc[label=='regular', 'HKONT'], ax=ax[1])
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_title('Distribution of HKONT attribute values');
#plt.show() # Plots created using seaborn need to be displayed like ordinary matplotlib plots. This can be done using the plt.show() command
fig.savefig("output.png")
'''
#### REMOVE THIS BLOCK = 1 ################################################################33

## PRE-PROCESSING of CATEGORICAL TRANSACTION ATTRIBUTES:
# select categorical attributes to be "one-hot" encoded
categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
# encode categorical attributes into a binary one-hot encoded representation 
ori_dataset_categ_transformed = pd.get_dummies(ori_dataset[categorical_attr_names])

## PRE-PROCESSING of NUMERICAL TRANSACTION ATTRIBUTES:  In order to faster approach a potential global minimum it is good practice to scale and normalize numerical input values prior to network training
# select "DMBTR" vs. "WRBTR" attribute
numeric_attr_names = ['DMBTR', 'WRBTR']
# add a small epsilon to eliminate zero values from data for log scaling
numeric_attr = ori_dataset[numeric_attr_names] + 1e-4
numeric_attr = numeric_attr.apply(np.log)
# normalize all numeric attributes to the range [0,1]
ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

## merge cat and num atts
# merge categorical and numeric subsets
ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)
#initialize X, Y
X = ori_subset_transformed.to_numpy() #convert df to numpy array
Y = []
Y_normal = []
for y in label:
	if y == "regular":
		Y.append([0])
		Y_normal.append(0)
	if y == "global":
		Y.append([1])
		Y_normal.append(1)
	if y == "local":
		Y.append([2])
		Y_normal.append(2)
clss = [0, 1, 2]

all_tau = [5, 10, 20] 
K = 10 #number of folds
seed_value = 1234
clfs = ['NB', 'RF', 'SVM']
# General Results dictionaries
clf_AUROC = dict()
clf_AUPR = dict()
clf_MAUC = dict()   
#maybe you can also use outlier detection:https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py

for clf_name in  clfs:
        '''
        # Apply Stratified K-Folds cross-validator
        Provides train/test indices to split data in train/test sets.
        This cross-validation object is a variation of KFold that returns stratified folds.
        The folds are made by preserving the percentage of samples for each class.
        ''' 
        skf = StratifiedKFold(n_splits=K, random_state = seed_value, shuffle = True)
        #kf =  KFold(n_splits=5, random_state = seed_value, shuffle = True)
        skf_idxs = skf.split(X, Y)

        # Set classifiers

        '''
        SVM
        linear kernel:
        One Vs One dec function shape: used for multi-class predictions
        probability true:
        '''
        # We use OneVsRestClassifier for multi-label prediction
        # predict_proba vs decision func: https://stackoverflow.com/questions/36543137/whats-the-difference-between-predict-proba-and-decision-function-in-scikit-lear
        # predict_proba to plot precision-recall curves: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
        clf = []
        if(clf_name == "SVM"):
            print("Classifier: ", clf_name)
            clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', random_state = seed_value, probability = True))
        if(clf_name == "RF"):
            print("Classifier: ", clf_name)
            clf = OneVsRestClassifier(RandomForestClassifier(n_estimators = 10, random_state = seed_value))
        if(clf_name == "NB"):
            print("Classifier: ", clf_name)
            clf = OneVsRestClassifier(GaussianNB())

        foldCounter = 1
        nPointsCurve = 0
        Y_test_all = [] # concatenated y_test of all folds
        Y_test_all_normal = []
        Y_score_all = [] # concatenated y_scores of all folds
        for train_index, test_index in skf_idxs:
            print("fold: ", foldCounter)
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = [X[i] for i in train_index], [X[j] for j in test_index]
            Y_train, Y_test = [Y[i] for i in train_index], [Y[j] for j in test_index]
            Y_test_normal = [Y_normal[j] for j in test_index]
            Y_train = MultiLabelBinarizer().fit_transform(Y_train)
            Y_test_all_normal.extend(Y_test_normal)
            Y_test = MultiLabelBinarizer().fit_transform(Y_test)
            clf.fit(X_train, Y_train)
            print('fitted')
            y_score = clf.predict_proba(X_test)
            # Why am i concatenating all k-fold data?? https://stackoverflow.com/questions/26587759/plotting-precision-recall-curve-when-using-cross-validation-in-scikit-learn
            if foldCounter == 1:
                Y_test_all = Y_test #initialize
                Y_score_all = y_score #initialize
            else:           
                Y_test_all = np.concatenate((Y_test_all, Y_test)) # aggregate this fold
                Y_score_all = np.concatenate((Y_score_all, y_score)) # aggregate this fold
            foldCounter = foldCounter + 1
            print(foldCounter)
        # get MAUC
        #YConverted = []
        #YConverted.extend(y for y in Y_test_all_normal)
        mapped = list(zip(Y_test_all_normal, Y_score_all))
        clf_MAUC[clf_name] = MAUC(mapped, len(clss))



# consolidate results data
dataMAUC = [] # convert to dataframe 
for clf in clfs:
    clf_result_mauc = [clf] # append classifier name
    dataMAUC.append(clf_result_mauc)

# Create the pandas DataFrame 
dmauc = pd.DataFrame(dataMAUC, columns = ['Classifier']) 
dmauc = pd.melt(dmauc, id_vars="Classifier", value_name="MAUC")
print(dmauc)

consolidated_results = open("results/consolidated/originalAtt_MAUC.txt","w")
for L in dataMAUC:
    consolidated_results.writelines(str(L) + "\n")


#interpreting AUROC AND AUPR:
# https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/
# http://www.chioka.in/differences-between-roc-auc-and-pr-auc/