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
X = X.tolist()
<<<<<<< Updated upstream
print(X)
=======
>>>>>>> Stashed changes
Y = []
for y in label:
	if y == "regular":
		Y.append([0])
	if y == "global":
		Y.append([1])
	if y == "local":
		Y.append([2])
clss = [0, 1, 2]

#performance evaluation setup
K = 10 #number of folds
seed_value = 1234
clfs = ['NB', 'RF', 'SVM']
# General Results dictionaries
clf_AUROC = dict()
clf_AUPR = dict()
#clf_MAUC = dict()

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
		Y_train = MultiLabelBinarizer().fit_transform(Y_train)
		Y_test_all_normal.extend(Y_test)
		Y_test = MultiLabelBinarizer().fit_transform(Y_test)
		clf.fit(X_train, Y_train)
		y_score = clf.predict_proba(X_test)
		# Why am i concatenating all k-fold data?? https://stackoverflow.com/questions/26587759/plotting-precision-recall-curve-when-using-cross-validation-in-scikit-learn
		if foldCounter == 1:
			Y_test_all = Y_test #initialize
			Y_score_all = y_score #initialize
		else:			
			Y_test_all = np.concatenate((Y_test_all, Y_test)) # aggregate this fold
			Y_score_all = np.concatenate((Y_score_all, y_score)) # aggregate this fold
		foldCounter = foldCounter + 1
	# get MAUC
	#clf_MAUC[clf_name] = MAUC(zip(Y_test_all_normal, Y_score_all), len(clss))
	# multi-class curve plot:
	# pr: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html 
	# ROC: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
	precision = dict()
	recall = dict() #PR
	average_precision = dict() #PR
	fpr = dict() #ROC
	tpr = dict() #ROC
	roc_auc = dict()
	# For each class
	for i in range(len(clss)):
		precision[i], recall[i], _ = precision_recall_curve(Y_test_all[:, i], Y_score_all[:, i])
		average_precision[i] =  average_precision_score(Y_test_all[:, i], Y_score_all[:, i])
		# Compute ROC curve and ROC area for each class
		fpr[i], tpr[i], _ = roc_curve(Y_test_all[:, i], Y_score_all[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	# A "micro-average": quantifying score on all classes jointly
	fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_all.ravel(), Y_score_all.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test_all.ravel(),Y_score_all.ravel())
	average_precision["micro"] = average_precision_score(Y_test_all, Y_score_all, average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

	# setup ROC plot
	# Compute macro-average ROC curve and ROC area
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(clss))]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(len(clss)):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= len(clss)

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	clssname = ['Regular', 'Global Anomaly', 'Local Anomaly']
	# Plot all ROC curves
	lw = 2
	plt.figure(figsize=(7, 8))
	'''
	#plot micro
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)
	#plot macro
	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='teal', linestyle=':', linewidth=4)
	'''
	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
	for i, color in zip(range(len(clss)), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of {0} class (area = {1:0.2f})'
	             ''.format(clssname[i], roc_auc[i]))
	    clf_AUROC[clssname[i] + clf_name] = roc_auc[i]

	fig = plt.gcf()
	fig.subplots_adjust(bottom=0.25)
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve per class (' + clf_name + ')')
	plt.legend(loc="lower right")
	fig.savefig("results/" + clf_name + "/basis_roc.png")

	# setup PR plot 
	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
	plt.figure(figsize=(7, 8))
	#f_scores = np.linspace(0.2, 0.8, num=4)
	lines = []
	labels = []
	'''
	#plot iso f_Scores
	for f_score in f_scores:
	    x = np.linspace(0.01, 1)
	    y = f_score * x / (2 * x - f_score)
	    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
	    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
	lines.append(l)
	labels.append('iso-f1 curves')
	l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
	lines.append(l)
	labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))
	'''
	#pr curves
	for i, color in zip(range(len(clss)), colors):
	    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
	    lines.append(l)
	    labels.append('PR curve of {0} class (area = {1:0.2f})'.format(clssname[i], average_precision[i]))
	    clf_AUPR[clssname[i] + clf_name] = average_precision[i]

	fig = plt.gcf()
	fig.subplots_adjust(bottom=0.25)
	foldCounter	= foldCounter + 1
	#after ploting all k fold curves, plot mean curve and compare
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve per class (' + clf_name + ')')
	plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
	fig.savefig("results/" + clf_name + "/basis_pr.png")



# consolidate AUROC and AUPR data
dataAUROC = [] # convert to dataframe 
dataAUPR = [] # convert to dataframe 

for clf in clfs:
	clf_result_auroc = [clf] # append classifier name
	clf_result_aupr = [clf] # append classifier name
	for group in clssname:
		clf_result_auroc.append(clf_AUROC[group + clf])
		clf_result_aupr.append(clf_AUPR[group + clf])
	#clf_result_auroc.append(clf_MAUC[clf])
	dataAUROC.append(clf_result_auroc)
	dataAUPR.append(clf_result_aupr)

# Create the pandas DataFrame 
#dfroc = pd.DataFrame(dataAUROC, columns = ['Classifier', 'Regular', 'Global', 'Local', 'MAUC']) 
dfroc = pd.DataFrame(dataAUROC, columns = ['Classifier', 'Regular', 'Global', 'Local']) 
dfroc = pd.melt(dfroc, id_vars="Classifier", var_name="group", value_name="AUROC")
dfpr = pd.DataFrame(dataAUPR, columns = ['Classifier', 'Regular', 'Global', 'Local']) 
dfpr = pd.melt(dfpr, id_vars="Classifier", var_name="group", value_name="AUPR")

# print dataframe. 
print(dfroc) 
print(dfpr) 
colors = ['navy', 'turquoise', 'darkorange']
sns.set_palette(colors)
# save auroc barplot. 
fig = sns.factorplot(x='Classifier', y='AUROC', hue='group', data=dfroc, kind='bar')
plt.title('Model performance per class (AUROC)')
fig.savefig("results/consolidated/basis_auroc.png")
# save aupr barplot. 
fig = sns.factorplot(x='Classifier', y='AUPR', hue='group', data=dfpr, kind='bar')
plt.title('Model performance per class (AUPR)')
fig.savefig("results/consolidated/basis_aupr.png")

#interpreting AUROC AND AUPR:
# https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/
# http://www.chioka.in/differences-between-roc-auc-and-pr-auc/
