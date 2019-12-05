#experiments with autoencoder generated features

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interp
import pandas as pd 
from imblearn.metrics import geometric_mean_score as gmean
from MAUC import MAUC

#all_tau = [5, 10, 20] 
all_tau = [5, 10]
K = 5 #number of folds
seed_value = 1234
#clfs = ['NB', 'RF', 'SVM']
clfs = ['NB', 'RF']

# General Results dictionaries
#clf_MAUC = dict()
clf_gmean = dict()

#maybe you can also use outlier detection:https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py

for clf_name in  clfs:

	for tau in all_tau:
		datasetFile = open("latent_data_sets/tau" + str(tau) +"_basisLS.txt","r")
		#initialize X, Y
		X = []
		Y = []
		#read data set
		for L in datasetFile:
			XY = L.split('\t')
			X.append([float(z) for z in XY[0:len(XY)-1]]) # get feature values for object L
			y = XY[len(XY)-1].split('\n')
			Y.append(int(y[0])) # get object L label
		clss = [0, 1, 2]
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
			clf = svm.SVC(kernel = 'linear', random_state = seed_value, probability = True)
		if(clf_name == "RF"):
			print("Classifier: ", clf_name)
			clf = RandomForestClassifier(n_estimators = 10, random_state = seed_value)
		if(clf_name == "NB"):
			print("Classifier: ", clf_name)
			clf = GaussianNB()

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
			Y_test_all_normal.extend(Y_test)
			clf.fit(X_train, Y_train)
			y_score = clf.predict(X_test)
			# Why am i concatenating all k-fold data?? https://stackoverflow.com/questions/26587759/plotting-precision-recall-curve-when-using-cross-validation-in-scikit-learn
			if foldCounter == 1:
				Y_test_all = Y_test #initialize
				Y_score_all = y_score #initialize
			else:			
				Y_test_all = np.concatenate((Y_test_all, Y_test)) # aggregate this fold
				Y_score_all = np.concatenate((Y_score_all, y_score)) # aggregate this fold
			foldCounter = foldCounter + 1
		# get gmean
		clf_gmean[clf_name + str(tau)] = gmean(Y_test_all_normal, Y_score_all)

# consolidate AUROC and AUPR data
for tau in all_tau:

	data_gmean = [] # convert to dataframe 

	for clf in clfs:
		result_gmean_clf = [clf] #append clf name
		result_gmean_clf.append(clf_gmean[clf + str(tau)]) #append gmean result
	data_gmean.append(result_gmean_clf) 

	print(data_gmean)
	# Create the pandas DataFrame 
	dfgmean = pd.DataFrame(data_gmean, columns = ['Classifier', 'Gmean']) 
	dfgmean = pd.melt(dfgmean, id_vars="Classifier", var_name="group", value_name="Gmean")

	colors = ['navy', 'turquoise', 'darkorange']
	sns.set_palette(colors)
	# save gmean for tau. 
	fig = sns.factorplot(x='Classifier', y='Gmean', hue='group', data=dfgmean, kind='bar')
	plt.title('GMEAN for tau = ' + str(tau))
	fig.savefig("results/consolidated/basis_gmean_tau" + str(tau) + ".png")

#interpreting AUROC AND AUPR:
# https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/
# http://www.chioka.in/differences-between-roc-auc-and-pr-auc/