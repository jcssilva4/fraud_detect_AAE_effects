#experiments with autoencoder generated features

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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interp
import pandas as pd 
from MAUC import MAUC

latentVecDim = 10
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

    for tau in all_tau:
        datasetFile = open("latent_data_sets/ldim" + str(latentVecDim) + "_tau" + str(tau) +"_basisLS.txt","r")
        #initialize X, Y
        X = []
        Y = [] # list of list of classes [[cl1], [cl2], ..., [cln] ]
        Y_normal = [] # list of classes [cl1, cl2, ... cln]
        #read data set
        for L in datasetFile:
            XY = L.split('\t')
            X.append([float(z) for z in XY[0:len(XY)-1]]) # get feature values for object L
            y = XY[len(XY)-1].split('\n')
            Y.append([int(y[0])]) # get object L label
            Y_normal.extend([int(y[0])]) # get object L label
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
        #YConverted = []
        #YConverted.extend(y for y in Y_test_all_normal)
        mapped = list(zip(Y_test_all_normal, Y_score_all))
        clf_MAUC[clf_name + str(tau)] = MAUC(mapped, len(clss))



# consolidate results data
dataMAUC = [] # convert to dataframe 
for clf in clfs:
    clf_result_mauc = [clf] # append classifier name
    for tau in all_tau:
        clf_result_mauc.append(clf_MAUC[clf + str(tau)])
    dataMAUC.append(clf_result_mauc)

# Create the pandas DataFrame 
dmauc = pd.DataFrame(dataMAUC, columns = ['Classifier', 'tau = 5', 'tau = 10', 'tau = 20']) 
dmauc = pd.melt(dmauc, id_vars="Classifier", var_name = 'group', value_name="MAUC")

# print dataframe. 
print(dmauc) 
colors = ['navy', 'turquoise', 'darkorange']
sns.set_palette(colors)
# save mauc barplot for tau. 
fig = sns.factorplot(x='Classifier', y='MAUC', hue='group', data=dmauc, kind='bar')
plt.title('Model performance for latent space dimension = ' + str(latentVecDim))
fig.savefig("results/consolidated/ldim" + str(latentVecDim) + "_MAUC.png")

consolidated_results = open("results/consolidated/numeric_ldim" + str(latentVecDim) + "_MAUC.txt","w")

for L in dataMAUC:
    consolidated_results.writelines(str(L) + "\n")


#interpreting AUROC AND AUPR:
# https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/
# http://www.chioka.in/differences-between-roc-auc-and-pr-auc/