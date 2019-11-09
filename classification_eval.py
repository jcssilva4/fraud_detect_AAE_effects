from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

seed_value = 1234
#initialize X, Y
X = []
Y = []
K = 5 #number of folds
#read data set
tau = 20
datasetFile = open("latent_data_sets/tau" + str(tau) +"_basisLS.txt","r")
for L in datasetFile:
	XY = L.split('\t')
	X.append([float(z) for z in XY[0:len(XY)-1]]) # get feature values for object L
	y = XY[len(XY)-1].split('\n')
	Y.append([int(y[0])]) # get object L label
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
clfSVM = OneVsRestClassifier(svm.SVC(kernel = 'linear', random_state = seed_value))

SVMprcurve_fold = []
foldCounter = 1
nPointsCurve = 0
Y_test_all = []
Y_score_all = []
for train_index, test_index in skf_idxs:
	print("fold: ", foldCounter)
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = [X[i] for i in train_index], [X[j] for j in test_index]
	Y_train, Y_test = [Y[i] for i in train_index], [Y[j] for j in test_index]
	Y_train = MultiLabelBinarizer().fit_transform(Y_train)
	Y_test = MultiLabelBinarizer().fit_transform(Y_test)
	clfSVM.fit(X_train, Y_train)
	y_score = clfSVM.decision_function(X_test)
	# Why i did this?? https://stackoverflow.com/questions/26587759/plotting-precision-recall-curve-when-using-cross-validation-in-scikit-learn
	for i in range(len(clss)):
		if(foldCounter == 1):
			Y_test_all.append([])
			Y_score_all.append([])
		Y_test_all[i].extend(Y_test[:, i])
		Y_score_all[i].extend(y_score[:, i])
	foldCounter = foldCounter + 1

# multi-class pr curve plot: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html 
precision = dict()
recall = dict()
average_precision = dict()
# For each class
for i in range(len(clss)):
	precision[i], recall[i], _ = precision_recall_curve(Y_test_all[i], Y_score_all[i])
	average_precision[i] =  average_precision_score(Y_test_all[i], Y_score_all[i])
# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test_all[i], Y_score_all[i], average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
#pr curves
labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))
clssname = ['Regular', 'Global Anomaly', 'Local Anomaly']
for i, color in zip(range(len(clss)), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for {0} class (area = {1:0.2f})'.format(clssname[i], average_precision[i]))
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
foldCounter	= foldCounter + 1
#after ploting all k fold curves, plot mean curve and compare
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve per class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
plt.show()