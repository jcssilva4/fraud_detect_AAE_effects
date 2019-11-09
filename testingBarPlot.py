import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

#clf_AUROC = dict()
#clf_AUPR = dict()
clf_AUROC_regular = dict()
clf_AUROC_global = dict()
clf_AUROC_local = dict()
clf_AUPR_regular = dict()
clf_AUPR_global = dict()
clf_AUPR_local = dict()

# initialize list of lists 
dataAUROC = [['NB', 0.5, 0.6, 0.4], ['RF', 0.5, 0.5, 0.4], ['SVM', 0.6, 0.3, 0.4]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(dataAUROC, columns = ['Classifier', 'Regular', 'Global', 'Local']) 
df = pd.melt(df, id_vars="Classifier", var_name="group", value_name="AUROC")
 
# print dataframe. 
print(df) 
colors = ['navy', 'turquoise', 'darkorange']
sns.set_palette(colors)
ax = sns.factorplot(x='Classifier', y='AUROC', hue='group', data=df, kind='bar')
plt.title('Model performance per class (AUROC)')
plt.show()