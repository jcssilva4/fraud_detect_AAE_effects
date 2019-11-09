'''
use this file to analyse results
load pre-trained model (trained in cluster)
'''

# importing python utility libraries
import os, sys, random, io, urllib
from datetime import datetime

# importing pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

# importing data science libraries
import pandas as pd
import random as rd
import numpy as np

# importing python plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from IPython.display import Image, display

#AAE parts
from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

USE_CUDA = False

'''
The raw.githubusercontent.com domain is used to serve unprocessed versions of files stored in GitHub repositories.
 If you browse to a file on GitHub and then click the Raw link, that's where you'll go.
'''

# print current Python version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The Python version: {}'.format(now, sys.version))

# print current PyTorch version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The PyTorch version: {}'.format(now, torch.__version__))

# init deterministic seed
seed_value = 1234
rd.seed(seed_value) # set random seed
np.random.seed(seed_value) # set numpy seed
torch.manual_seed(seed_value) # set pytorch seed CPU
if (torch.backends.cudnn.version() != None and USE_CUDA == True):
    torch.cuda.manual_seed(seed_value) # set pytorch seed GPU

if not os.path.exists('./results'): os.makedirs('./results')  # create results directory

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

## PRETRAINED MODEL RESTORATION
mini_batch_size = 128

# creation of the imposed latent prior distribution
# define the number of gaussians
tau = 20
# define radius of each gaussian
radius = 0.8
# define the sigma of each gaussian
sigma = 0.01
# define the dimensionality of each gaussian
dim = 2
# determine x and y coordinates of the target mixture of gaussians
x_centroid = (radius * np.sin(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
y_centroid = (radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
# determine each gaussians mean (centroid) and standard deviation
mu_gauss = np.vstack([x_centroid, y_centroid]).T
# determine the number of samples to be created per gaussian
samples_per_gaussian = 100000
# iterate over the number of distinct gaussians
for i, mu in enumerate(mu_gauss):
    # case: first gaussian
    if i == 0:
        # randomly sample from gaussion distribution 
        z_continous_samples_all = np.random.normal(mu, sigma, size=(samples_per_gaussian, dim))
    # case: non-first gaussian
    else:
        # randomly sample from gaussian distribution
        z_continous_samples = np.random.normal(mu, sigma, size=(samples_per_gaussian, dim))
        # collect and stack new samples
        z_continous_samples_all = np.vstack([z_continous_samples_all, z_continous_samples])

# restore pretrained model checkpoint
if tau == 5:
    print("restoring tau = 5 model")
    encoder_model_name = 'https://github.com/jcssilva4/finalFraudDetect_DeepLearning/blob/master/models/tau5/20190930-15_29_22_ep_5000_encoder_model.pth?raw=true'
    decoder_model_name = 'https://github.com/jcssilva4/finalFraudDetect_DeepLearning/blob/master/models/tau5/20190930-15_29_22_ep_5000_decoder_model.pth?raw=true'
elif tau == 10:
    print("restoring tau = 10 model")
    encoder_model_name = 'https://github.com/jcssilva4/finalFraudDetect_DeepLearning/blob/master/models/tau10/20190926-23_25_55_ep_5000_encoder_model.pth?raw=true'
    decoder_model_name = 'https://github.com/jcssilva4/finalFraudDetect_DeepLearning/blob/master/models/tau10/20190926-23_25_55_ep_5000_decoder_model.pth?raw=true'
elif tau == 20:
    print("restoring tau = 20 model")
    encoder_model_name = 'https://github.com/jcssilva4/finalFraudDetect_DeepLearning/blob/master/models/tau20/20191003-18_15_36_ep_5000_encoder_model.pth?raw=true'
    decoder_model_name = 'https://github.com/jcssilva4/finalFraudDetect_DeepLearning/blob/master/models/tau20/20191003-18_15_36_ep_5000_decoder_model.pth?raw=true'

# Read stored model from the remote location
encoder_bytes = urllib.request.urlopen(encoder_model_name)
decoder_bytes = urllib.request.urlopen(decoder_model_name)

# Load tensor from io.BytesIO object
encoder_buffer = io.BytesIO(encoder_bytes.read())
decoder_buffer = io.BytesIO(decoder_bytes.read())

# init training network classes / architectures
encoder_eval = Encoder(input_size=ori_subset_transformed.shape[1], hidden_size=[256, 64, 16, 4, 2])
decoder_eval = Decoder(output_size=ori_subset_transformed.shape[1], hidden_size=[2, 4, 16, 64, 256])

# push to cuda if cudnn is available
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    encoder_eval = encoder_eval.cuda()
    decoder_eval = decoder_eval.cuda()
    
# load trained models
# since the model was trained on a gpu and will be restored in a cpu we need to provide: map_location = 'cpu'
encoder_eval.load_state_dict(torch.load(encoder_buffer, map_location = 'cpu')) 
decoder_eval.load_state_dict(torch.load(decoder_buffer, map_location = 'cpu')) 

## specify a dataloader that provides the ability to evaluate the journal entrie in an "unshuffled" batch-wise manner:
# convert pre-processed data to pytorch tensor
torch_dataset = torch.from_numpy(ori_subset_transformed.values).float()

# convert to pytorch tensor - none cuda enabled
dataloader_eval = DataLoader(torch_dataset, batch_size=mini_batch_size, shuffle=False, num_workers=0)

# determine if CUDA is available at the compute node
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    
    # push dataloader to CUDA
    dataloader_eval = DataLoader(torch_dataset.cuda(), batch_size=mini_batch_size, shuffle=False)

# VISUALIZE LATENT SPACE REPRESETATION
# set networks in evaluation mode (don't apply dropout)
encoder_eval.eval()
decoder_eval.eval()

# init batch count
batch_count = 0

# iterate over epoch mini batches
for enc_transactions_batch in dataloader_eval:

    # determine latent space representation of all transactions
    z_enc_transactions_batch = encoder_eval(enc_transactions_batch)
    
    # case: initial batch 
    if batch_count == 0:

      # collect reconstruction errors of batch
      z_enc_transactions_all = z_enc_transactions_batch
      
    # case: non-initial batch
    else:
      
      # collect reconstruction errors of batch
      z_enc_transactions_all = torch.cat((z_enc_transactions_all, z_enc_transactions_batch), dim=0)
    
    # increase batch count
    batch_count += 1

# convert to numpy array
z_enc_transactions_all = z_enc_transactions_all.cpu().detach().numpy()

# prepare plot
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)


# obtain regular transactions as well as global and local anomalies
regular_data = z_enc_transactions_all[label == 'regular']
global_outliers = z_enc_transactions_all[label == 'global']
local_outliers = z_enc_transactions_all[label == 'local']
print("|R| = " + str(len(regular_data)) + " ## |GA| = " + str(len(global_outliers)) + " ## |LA| = " + str(len(local_outliers)))

# plot reconstruction error scatter plot
ax.scatter(regular_data[:, 0], regular_data[:, 1], c='C0', marker="o", label='regular', edgecolors='w', linewidth=0.5) # plot regular transactions
ax.scatter(global_outliers[:, 0], global_outliers[:, 1], c='C1', marker="x", label='global', edgecolors='w', s=60) # plot global outliers
ax.scatter(local_outliers[:, 0], local_outliers[:, 1], c='C3', marker="x", label='local', edgecolors='w', s=60) # plot local outliers

# save base article latent space for t = tau
'''
labels
0 stands for regular 
1 stands for global A
2 stands for local A
'''
base_dataSet = open("latent_data_sets/tau" + str(tau) +"_basisLS.txt","w")
for L in ["\t".join(item) for item in regular_data.astype(str)]:
    base_dataSet.writelines(L + "\t0\n")
for L in ["\t".join(item) for item in global_outliers.astype(str)]:
    base_dataSet.writelines(L + "\t1\n")
for L in ["\t".join(item) for item in local_outliers.astype(str)]:
    base_dataSet.writelines(L + "\t2\n")
