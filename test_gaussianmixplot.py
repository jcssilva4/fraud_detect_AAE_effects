
import os, sys, random, io, urllib
from datetime import datetime
import time #elapsed time

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
from mpl_toolkits.mplot3d import Axes3D
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
#latent space configs
latentVecDim = 3 # original value = 2


# creation of the imposed latent prior distribution
# define the number of gaussians
tau = 20
# define radius of each gaussian
radius = 0.8
# define the sigma of each gaussian
sigma = 0.01
# define the dimensionality of each gaussian
dim = latentVecDim
# determine x and y coordinates of the target mixture of gaussians
centroids_dim_n = []
for i in range(0, latentVecDim):
    if i%2 == 0: #if i is an even dim, then
        centroids_dim_n.append((radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2)
    else:
        centroids_dim_n.append((radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2)
#print(centroids_dim_n)
#x_centroid = (radius * np.sin(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
#y_centroid = (radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
# determine each gaussians mean (centroid) and standard deviation
mu_gauss = np.vstack(centroids_dim_n).T
# determine the number of samples to be created per gaussian
samples_per_gaussian = 100
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

# init the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection = '3d')

# plot reconstruction error scatter plot
ax.scatter(z_continous_samples_all[:, 0], z_continous_samples_all[:, 1], z_continous_samples_all[:, 2], c='C0', marker="o", edgecolors='w', linewidth=0.5) 
ax.set_xlabel("z1")
ax.set_ylabel("z2")
ax.set_zlabel("z3")
# add plot title
ax.set_title('Prior Latent Space Distribution $p(z)$');
plt.show()
'''

#latent space configs
latentVecDim = 2 # original value = 2


# creation of the imposed latent prior distribution
# define the number of gaussians
tau = 20
# define radius of each gaussian
radius = 0.8
# define the sigma of each gaussian
sigma = 0.01
# define the dimensionality of each gaussian
dim = latentVecDim
# determine x and y coordinates of the target mixture of gaussians
centroids_dim_n = []
for i in range(0, latentVecDim):
    if i%2 == 0: #if i is an even dim, then
        centroids_dim_n.append((radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2)
    else:
        centroids_dim_n.append((radius * np.sin(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2)
#print(centroids_dim_n)
#x_centroid = (radius * np.sin(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
#y_centroid = (radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
# determine each gaussians mean (centroid) and standard deviation
mu_gauss = np.vstack(centroids_dim_n).T
# determine the number of samples to be created per gaussian
samples_per_gaussian = 100
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

# init the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

# plot reconstruction error scatter plot
ax.scatter(z_continous_samples_all[:, 0], z_continous_samples_all[:, 1], c='C0', marker="o", edgecolors='w', linewidth=0.5) 
ax.set_xlabel("z1")
ax.set_ylabel("z2")
# add plot title
ax.set_title('Prior Latent Space Distribution $p(z)$');
plt.show()