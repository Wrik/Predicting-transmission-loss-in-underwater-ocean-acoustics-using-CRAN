#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import scipy
import time
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import metrics

from tensorflow.keras import layers, losses
from tensorflow.keras.models import save_model, load_model, Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, concatenate
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from numpy import inf


# In[2]:


# physical and data parameters
zmax = 5000 # depth
rmax = 100*1e3 # Range
nz = 2049 # number of grid points in x 
nr = 352 # number of snapshots along range for each source
dz = zmax/(nz-1) # range-wise discretization/receiver depth interval
dr = rmax/(nr-1) # range-wise discretization 
nsrcs = 21 # number of depth-distributed training sources
nval = 3 # number of depth-distributed validation sources
ntest = 12 # number of depth-distributed test sources


# In[3]:


# Reading and processing training set
train_data = np.zeros((nsrcs*nr,nz))
for j in range(nsrcs):
    filename = str(os.path.abspath(os.getcwd())) + '\training_sets\case' + str(1) + 'sid' + str(j+1) + "_Coh_gb.shd.mat.csv"
    train_data[j*nr:(j+1)*nr,:] = np.loadtxt(filename,delimiter=',')       


# In[4]:


# Reading and processing validation set
val_data = np.zeros((nval*nr,nz))
for k in range(nval):
    j = k + nsrcs
    filename = str(os.path.abspath(os.getcwd())) + "\validation_sets\case" + str(1) + 'sid' + str(j+1) + "_Coh_gb.shd.mat.csv"
    val_data[k*nr:(k+1)*nr,:] = np.loadtxt(filename,delimiter=',')      


# In[5]:


# Reading and processing test set
test_data = np.zeros((ntest*nr,nz))
for k in range(ntest):
    j = k + nsrcs + nval
    filename = str(os.path.abspath(os.getcwd())) + "\test_sets\case" + str(1) + 'sid' + str(j+1) + "_Coh_gb.shd.mat.csv"
    test_data[k*nr:(k+1)*nr,:] = np.loadtxt(filename,delimiter=',')


# In[6]:


# network parameters
Nf = nz # feature space
latent_dim = 64 # dimension of latent states
lkernel1 = 14 # (Conv1D layer 1 filter size)
lkernel2 = 10 # (Conv1D layer 1filter size)
lkernel3 = 6 # (Conv1D layer 1 filter size)
nfilter = 16 # (Conv1D layer 1 number of filters)
r = 11 # number of range-wise sequences
Ns = int(nr/r) # sequence length
Nr = 512 # size of LSTM hidden layer


# In[7]:


# Normalize training and validation data
x_train = train_data
x_val = val_data
logical1 = x_train>=200
x_train = x_train-100*logical1
logical2 = x_val>=200
x_val = x_val-100*logical2
mu = np.min([40.0,np.min(x_train)])
sigsq = np.square(np.abs(np.max(x_train)-mu)) 
x_train_norm = (x_train - mu)/(1.0*np.sqrt(sigsq))
x_val_norm = (x_val - mu)/(1.0*np.sqrt(sigsq))
x_train_norm = x_train_norm[...,tf.newaxis]
x_val_norm = x_val_norm[...,tf.newaxis]


# In[8]:


# function for computing ssim
def ssim(ypred,ytrue):
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    L=np.max((np.max(ytrue) - np.min(ytrue),np.max(ypred) - np.min(ypred)))
    sigmasqx = np.var(ytrue)
    sigmasqy = np.var(ypred)
    c_xy = (2*np.sqrt(sigmasqx)*np.sqrt(sigmasqy)+(0.02*L)**2)/(sigmasqx+sigmasqy+(0.02*L)**2)
    muy = np.mean(ypred)
    mux = np.mean(ytrue)
    l_xy = (2*mux*muy+(0.01*L)**2)/(mux**2+muy**2+(0.01*L)**2)
    sigma_xy = np.mean(np.multiply((ytrue-mux),(ypred-muy)))
    s_xy = (sigma_xy+0.5*(0.02*L)**2)/((np.sqrt(sigmasqx)*np.sqrt(sigmasqy))+0.5*(0.02*L)**2)
    ssim = np.power(l_xy,alpha)*np.power(c_xy,beta)*np.power(s_xy,gamma)
    return ssim


# # 1d CRAN

# In[9]:


# standard MSE loss
def norm_mse_loss(x, x_hat):
    norm_mse_loss = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(x, x_hat))
    return norm_mse_loss


# In[10]:


# load trained 1d CAN model
my_can1d_model = load_model(r'provide path',
                           custom_objects={'norm_mse_loss': norm_mse_loss,}
                                          )


# In[11]:


my_can1d_model.summary()


# # CAN Validation

# In[12]:


# denormalize and compute mean validation SSIM
x_hat_val = np.multiply(my_can1d_model(x_val_norm),np.sqrt(sigsq))+mu
x_hat_val = np.reshape(x_hat_val,[x_hat_val.shape[0],Nf])
pvaldiff = x_hat_val - x_val
ssim_can_vec_val = np.zeros(nval)
for i in range(nval):
    ssim_can_vec_val[i] = ssim(x_val[i*nr:(i+1)*nr,:],x_hat_val[i*nr:(i+1)*nr,:])
print('Mean SSIM CAN val:',np.mean(ssim_can_vec_val)) 
