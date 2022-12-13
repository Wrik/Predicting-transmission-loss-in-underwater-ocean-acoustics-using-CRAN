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


# In[3]:


# physical and data parameters
zmax = 5000 # depth
rmax = 100*1e3 # Range
nz = 2049 # number of grid points in x 
nr = 352 # number of snapshots along range for each source
dz = zmax/(nz-1) # range-wise discretization/receiver depth interval
dr = rmax/(nr-1) # range-wise discretization 
nsrcs = 21 # number of depth-distributed training sources


# In[4]:


# Reading and processing training set
start_time = time.time()
train_data = np.zeros((nsrcs*nr,nz))
for j in range(nsrcs):
    filename = str(os.path.abspath(os.getcwd())) + '\training_sets\case' + str(1) + 'sid' + str(j+1) + "_Coh_gb.shd.mat.csv"
    train_data[j*nr:(j+1)*nr,:] = np.loadtxt(filename,delimiter=',')       


# In[5]:


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


# In[10]:


# Normalize training data
x_train = train_data
logical1 = x_train>=200
x_train = x_train-100*logical1
mu = np.min([40.0,np.min(x_train)])
sigsq = np.square(np.abs(np.max(x_train)-mu)) 
x_train_norm = (x_train - mu)/(1.0*np.sqrt(sigsq))
x_train_norm = x_train_norm[...,tf.newaxis]


# # 1d CRAN

# In[11]:


# standard MSE loss
def norm_mse_loss(x, x_hat):
    norm_mse_loss = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(x, x_hat))
    return norm_mse_loss


# In[12]:


# encoder - encapulste 1D CNN and maxpooling layers as encoder layer
x = Input(shape=(Nf,1))
c1 = Conv1D(nfilter, (lkernel1), activation=LeakyReLU(alpha=0.05), padding='valid', strides=1)(x)
m1 = MaxPooling1D(pool_size = (4), padding='same')(c1)
c2 = Conv1D(2*nfilter, (lkernel2), activation=LeakyReLU(alpha=0.05), padding='valid', strides=1)(m1)
m2 = MaxPooling1D(pool_size = (4), padding='same')(c2)
c3 = Conv1D(3*nfilter, (lkernel3), activation=LeakyReLU(alpha=0.05), padding='valid', strides=1)(m2)
f1 = Flatten()(c3)
d1 = Dense(latent_dim)(f1)
CNN_encoder = Model(inputs=x,outputs=d1, name="CNNencoder")
enc_states = CNN_encoder(x)

# decoder - encapulste 1D trCNN and upsampling layers as decoder layer
s = Input(shape=(latent_dim))
d3 = Dense(f1.shape[1])(s)
r1 = Reshape((c3.shape[1],3*nfilter))(d3)
ct1 = Conv1DTranspose(2*nfilter, kernel_size=lkernel3, activation=LeakyReLU(alpha=0.05),strides=1, padding='valid')(r1)
u1 = UpSampling1D((4))(ct1)
ct2 = Conv1DTranspose(nfilter, kernel_size=lkernel2, activation=LeakyReLU(alpha=0.05),strides=1, padding='valid')(u1)
u2 = UpSampling1D((4))(ct2)
ct3 = Conv1DTranspose(1, kernel_size=(lkernel1), activation=LeakyReLU(alpha=0.05), strides=1, padding='valid')(u2)
CNN_decoder = Model(inputs=s, outputs=ct3, name="CNNdecoder")
x_hat = CNN_decoder(enc_states)

can1d_model = Model(inputs=x, outputs=x_hat, name="CAN1D")


# In[13]:


# define custom learning rate
lr_schedule_can = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.0005,
    decay_steps=350,
    decay_rate=0.995)


# In[14]:


can1d_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_can), 
                   loss = norm_mse_loss,
                  )


# In[15]:


can1d_model.summary()


# In[28]:


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


# # 1d CAN training

# In[ ]:

start_time = time.time()
History = can1d_model.fit(x_train_norm, x_train_norm,
               batch_size=64,
               epochs=200,
               verbose=0
              )
print("Time elapsed (s)",(time.time() - start_time))
train_loss = np.array(History.history['loss'])
np.savetxt("can1d_hyperparams_train_loss.txt", train_loss, delimiter=",")


# In[ ]:


save_model(can1d_model,r'provide path',
          overwrite=True, include_optimizer=True)           


# In[ ]:


my_can1d_model = can1d_model


# In[55]:


# denormalize and compute mean training SSIM
x_hat_train = np.multiply(my_can1d_model(x_train_norm),np.sqrt(sigsq))+mu
x_hat_train = np.reshape(x_hat_train,[x_hat_train.shape[0],Nf])
ptraindiff = x_hat_train - x_train
ssim_can_vec_train = np.zeros(nsrcs)
for i in range(nsrcs):
    ssim_can_vec_train[i] = ssim(x_train[i*nr:(i+1)*nr,:],x_hat_train[i*nr:(i+1)*nr,:])
print('Mean SSIM CAN train:',np.mean(ssim_can_vec_train))   

