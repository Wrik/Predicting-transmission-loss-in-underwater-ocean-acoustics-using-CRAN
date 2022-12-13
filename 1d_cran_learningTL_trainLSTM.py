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
nr = 352
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


# In[17]:


# propagator
p = keras.Input(shape=[Ns,latent_dim])
singleshot_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(Nr, 
                         return_sequences=False),
    tf.keras.layers.Dense(units=latent_dim*Ns),
    tf.keras.layers.Reshape([Ns, latent_dim])
    ],name="SS-LSTM")

p_hat = singleshot_lstm_model(p)


# In[18]:


lr_schedule_rnn = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.0005,
    decay_steps=400,
    decay_rate=0.9975)


# In[19]:


singleshot_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_rnn), 
                   loss = norm_mse_loss,
                  )


# In[20]:


singleshot_lstm_model.summary()


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



# # SS-LSTM training


# load trained convolutional autoencoder


my_can1d_model = load_model(r'provide path',
                           custom_objects={'norm_mse_loss': norm_mse_loss,}
                                          )


# In[ ]:


my_can1d_model.summary()


# In[37]:


# retrieving latent states from trained 1d CAN
for i in range(len(my_can1d_model.layers)):
    layer = my_can1d_model.layers[i]
    print(i, layer.name, layer.output.shape)
myCNNencoder = my_can1d_model.layers[1]
myCNNdecoder = my_can1d_model.layers[2]   
enc_states = myCNNencoder(x_train_norm)


# In[57]:


# Normalize to ensure data lies [-1,1] for better training with tanh activation.
# Tune beta_ls accordingly
beta_ls = 2 
enc_states_norm = enc_states/beta_ls


# In[58]:


# LSTM training data organization: separate all latent states into input and output sequence sets for each source
enc_states_in = np.zeros((nsrcs*(nr-Ns),latent_dim))
enc_states_out = np.zeros((nsrcs*(nr-Ns),latent_dim))
for j in range(nsrcs):
    enc_states_in[j*(nr-Ns):(j+1)*(nr-Ns),:] = enc_states_norm[j*nr:(j+1)*nr-Ns,:]
    enc_states_out[j*(nr-Ns):(j+1)*(nr-Ns),:] = enc_states_norm[j*nr+Ns:(j+1)*nr,:]
enc_states_in = np.reshape(enc_states_in,[-1,Ns,latent_dim])
enc_states_out = np.reshape(enc_states_out,[-1,Ns,latent_dim])


# In[ ]:


# Fit the lstm model 
start_time = time.time()
History = singleshot_lstm_model.fit(enc_states_in, enc_states_out,
               batch_size=8,
               epochs=10,
               verbose=0
              )
print("Time elapsed (s)",(time.time() - start_time))
train_loss = np.array(History.history['loss'])
np.savetxt("sslstm_hyperparams_train_loss.txt", train_loss, delimiter=",")


# In[ ]:


save_model(singleshot_lstm_model,r'provide path',
          overwrite=True, include_optimizer=True)  


# In[ ]:


my_lstm_model = singleshot_lstm_model

# Obtain lstm training output sequences from training input sequences 
nTx = r-1 
eout_hat_train = np.zeros((nTx*nsrcs,Ns,latent_dim))
start_time = time.time()
for i in range(nsrcs):
    eout_hat_train[i*nTx:i*nTx+1,:,:] = my_lstm_model(enc_states_in[i*nTx:i*nTx+1,:,:])
    for Tid in range(nTx-1):
        eout_hat_train[i*nTx+Tid+1:i*nTx+1+Tid+1,:,:] = my_lstm_model(eout_hat_train[i*nTx+Tid:i*nTx+Tid+1,:,:])   
# Reset latent state sequences into order of physical data and denormalize
lstates_hat_train = beta_ls*np.reshape(eout_hat_train,[-1,latent_dim])
lstates_out_train = beta_ls*np.reshape(enc_states_out,[-1,latent_dim])


# In[60]:


# retrieve high-dimensional physical data from LSTM latent state training outputs via CNN decoder
# and compute CRAN training mean SSIM
dec_output = myCNNdecoder(lstates_hat_train[0:nsrcs*(nr-Ns),:])
y_hat_train = np.multiply(np.reshape(dec_output,[-1,Nf]),np.sqrt(sigsq))+mu
y_train = np.zeros((nsrcs*(nr-Ns),nz))
for j in range(nsrcs):
    y_train[j*(nr-Ns):(j+1)*(nr-Ns),:] = x_train[j*nr+Ns:(j+1)*nr,:]
ytraindiff = y_hat_train - y_train
ssim_sslstm_vec_train = np.zeros(nsrcs)
for i in range(nsrcs):
    ssim_sslstm_vec_train[i] = ssim(y_train[i*(nr-Ns):(i+1)*(nr-Ns),:],y_hat_train[i*(nr-Ns):(i+1)*(nr-Ns),:])
print('Mean SSIM SS-LSTM train:',np.mean(ssim_sslstm_vec_train))    


# In[ ]:




