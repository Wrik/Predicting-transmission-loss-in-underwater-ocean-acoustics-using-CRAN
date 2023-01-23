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
nr = 352
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


# # SS-LSTM Validation

# In[13]:


# load trained 1d CAN model
my_lstm_model = load_model(r'provide path',
                           custom_objects={'norm_mse_loss': norm_mse_loss,}
                                          )


# In[14]:


my_lstm_model.summary()


# In[15]:


# retrieving trained encoder and decoder networks
for i in range(len(my_can1d_model.layers)):
    layer = my_can1d_model.layers[i]
    print(i, layer.name, layer.output.shape)
myCNNencoder = my_can1d_model.layers[1]
myCNNdecoder = my_can1d_model.layers[2] 


# In[16]:


enc_states_val = myCNNencoder(x_val_norm) # obtain encoded validation physical data 
# Normalize validation latent states similar to training
beta_ls = 2
enc_states_val_norm = enc_states_val/beta_ls
# LSTM training data organization: separate all latent states into input and output sequence sets for each source
# Note: during prediction phase only the first sequence set (Ns snapshots) is required as input for autoregresive prediction 
enc_states_in_val = np.zeros((nval*Ns,latent_dim))
enc_states_out_val = np.zeros((nval*(nr-Ns),latent_dim))
for j in range(nval):
    enc_states_in_val[j*Ns:(j+1)*Ns,:] = enc_states_val_norm[j*nr:j*nr+Ns,:]
    enc_states_out_val[j*(nr-Ns):(j+1)*(nr-Ns),:] = enc_states_val_norm[j*nr+Ns:(j+1)*nr,:]
enc_states_in_val = np.reshape(enc_states_in_val,[-1,Ns,latent_dim])
enc_states_out_val = np.reshape(enc_states_out_val,[-1,Ns,latent_dim])


# In[17]:


# Obtain lstm autoregressive prediction output from initiating input sequence 
nTx = r-1 
eout_hat_val = np.zeros((nTx*nval,Ns,latent_dim))
for i in range(nval):
    eout_hat_val[i*nTx:i*nTx+1,:,:] = my_lstm_model(enc_states_in_val[i:i+1,:,:])
    for Tid in range(nTx-1):
        eout_hat_val[i*nTx+Tid+1:i*nTx+1+Tid+1,:,:] = my_lstm_model(eout_hat_val[i*nTx+Tid:i*nTx+Tid+1,:,:])      
lstates_hat_val = beta_ls*np.reshape(eout_hat_val,[-1,latent_dim]) # Reshape into physical dimensions and denormalize


# In[18]:


# retrieve high-dimensional physical data from latent states outputs via CNN decoder
dec_output_val = myCNNdecoder(lstates_hat_val[0:3*(nr-Ns),:])
y_hat_val = np.multiply(np.reshape(dec_output_val,[-1,Nf]),np.sqrt(sigsq))+mu
y_val= np.zeros((nval*(nr-Ns),nz))
for j in range(nval):
    y_val[j*(nr-Ns):(j+1)*(nr-Ns),:] = x_val[j*nr+Ns:(j+1)*nr,:]
yvaldiff = y_hat_val - y_val
# Compute CRAN prediction mean SSIM
# Note: in the prediction phase of CRAN operation SS-LSTM validation is same as CRAN validation
ssim_sslstm_vec_val = np.zeros(nval)
for i in range(nval):
    ssim_sslstm_vec_val[i] = ssim(y_val[i*(nr-Ns):(i+1)*(nr-Ns),:],y_hat_val[i*(nr-Ns):(i+1)*(nr-Ns),:])
print('SSIM SS-LSTM val:',ssim_sslstm_vec_val)   
print('Mean SSIM SS-LSTM val:',np.mean(ssim_sslstm_vec_val)) 


# # CRAN testing

# In[19]:


# Normalize test data
x_test = test_data
logical3 = x_test>=200
x_test = x_test-100*logical3
x_test_norm = (x_test - mu)/(1.0*np.sqrt(sigsq))
x_test_norm = x_test_norm[...,tf.newaxis]


# In[20]:


enc_states_test = myCNNencoder(x_test_norm) # obtain encoded validation physical data 
enc_states_test_norm = enc_states_test/beta_ls # Normalize test latent states 
# LSTM training data organization: separate all latent states into input and output sequence sets for each source
# Note: during prediction phase only the first sequence set (Ns snapshots) is required as input for autoregresive prediction
enc_states_in_test = np.zeros((ntest*Ns,latent_dim))
enc_states_out_test = np.zeros((ntest*(nr-Ns),latent_dim))
for j in range(ntest):
    enc_states_in_test[j*Ns:(j+1)*Ns,:] = enc_states_test_norm[j*nr:j*nr+Ns,:]
    enc_states_out_test[j*(nr-Ns):(j+1)*(nr-Ns),:] = enc_states_test_norm[j*nr+Ns:(j+1)*nr,:]
enc_states_in_test = np.reshape(enc_states_in_test,[-1,Ns,latent_dim])
enc_states_out_test = np.reshape(enc_states_out_test,[-1,Ns,latent_dim])


# In[21]:


# Obtain lstm autoregressive prediction output from initiating input sequence 
eout_hat_test = np.zeros((nTx*ntest,Ns,latent_dim))
for i in range(ntest):
    eout_hat_test[i*nTx:i*nTx+1,:,:] = my_lstm_model(enc_states_in_test[i:i+1,:,:])
    for Tid in range(nTx-1):
        eout_hat_test[i*nTx+Tid+1:i*nTx+1+Tid+1,:,:] = my_lstm_model(eout_hat_test[i*nTx+Tid:i*nTx+Tid+1,:,:])      
lstates_hat_test = beta_ls*np.reshape(eout_hat_test,[-1,latent_dim]) # Reshape into physical dimensions and denormalize


# In[22]:


# retrieve high-dimensional physical data from latent states outputs via CNN decoder
dec_output_test = myCNNdecoder(lstates_hat_test) 
y_hat_test = np.multiply(np.reshape(dec_output_test,[-1,Nf]),np.sqrt(sigsq))+mu
y_test= np.zeros((ntest*(nr-Ns),nz))
for j in range(ntest):
    y_test[j*(nr-Ns):(j+1)*(nr-Ns),:] = x_test[j*nr+Ns:(j+1)*nr,:]
ytestdiff = y_hat_test - y_test
# Compute CRAN prediction mean SSIM
ssim_cran_vec_test = np.zeros(ntest)
for i in range(ntest):
    ssim_cran_vec_test[i] = ssim(y_test[i*(nr-Ns):(i+1)*(nr-Ns),:],y_hat_test[i*(nr-Ns):(i+1)*(nr-Ns),:])
print('SSIM CRAN test:',ssim_cran_vec_test)   
print('Mean SSIM CRAN test:',np.mean(ssim_cran_vec_test)) 


# # Post-processing utilities: to be used as-is only for observing plots via Jupyter notebook or similar GUI

# In[23]:


zvec=[]
rvec=[]
[ zvec.append(dz*(i+1)) for i in range(Nf) ]
[ rvec.append(dr*i) for i in range(nr) ]
zvec = np.array(zvec)
rvec = np.array(rvec)+dr
zmat, rmat = np.meshgrid(zvec, rvec)


# In[24]:


cid = 4
ytrue = x_test[cid*nr:(cid+1)*nr]
ypred = np.append(x_test[cid*nr:cid*nr+Ns],y_hat_test[cid*(nr-Ns):(cid+1)*(nr-Ns)],axis=0)
print(ytrue.shape,ypred.shape)

plt.rcParams.update({'font.size':12})
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
levels = np.linspace(mu,220,64)
cs = ax.contourf(rmat.transpose()/1000, zmat.transpose(), ytrue.transpose(), 
                  levels=levels, cmap=cm.jet)
colorbar = plt.colorbar(cs)
ax.set_xlabel("$R$ (km)")
ax.set_ylabel("$z$ (m)")
ax.invert_yaxis()

plt.rcParams.update({'font.size':14})
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
levels = np.linspace(mu,220,64)
cs = ax.contourf(rmat.transpose()/1000, zmat.transpose(), ypred.transpose(), 
                  levels=levels, cmap=cm.jet)
colorbar = plt.colorbar(cs)
ax.set_xlabel("$R$ (km)")
ax.set_ylabel("$z$ (m)")
ax.invert_yaxis()


# In[25]:


ifig = 1
n = 3
m = 1
plt.rcParams.update({'font.size':14})
plt.rcParams['text.usetex'] = True
for bid in range(m):
    for i in range(n):
        y = ytrue[:,500+400*i]
        y_hat = ypred[:,500+400*i]
        l1 = np.divide((y_hat-y),y)
        fig,ax = plt.subplots(figsize=(6,4), tight_layout=True)
        plt.plot(rvec/1000.0,y_hat,'b--',label="pred")
        plt.plot(rvec/1000.0,y,'r',label="true",linewidth=1)
        plt.grid()
        plt.legend(numpoints=1)
        plt.legend(loc='lower right')
        plt.ylim(40,220)
        plt.xlabel("$R$ (Km)")
        plt.ylabel("Loss (dB)") 
        ax.invert_yaxis()
        print(np.round_((500+400*i)*dz,decimals=3))

l1_1 = np.divide((ypred[:,500]-ytrue[:,500]),ytrue[:,500])  
l1_3 = np.divide((ypred[:,500+400*2]-ytrue[:,500+400*2]),ytrue[:,500+400*2]) 
fig,ax = plt.subplots(figsize=(6,4), tight_layout=True)
plt.plot(rvec/1000.0,l1_1,'b--',label="1200 m")
plt.plot(rvec/1000.0,l1_3,'r:',label="3200 m")
plt.grid()
plt.legend(numpoints=1)
plt.xlabel("$R$ (Km)")
plt.ylabel("$1 - {\mathrm{TL}}_{pred}/{\mathrm{TL}}_{true}$")


# In[ ]:




