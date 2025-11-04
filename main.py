#################
# Set directory #
#################

pathdat             = "C:/Users/anastasia/MyProjects/Codebase/data/"

import os, sys
cwd = os.getcwd()
print(f"Current working directory is: {cwd}")
sys.path.append(cwd)


############# 
# Libraries #
#############

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


############ 
# Testings #
############

nT = 365
nD = 5 

from functions import LGSSM, KalmanFilter

X, Y = LGSSM(nT,nD)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75)
    plt.show() 
    
X_fil = KalmanFilter(Y)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, c=colors[i]) 
    plt.plot(X_fil[:,i], linewidth=1, alpha=0.75, c=colors[i], linestyle='dashed') 
plt.show() 