#################
# Set directory #
#################

# path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/"
# pathdat             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/data/"
# pathfig             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/plots/"

path                = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/scripts"
pathdat             = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/data"

import os, sys
os.chdir(path)
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
# Simulate #
############

nT = 100
nD = 1 

from functions import LGSSM, KalmanFilter

X, Y = LGSSM(nT,nD)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, c=colors[i]) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75, c=colors[i], linestyle='dashed') 
    plt.show() 

X_KF = KalmanFilter(Y,V=tf.eye(nD)*10,W=tf.eye(nD)*10)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, c=colors[i]) 
    plt.plot(X_KF[:,i], linewidth=1, alpha=0.75, c=colors[i], linestyle='dashed') 
    plt.show() 


from functions import SVSSM, ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter


A       = tf.eye(nD) * 0.99
X, Y    = SVSSM(nT,nD, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75)
    plt.show() 


X_f     = KalmanFilter(Y, A=A)

X_fil   = ExtendedKalmanFilter(Y, A=A)

for o in X_fil:
    print(o, '\n')

X_fil[0] + 1e-5
tf.math.exp(X_fil[0])



X_filt  = UnscentedKalmanFilter(Y, A=A)
X_filt2 = ParticleFilter(Y, A=A, N=5)

for i in range(nD):
    plt.plot(X[:,i], linewidth=1)  
    plt.plot(X_f[:,i], linewidth=1, alpha=0.75, linestyle='dotted', c=colors[1]) 
    plt.show() 

for i in range(nD):
    plt.plot(X[:,i], linewidth=1)  
    plt.plot(X_fil[:,i], linewidth=1, alpha=0.75, linestyle='dotted', c=colors[2])  
    plt.show() 

for i in range(nD):
    plt.plot(X[:,i], linewidth=1)  
    plt.plot(X_filt[:,i], linewidth=1, alpha=0.75, linestyle='dotted', c=colors[3]) 
    plt.show() 

for i in range(nD):
    plt.plot(X[:,i], linewidth=1)  
    plt.plot(X_filt2[:,i], linewidth=1, alpha=0.75, linestyle='dotted')  
    plt.show() 
    
