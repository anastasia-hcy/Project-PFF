#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM"
pathdat             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/data"
pathfig             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/plots"

# path                = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/scripts"
# pathdat             = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/data"

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


from functions import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter
from functions import LEDH
from functions import LGSSM
from functions import SVSSM

############ 
# Simulate #
############








nT = 100
nD = 1










X, Y = LGSSM(nT,nD)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_KF = KalmanFilter(Y)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 











X, Y    = SVSSM(nT,nD)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75)
    plt.show() 

X_EKF = ExtendedKalmanFilter(Y)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_UKF = UnscentedKalmanFilter(Y)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_UKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
X_PF = ParticleFilter(Y, N=10)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_PF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_LEDH = LEDH(Y, N=10, stepsize=0.1)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 

X_LEDH = LEDH(Y, N=10, stepsize=0.1, method="UKF")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
    
    
    
    
    
    
    
    

A       = tf.eye(nD, dtype=tf.float64) * 0.75
X, Y    = SVSSM(nT,nD, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75)
    plt.show() 
    
X_EKF = ExtendedKalmanFilter(Y, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_UKF = UnscentedKalmanFilter(Y, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_UKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
X_PF = ParticleFilter(Y, N=10, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_PF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_LEDH = LEDH(Y, A=A, N=10, stepsize=0.1)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 
    