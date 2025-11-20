#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM"
# pathdat             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/data"
# pathfig             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/plots"

# path                = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/scripts"

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
tf.random.set_seed(123)

from functions import LGSSM, SVSSM
from functions import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from functions import ParticleFilter
from functions import EDH, LEDH, KernelPFF

############ 
# Simulate #
############



nT = 100
nD = 5





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


A       = tf.linalg.diag(tf.range(0.05,0.98,0.98/nD, dtype=tf.float64)[:nD])
X, Y    = SVSSM(nT, nD, A=A)
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
    
X_PF, ess_PF, weights_PF, particles_PF = ParticleFilter(Y, N=10, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_PF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_EDH, ess_EDH, weights_EDH, Jx_EDH, Jw_EDH = EDH(Y, N=10, A=A, stepsize=0.2)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
X_EDH2, ess_EDH2 = EDH(Y, N=10, A=A, stepsize=0.2, method="EKF")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EDH2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 
X_LEDH, ess_LEDH = LEDH(Y, N=10, A=A, stepsize=0.2)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_LEDH2, ess_LEDH2 = LEDH(Y, N=10, A=A, stepsize=0.2, method="EKF")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 
X_KPFF = KernelPFF(Y, N=10, A=A, stepsize=0.1)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_KPFF2 = KernelPFF(Y, N=10, A=A, stepsize=0.2, method="scalar")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

    
     
 

    
    
    
    
    

A       = tf.eye(nD, dtype=tf.float64) * 0.95
X, Y    = SVSSM(nT, nD, A=A)
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
    
X_PF, ess_PF = ParticleFilter(Y, N=10, A=A)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_PF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_EDH, ess_EDH = EDH(Y, N=10, A=A, stepsize=0.2)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
X_EDH2, ess_EDH2 = EDH(Y, N=10, A=A, stepsize=0.2, method="EKF")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EDH2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 
X_LEDH, ess_LEDH = LEDH(Y, N=10, A=A, stepsize=0.2)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_LEDH2, ess_LEDH2 = LEDH(Y, N=10, A=A, stepsize=0.2, method="EKF")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 
X_KPFF = KernelPFF(Y, N=10, A=A, stepsize=0.1)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

X_KPFF2 = KernelPFF(Y, N=10, A=A, stepsize=0.2, method="scalar")
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

    
     
 

    