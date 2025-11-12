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

import numpy as np 
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

##############
# Unit tests #
##############

def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

def test_sum_tuple():
    assert sum((2, 2, 2)) == 6, "Should be 6"

if __name__ == "__main__":
    test_sum()
    test_sum_tuple()
    print("Everything passed")


#####################
# Unit tests drafts #
####################

nT              = 365
nD              = 5





from functions import SE_kernel, SE_Cov

v              = tf.random.normal((nD,), 0, 1)
M              = tf.Variable(tf.zeros((nD,nD), dtype=tf.float32))
M2             = tf.Variable(tf.zeros((nD,nD), dtype=tf.float32))
for i in range(nD): 
    for j in range(nD): 
        M[i,j].assign( SE_kernel(v[i], v[j], 1.0, 1.0) )
    for j in range(i,nD):
        M2[i,j].assign( SE_kernel(v[i], v[j], 1.0, 1.0) )
        M2[j,i].assign( SE_kernel(v[i], v[j], 1.0, 1.0) )
         
np.all( M == M2 )
np.all( M == SE_Cov(nD, v) )

v2             = tf.linalg.matvec(M, v)
M3             = tf.Variable(tf.zeros((nD,nD), dtype=tf.float32))
for i in range(nD):
    for j in range(nD): 
        M3[i,j].assign(v2[i] * v[j])
np.all(
    tf.linalg.tensordot(tf.linalg.matvec(M, v), v, axes=0) == M3
) 

v3 = tf.random.normal((nD,), 0, 1)
np.all( 
    tf.linalg.diag_part( 
        tf.linalg.tensordot(tf.linalg.matvec(M, v3 ), v3, axes=0) 
    ) == tf.linalg.diag_part( v3 * M * v3) 
)

np.all( 
    tf.linalg.diag_part(M) * (v3**2) == tf.linalg.diag_part( v3 * M * v3) 
)







from functions import norm_rvs

np.all(
    tf.linalg.matvec(tf.linalg.diag([1.0,2.0,3.0]), tf.ones((3,))) + tf.ones((3,)) == tf.constant([2.0, 3.0, 4.0], dtype=tf.float32)
)

norm_rvs(1, tf.zeros((1,)), tf.eye(1))

x0 = norm_rvs(nT, tf.zeros((nT,)), tf.eye(nT)) 
y0 = norm_rvs(nT, x0, tf.eye(nT))

plt.plot(x0, linewidth=1, alpha=0.75) 
plt.plot(y0, linewidth=1, alpha=0.75) 
plt.show() 
 






from functions import LGSSM

v       = tf.random.normal((nD,), 0, 1)
A       = tf.random.normal((nD,nD), 0, 1)
B       = tf.random.normal((nD,nD), 0, 1)
np.all( 
    A @ B @ tf.transpose(A) == tf.matmul( tf.matmul(A,B), A, transpose_b=True) 
)
np.all(
    v * A * tf.transpose(v) == tf.linalg.matvec(A, v) * v
)


X, Y = LGSSM(nT,nD)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, c=colors[i]) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75, c=colors[i], linestyle='dashed') 
plt.show() 


    




from functions import KalmanFilter



X_fil = KalmanFilter(Y,V=tf.eye(nD)*10,W=tf.eye(nD)*10)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, c=colors[i]) 
    plt.plot(X_fil[:,i], linewidth=1, alpha=0.75, c=colors[i], linestyle='dashed') 
plt.show() 










from functions import SVSSM

v2      = tf.random.normal((nD,), 0, 1)
np.all( 
    v * tf.eye(nD) * v == tf.linalg.diag(v**2) 
)
np.all(
    tf.linalg.matvec(B, v) * (tf.eye(nD)*v2) * tf.linalg.matvec(B, v) == tf.linalg.diag(tf.linalg.matvec(B, v) * v2 * tf.linalg.matvec(B, v))
) 


A       = tf.linalg.diag(tf.ones(nD)*0.98)
X, Y    = SVSSM(nT,nD,A=A)

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75)
    plt.show() 









from functions import ExtendedKalmanFilter

X_f     = KalmanFilter(Y,A=A,V=tf.eye(nD)*10,W=tf.eye(nD)*10)

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_f[:,i], linewidth=1, alpha=0.75, linestyle='dotted') 
    plt.show() 

X_fil   = ExtendedKalmanFilter(Y,A=A)

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_fil[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 







from functions import SigmaPoints, Scaling, SigmaWeights

A = tf.random.normal((nD,nD), 0, 1)
SP = SigmaPoints(nD, X[0,:], tf.eye(nD), tf.eye(nD), 1.0)

M = SP[:,:nD] @ tf.transpose(A)
M.shape
for i in range(2*nD+1):
    if np.all(
        np.round(M[i,:], decimals=5) == np.round(tf.linalg.matvec(A, SP[i,:nD]), decimals=5)
    ) == False:
        print("Mismatch at index ", i)
        print(M[i,:])
        print(tf.linalg.matvec(A, SP[i,:nD]) )
        
sw = SigmaWeights(nD)  
        

from functions import UnscentedKalmanFilter 

X_filt = UnscentedKalmanFilter(Y, A=A)
    
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_filt[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

    


tf.linalg.det(tf.eye(nD)*1e-5)