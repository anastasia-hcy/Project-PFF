#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/"
pathdat             = "C:/Users/anastasia/MyProjects/JPMorgan/data/"
pathfig             = "C:/Users/anastasia/MyProjects/JPMorgan/Docs/"

import os, sys
os.chdir(path)
cwd = os.getcwd()
print(f"Current working directory is: {cwd}")
sys.path.append(cwd)

############# 
# Libraries #
#############

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

import psutil
import time
import pickle as pkl 

def get_current_process_ram_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mib = memory_info.rss / (1024 ** 2)  # Convert to MiB
    vms_mib = memory_info.vms / (1024 ** 2)  # Convert to MiB
    print(f"Current process RAM usage (RSS): {rss_mib:.2f} MiB")
    print(f"Current process RAM usage (VMS): {vms_mib:.2f} MiB")

get_current_process_ram_usage()















######################## 
# Question 1 - Warm up #
########################

nT              = 100
nD              = 4
Np              = 100
Nl              = 30




"""
Data from linear Gaussian state space model
"""
from functions import LGSSM

X1, Y1          = LGSSM(nT,nD)

"""
Data from stochastic volatility model
"""
# from functions import SVSSM
# from functions import SE_Cov_div

A               = tf.linalg.diag(tf.constant(tf.linspace(0.5, 0.80, nD).numpy(), dtype=tf.float64))
# X, Y            = SVSSM(nT, nD, A=A, V=Cx)
    
with open(pathdat+"dataSV.pkl", 'rb') as file:
    dataSV = pkl.load(file)    
X               = dataSV['States']
Y               = dataSV['Obs']
Cx              = dataSV['Cov']



fig, ax = plt.subplots(1,2, figsize=(12,4))
for i in range(nD):
    ax[0].plot(X1[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[0].plot(Y1[:,i], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i in range(nD):
    ax[1].plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[1].plot(Y[:,i], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
plt.tight_layout()
plt.show()



"""
Standard Kalman Filter
"""
from functions import KalmanFilter 

start_cpu_time  = time.process_time()
initial_rss     = psutil.Process(os.getpid()).memory_info().rss

X_KF            = KalmanFilter(Y1)

final_rss       = psutil.Process(os.getpid()).memory_info().rss
memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

end_cpu_time    = time.process_time()
cpu_time_taken  = end_cpu_time - start_cpu_time

print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


"""
Extended Kalman Filter
"""

# from functions import ExtendedKalmanFilter 

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_EKF           = ExtendedKalmanFilter(Y, A=A)

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

# with open(pathdat+"res_EKF.pkl", "wb") as file:
#     pkl.dump({"res": X_EKF, "cpu": [cpu_time_taken, memory_increase_mib]}, file)
    
with open(pathdat+"res_EKF.pkl", 'rb') as file:
    res_EKF = pkl.load(file)
    
X_EKF           = res_EKF['res']

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 



"""
Unscented Kalman Filter
"""

# from functions import UnscentedKalmanFilter

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_UKF           = UnscentedKalmanFilter(Y, A=A)

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


# with open(pathdat+"res_UKF.pkl", "wb") as file:
#     pkl.dump({"res": X_UKF, "cpu": [cpu_time_taken, memory_increase_mib]}, file)
    
with open(pathdat+"res_UKF.pkl", 'rb') as file:
    res_UKF = pkl.load(file)
    
X_UKF           = res_UKF['res']

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_UKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    




fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    ax.plot(X1[:,i], linewidth=1, alpha=0.5, color="green") 
    ax.plot(X_KF[:,i], linewidth=1, alpha=0.5, color="red", linestyle='dashed') 
plt.tight_layout()
plt.savefig(pathfig+"KF.png")

fig, ax = plt.subplots(1,2, figsize=(12,4))
for i in range(nD):
    ax[0].plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[0].plot(X_EKF[:,i], linewidth=1, alpha=0.5, color="red", linestyle='dashed') 
for i in range(nD):
    ax[1].plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[1].plot(X_UKF[:,i], linewidth=1, alpha=0.5, color="red", linestyle='dashed') 
plt.tight_layout()
plt.savefig(pathfig+"KF_all.png")



"""
Standard Particle Filter
"""

# from functions import ParticleFilter

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_PF, ess_PF, weights_PF, particles_PF, particles2_PF = ParticleFilter(Y, A=A, N=Np)

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

# with open(pathdat+"res_PF.pkl", "wb") as file:
#     pkl.dump({"res": X_PF, 
#               "ess": ess_PF,
#               "weights": weights_PF,
#               "particles": particles_PF,
#               "particles2": particles2_PF,
#               "cpu": [cpu_time_taken, memory_increase_mib]}, file)

with open(pathdat+"res_PF.pkl", 'rb') as file:
    res_PF = pkl.load(file)
    
X_PF            = res_PF['res']
ess_PF          = res_PF['ess']
weights_PF      = res_PF['weights']
particles_PF    = res_PF['particles']
particles2_PF    = res_PF['particles2']



from scipy.stats import multivariate_normal
import numpy as np 

i = 20

a1 = np.linspace(-6, 3, 100)
a2 = np.linspace(-5.5, 6, 100)
bx, by = np.meshgrid(a1,a2)
pos = np.dstack((bx, by))
rv = multivariate_normal([X[i-1,0],X[i-1,1]], Cx[0:2,0:2])
bz = rv.pdf(pos)

fig, ax = plt.subplots(1,2, figsize=(12,4))
for i in range(nD):
    ax[0].plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[0].plot(X_PF[:,i], linewidth=1, alpha=0.5, color="red", linestyle='dashed') 
ax[1].contour(bx,by,bz,levels=10, alpha=0.5)
ax[1].scatter(particles_PF[i,:,0], particles_PF[i,:,1], color='black',alpha=0.5)
ax[1].scatter(particles2_PF[i,:,0], particles2_PF[i,:,1], color='red')
plt.tight_layout()
plt.savefig(pathfig+"PF.png")
plt.show()




fig, ax = plt.subplots(figsize=(6,4))

ax.plot(tf.reduce_mean((X - X_UKF)**2, axis=1), linewidth=1, alpha=0.5, color="green") 
ax.plot(tf.reduce_mean((X - X_PF)**2, axis=1), linewidth=1, alpha=0.5, color="red") 
plt.tight_layout()
plt.savefig(pathfig+"PF2.png")













##############
# Question 2 # 
##############


"""
EDH 
"""

# from functions import EDH

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_EDH, ess_EDH, weights_EDH, Jx_EDH, Jw_EDH = EDH(Y, A=A, N=Np, method='EKF')

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


# with open(pathdat+"res_EDH_EKF.pkl", "wb") as file:
#     pkl.dump({"res": X_EDH, 
#               "ess": ess_EDH,
#               "weights": weights_EDH,
#               "Jx": Jx_EDH,
#               "Jw": Jw_EDH,
#               "cpu": [cpu_time_taken, memory_increase_mib]}, file)


with open(pathdat+"res_EDH.pkl", 'rb') as file:
    res_EDH = pkl.load(file)
    
X_EDH           = res_EDH['res']
ess_EDH         = res_EDH['ess']
weights_EDH     = res_EDH['weights']
Jx_EDH          = res_EDH['Jx']
Jw_EDH          = res_EDH['Jw']


with open(pathdat+"res_EDH_EKF.pkl", 'rb') as file:
    res_EDH_EKF = pkl.load(file)
    
X_EDH_EKF           = res_EDH_EKF['res']
ess_EDH_EKF         = res_EDH_EKF['ess']
weights_EDH_EKF     = res_EDH_EKF['weights']
Jx_EDH_EKF          = res_EDH_EKF['Jx']
Jw_EDH_EKF          = res_EDH_EKF['Jw']

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_EDH_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 
 
"""
LEDH 
"""

# from functions import LEDH

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_LEDH, ess_LEDH, weights_LEDH, Jx_LEDH, Jw_LEDH = LEDH(Y, A=A, N=Np, method='EKF')

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


# with open(pathdat+"res_LEDH_EKF.pkl", "wb") as file:
#     pkl.dump({"res": X_LEDH, 
#               "ess": ess_LEDH,
#               "weights": weights_LEDH,
#               "Jx": Jx_LEDH,
#               "Jw": Jw_LEDH,
#               "cpu": [cpu_time_taken, memory_increase_mib]}, file)


with open(pathdat+"res_LEDH.pkl", 'rb') as file:
    res_LEDH = pkl.load(file)
    
X_LEDH          = res_LEDH['res']
ess_LEDH        = res_LEDH['ess']
weights_LEDH    = res_LEDH['weights']
Jx_LEDH         = res_LEDH['Jx']
Jw_LEDH         = res_LEDH['Jw']


with open(pathdat+"res_LEDH_EKF.pkl", 'rb') as file:
    res_LEDH_EKF = pkl.load(file)
    
X_LEDH_EKF          = res_LEDH_EKF['res']
ess_LEDH_EKF        = res_LEDH_EKF['ess']
weights_LEDH_EKF    = res_LEDH_EKF['weights']
Jx_LEDH_EKF         = res_LEDH_EKF['Jx']
Jw_LEDH_EKF         = res_LEDH_EKF['Jw']

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_LEDH_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
    
    
    
    
    
    
    

fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].plot(tf.reduce_mean((X - X_EDH_EKF)**2, axis=1), linewidth=1, alpha=0.5, color="green") 
ax[0].plot(tf.reduce_mean((X - X_LEDH_EKF)**2, axis=1), linewidth=1, alpha=0.5, color="red") 

    
ax[1].plot(tf.reduce_mean((X - X_EDH)**2, axis=1), linewidth=1, alpha=0.5, color="green") 
ax[1].plot(tf.reduce_mean((X - X_LEDH)**2, axis=1), linewidth=1, alpha=0.5, color="red") 

plt.tight_layout()
plt.savefig(pathfig+"EDH.png")

plt.show()
    
    
    
    
    
    
    
    
    
    
    
"""
KPFF
"""
import numpy as np
from functions import SE_Cov_div, SVSSM, KernelPFF

nD              = 10
nT              = 80
Ny              = nD - 2

unobserved      = [3,7]
observed        = [0,1,2,4,5,6,8,9]

# A_sparse        = tf.linalg.diag(tf.constant(tf.linspace(0.5, 0.80, nD).numpy(), dtype=tf.float64))
# B_sparse        = tf.Variable(tf.zeros((Ny,nD), dtype=tf.float64))
# for i,j in zip(range(Ny),observed):
#     B_sparse[i,j].assign(1.0)
    
# Cx_sparse, DivCx_sparse = SE_Cov_div(nD, tf.random.normal((nD,), dtype=tf.float64))
# X_sparse, Y_sparse = SVSSM(nT, nD, n_sparse=Ny, A=A_sparse, B=B_sparse, V=Cx_sparse)

with open(pathdat+"dataSV_sparse.pkl", 'rb') as file:
    dataSV_sparse = pkl.load(file)    
X_sparse        = dataSV_sparse['States']
Y_sparse        = dataSV_sparse['Obs']
A_sparse        = dataSV_sparse['A']
B_sparse        = dataSV_sparse['B']
Cx_sparse       = dataSV_sparse['Cx']

for i,j in zip(observed,range(Ny)):
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y_sparse[:,j], linewidth=1, alpha=0.75)
    plt.show()

Np              = 20
Nl              = 30

"""
KPFF - scalar
"""

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_KPFF2, Jx_KPFF2, Jw_KPFF2, particles_KPFF2, particles2_KPFF2 = KernelPFF(Y_sparse, Nx=nD, A=A_sparse, B=B_sparse, N=Np, method="scalar")

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


# with open(pathdat+"res_KPFF2_sparse.pkl", "wb") as file:
#     pkl.dump({"res": X_KPFF2, 
#               "Jx": Jx_KPFF2,
#               "Jw": Jw_KPFF2,
#               "particles": particles_KPFF2,
#               "particles2": particles2_KPFF2,
#               "cpu": [cpu_time_taken, memory_increase_mib]}, file)


with open(pathdat+"res_KPFF2_sparse.pkl", 'rb') as file:
    res_KPFF2 = pkl.load(file)
    
X_KPFF2          = res_KPFF2['res']
Jx_KPFF2         = res_KPFF2['Jx']
Jw_KPFF2         = res_KPFF2['Jw']
particles_KPFF2  = res_KPFF2['particles']
particles2_KPFF2 = res_KPFF2['particles2']


for i in range(nD):
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    



"""
KPFF - kernel
"""

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss
    
# X_KPFF, Jx_KPFF, Jw_KPFF, particles_KPFF, particles2_KPFF = KernelPFF(Y_sparse, Nx=nD, A=A_sparse, B=B_sparse, N=Np)

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


# with open(pathdat+"res_KPFF_sparse.pkl", "wb") as file:
#     pkl.dump({"res": X_KPFF, 
#               "Jx": Jx_KPFF,
#               "Jw": Jw_KPFF,
#               "particles": particles_KPFF,
#               "particles2": particles2_KPFF,
#               "cpu": [cpu_time_taken, memory_increase_mib]}, file)


with open(pathdat+"res_KPFF_sparse.pkl", 'rb') as file:
    res_KPFF = pkl.load(file)
    
X_KPFF          = res_KPFF['res']
Jx_KPFF         = res_KPFF['Jx']
Jw_KPFF         = res_KPFF['Jw']
particles_KPFF  = res_KPFF['particles']
particles2_KPFF = res_KPFF['particles2']

for i in unobserved:
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_KPFF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

 
 
 
plt.scatter(particles_KPFF2[10,:,3],  particles_KPFF2[10,:,4])
plt.scatter(particles2_KPFF2[10,:,3],  particles2_KPFF2[10,:,4])
plt.show() 
 
 
plt.scatter(particles_KPFF[10,:,7],  particles_KPFF[10,:,8])
plt.scatter(particles2_KPFF[10,:,7],  particles2_KPFF[10,:,8])
plt.show() 
 
   

 
 
 
 
 
 

#############################
# Plots, figures and tables # 
#############################

import numpy as np


def conditioning(J):
    """Compute the conditioning number of a given Jacobian matrix using the 2-norm"""
    norm        = tf.norm(J, ord=2)
    Jinv        = tf.linalg.inv(J)
    norminv     = tf.norm(Jinv, ord=2)
    return norm * norminv

def conditioning_asym(J):
    """Compute the conditioning number of a given assymetric Jacobian matrix using the eigenvalues"""
    _, s, _ = np.linalg.svd( J )
    return np.max(s) / np.min(s)

JxConds_KPFF_sparse = np.zeros((nT,Nl,Np))  
for i in range(nT):
    for j in range(Nl): 
        for k in range(Np): 
            JxConds_KPFF_sparse[i,j,k] = conditioning_asym(Jx_KPFF[i,j,k,:,:])

JwConds_KPFF_sparse = np.zeros((nT,Nl,Np))  
for i in range(nT):
    for j in range(Nl): 
        for k in range(Np): 
            JwConds_KPFF_sparse[i,j,k] = conditioning_asym(Jw_KPFF[i,j,k,:,:])
            
JxConds_KPFF2_sparse = np.zeros((nT,Nl,Np))  
for i in range(nT):
    for j in range(Nl): 
        for k in range(Np): 
            JxConds_KPFF2_sparse[i,j,k] = conditioning_asym(Jx_KPFF2[i,j,k,:,:])

JwConds_KPFF2_sparse = np.zeros((nT,Nl,Np))  
for i in range(nT):
    for j in range(Nl): 
        for k in range(Np): 
            JwConds_KPFF2_sparse[i,j,k] = conditioning_asym(Jw_KPFF2[i,j,k,:,:])


JxConds_KPFF_sparse2 = tf.where( tf.math.logical_and(tf.math.is_inf(JxConds_KPFF_sparse), JxConds_KPFF_sparse > 0.0) , tf.cast(1e9, tf.float64), JxConds_KPFF_sparse)
JxConds_KPFF2_sparse2 = tf.where( tf.math.logical_and(tf.math.is_inf(JxConds_KPFF2_sparse), JxConds_KPFF2_sparse > 0.0) , tf.cast(1e9, tf.float64), JxConds_KPFF2_sparse)


fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].plot(tf.reduce_mean(JxConds_KPFF_sparse2, axis=(1,2)), linewidth=1, alpha=0.5, color="green") 
ax[0].plot(tf.reduce_mean(JxConds_KPFF2_sparse2, axis=(1,2)), linewidth=1, alpha=0.5, color="red") 

    
ax[1].plot(tf.reduce_mean(JwConds_KPFF_sparse, axis=(1,2)), linewidth=1, alpha=0.5, color="green") 
ax[1].plot(tf.reduce_mean(JwConds_KPFF2_sparse, axis=(1,2)), linewidth=1, alpha=0.5, color="red") 

plt.tight_layout()
plt.savefig(pathfig+"KPFF_J.png")

plt.show()



# with open(pathdat+"J_conds.pkl", "wb") as file:
#     pkl.dump({"Jx_EDH_EKF": JxConds_EDH_EKF, 
#               "Jw_EDH_EKF": JwConds_EDH_EKF,
#               "Jx_LEDH_EKF": JxConds_LEDH_EKF,
#               "Jw_LEDH_EKF": JwConds_LEDH_EKF,
#               "Jw_EDH": JwConds_EDH, 
#               "Jw_LEDH": JwConds_LEDH,
#               "Jx_KPFF2_sparse": JxConds_KPFF2_sparse,
#               "Jw_KPFF2_sparse": JwConds_KPFF2_sparse
#               "Jx_KPFF2": JxConds_KPFF2,
#               "Jw_KPFF2": JwConds_KPFF2}, file)

with open(pathdat+"J_conds.pkl", 'rb') as file:
    J_Conds = pkl.load(file)

JxConds_EDH_EKF     = J_Conds["Jx_EDH_EKF"]
JwConds_EDH_EKF     = J_Conds["Jw_EDH_EKF"]

JxConds_LEDH_EKF    = J_Conds["Jx_LEDH_EKF"]
JwConds_LEDH_EKF    = J_Conds["Jw_LEDH_EKF"]

JwConds_EDH         = J_Conds["Jw_EDH"]
JwConds_LEDH        = J_Conds["Jw_LEDH"]

JxConds_KPFF_sparse = J_Conds["Jx_KPFF_sparse"]
JwConds_KPFF_sparse = J_Conds["Jw_KPFF_sparse"]

JxConds_KPFF2       = J_Conds["Jx_KPFF2"]
JwConds_KPFF2       = J_Conds["Jw_KPFF2"]

JxConds_KPFF2_sparse = J_Conds["Jx_KPFF2_sparse"]
JwConds_KPFF2_sparse = J_Conds["Jw_KPFF2_sparse"]

    

fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].plot(JxConds_EDH_EKF, linewidth=1, alpha=0.5, color="green") 
ax[0].plot(tf.reduce_mean(JxConds_LEDH_EKF, axis=1), linewidth=1, alpha=0.5, color="red") 

    
ax[1].plot(JwConds_EDH_EKF, linewidth=1, alpha=0.5, color="green") 
ax[1].plot(tf.reduce_mean(JwConds_LEDH_EKF, axis=1),  linewidth=1, alpha=0.5, color="red") 

plt.tight_layout()
plt.savefig(pathfig+"EDH_J_EKF.png")

plt.show()


fig, ax = plt.subplots(figsize=(6,4))

ax.plot(JwConds_EDH, linewidth=1, alpha=0.5, color="green") 
ax.plot(tf.reduce_mean(JwConds_LEDH, axis=1), linewidth=1, alpha=0.5, color="red") 

    
plt.tight_layout()
plt.savefig(pathfig+"EDH_J_UKF.png")

plt.show()




fig, ax = plt.subplots(figsize=(12,4))
ax.plot(np.repeat(1,10), np.arange(10), alpha=0.75, color=colors[0])
ax.plot(np.repeat(2,10), np.arange(10), alpha=0.75, color=colors[1])
ax.plot(np.repeat(3,10), np.arange(10), alpha=0.75, color=colors[2])
ax.plot(np.repeat(4,10), np.arange(10),  alpha=0.75, color=colors[3])
ax.plot(np.repeat(5,10), np.arange(10), alpha=0.75, color=colors[4])
ax.plot(np.repeat(6,10), np.arange(10),  alpha=0.75, color=colors[5])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12,4))
ax.plot(JxConds_EDH_EKF, alpha=0.75, color=colors[2])
ax.plot(JxConds_LEDH_EKF.mean(axis=1), alpha=0.75, color=colors[3])
ax.plot(JxConds_KPFF2.mean(axis=(1,2)), alpha=0.75, color=colors[4])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12,4))
ax.plot(JwConds_EDH, alpha=0.75, color=colors[2])
ax.plot(JwConds_LEDH.mean(axis=1), alpha=0.75, color=colors[3])
ax.plot(JwConds_KPFF2.mean(axis=(1,2)), alpha=0.75, color=colors[4])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12,4))
ax.plot(JxConds_KPFF2.mean(axis=(1,2)), alpha=0.75, color=colors[4])
plt.tight_layout()
plt.show()







from itertools import permutations

PermSets            = list(permutations(np.arange(nD).tolist()))
PermList            = [list(l) for l in PermSets]
def compute_OMT(x, xhat, n=nD, nPerm=len(PermList), Perm=PermList):
    """Compute the OMAT error of between two vectors."""
    dis             = np.zeros((nPerm,))
    for i in range(nPerm):
        idx         = Perm[i]
        dis[i]      = np.sqrt(np.sum((x - xhat[idx])**2))
    return np.min(dis) / n

dis_EKF   = np.zeros((nT,))
dis_UKF   = np.zeros((nT,))
dis_PF    = np.zeros((nT,))
dis_EDH   = np.zeros((nT,))
dis_LEDH  = np.zeros((nT,))
dis_KPFF2 = np.zeros((nT,))

for i in range(nT):
    dis_EKF[i] = compute_OMT(X_EKF[i,:].numpy(), X[i,:].numpy())
    dis_UKF[i] = compute_OMT(X_UKF[i,:].numpy(), X[i,:].numpy())
    dis_PF[i] = compute_OMT(X_PF[i,:].numpy(), X[i,:].numpy())
    dis_EDH[i] = compute_OMT(X_EDH[i,:].numpy(), X[i,:].numpy())
    dis_LEDH[i] = compute_OMT(X_LEDH[i,:].numpy(), X[i,:].numpy())
    dis_KPFF2[i] = compute_OMT(X_KPFF2[i,:].numpy(), X[i,:].numpy())

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(dis_UKF, alpha=0.75, color=colors[0])
ax.plot(dis_PF, alpha=0.75, color=colors[1])
ax.plot(dis_EDH, alpha=0.75, color=colors[2])
ax.plot(dis_LEDH, alpha=0.75, color=colors[3])
ax.plot(dis_KPFF2, alpha=0.75, color=colors[4])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(dis_EKF, alpha=0.75, color=colors[5])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12,4))
ax.plot( tf.reduce_mean((X_UKF - X)**2, axis=1), alpha=0.75, color=colors[0])
ax.plot( tf.reduce_mean((X_PF - X)**2, axis=1), alpha=0.75, color=colors[1])
ax.plot( tf.reduce_mean((X_EDH - X)**2, axis=1), alpha=0.75, color=colors[2])
ax.plot( tf.reduce_mean((X_LEDH - X)**2, axis=1), alpha=0.75, color=colors[3])
ax.plot( tf.reduce_mean((X_KPFF2 - X)**2, axis=1), alpha=0.75, color=colors[4])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12,4))
ax.plot( tf.reduce_mean((X_EKF - X)**2, axis=1), alpha=0.75, color=colors[5])
plt.tight_layout()
plt.show()




i = 1
fig, ax = plt.subplots(figsize=(12,4))
# ax.plot( (X_UKF[:,i] - X[:,i])**2, alpha=0.75, color=colors[0])
ax.plot( (X_PF[:,i] - X[:,i])**2, alpha=0.75, color=colors[1])
ax.plot( (X_EDH_EKF[:,i] - X[:,i])**2, alpha=0.75, color=colors[2])
ax.plot( (X_LEDH_EKF[:,i] - X[:,i])**2, alpha=0.75, color=colors[3])
ax.plot( (X_KPFF2[:,i] - X[:,i])**2, alpha=0.75, color=colors[4])
plt.tight_layout()
plt.show()

