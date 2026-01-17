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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

import pickle as pkl 
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from scipy.stats import multivariate_normal
import numpy as np 

import psutil
import time

def get_current_process_ram_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mib = memory_info.rss / (1024 ** 2)  # Convert to MiB
    vms_mib = memory_info.vms / (1024 ** 2)  # Convert to MiB
    print(f"Current process RAM usage (RSS): {rss_mib:.2f} MiB")
    print(f"Current process RAM usage (VMS): {vms_mib:.2f} MiB")

get_current_process_ram_usage()


##################### 
# Simulate datasets #
#####################

nT              = 100
nD              = 4
nX              = 10
Ny              = nX - 2
unobserved      = [3,7]
observed        = [0,1,2,4,5,6,8,9]

"""
Simulate non-sparse data 
"""

# from scripts import SSM, SE_Cov_div

# X1, Y1          = SSM(nT, nD)

# A               = tf.linalg.diag(tf.constant(tf.linspace(0.5, 0.85, nD).numpy(), dtype=tf.float64))
# Cx, _           = SE_Cov_div(nD, tf.random.normal((nD,), dtype=tf.float64))
# X, Y            = SSM(nT, nD, model="SV", A=A, V=Cx)


"""
Simulate sparse data 
"""

# A_sparse        = tf.linalg.diag(tf.constant(tf.linspace(0.5, 0.85, nX).numpy(), dtype=tf.float64))
# B_sparse        = tf.Variable(tf.zeros((Ny,nX), dtype=tf.float64))
# for i,j in zip(range(Ny),observed):
#     B_sparse[i,j].assign(1.0)
# Cx_sparse, _    = SE_Cov_div(nX, tf.random.normal((nX,), dtype=tf.float64))

# X_sparse, Y_sparse    = SSM(nT, nX, model="SV", n_sparse=Ny, A=A_sparse, B=B_sparse, V=Cx_sparse)
# X1_sparse, Y1_sparse  = SSM(nT, nX, n_sparse=Ny,  B=B_sparse)

"""
Save simulated data
"""

# dat = {'LG_States': X1, 'LG_Obs': Y1, 
#        'SV_States': X, 'SV_Obs': Y, 
#        'A': A, 'Cx': Cx, 
#        'sparse_LG_States': X1_sparse, 'sparse_LG_Obs': Y1_sparse,
#        'sparse_States': X_sparse, 'sparse_Obs': Y_sparse, 
#        'sparse_A': A_sparse, 'sparse_B': B_sparse, 'sparse_Cx': Cx_sparse}
# with open(pathdat+"data_sim.pkl", 'wb') as file:
#     pkl.dump(dat, file)    



"""
Load simulated data
"""

with open(pathdat+"data_sim.pkl", 'rb') as file:
    data        = pkl.load(file)    
X1              = data['LG_States']
Y1              = data['LG_Obs']
X               = data['SV_States']
Y               = data['SV_Obs']
A               = data['A']
Cx              = data['Cx']
X1_sparse       = data['sparse_LG_States']
Y1_sparse       = data['sparse_LG_Obs']
X_sparse        = data['sparse_States']
Y_sparse        = data['sparse_Obs']
A_sparse        = data['sparse_A']
B_sparse        = data['sparse_B']
Cx_sparse       = data['sparse_Cx']


"""
Plots simulated data
"""

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.5, color="green") 
    plt.plot(Y1[:,i], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
plt.show()

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    plt.plot(Y[:,i], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
plt.show()

fig, ax = plt.subplots(figsize=(6,4))
for i,j in zip(observed,range(Ny)):
    plt.plot(X1_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
    plt.plot(Y1_sparse[:,j], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i in unobserved:
    plt.plot(X1_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
plt.show()


fig, ax = plt.subplots(figsize=(6,4))
for i,j in zip(observed,range(Ny)):
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
    plt.plot(Y_sparse[:,j], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i in unobserved:
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
plt.show()


fig, ax = plt.subplots(1,4, figsize=(24,4))
for i in range(nD):
    ax[0].plot(X1[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[0].plot(Y1[:,i], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i in range(nD):
    ax[1].plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[1].plot(Y[:,i], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i,j in zip(observed,range(Ny)):
    ax[2].plot(X1_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[2].plot(Y1_sparse[:,j], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i in unobserved:
    ax[2].plot(X1_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
for i,j in zip(observed,range(Ny)):
    ax[3].plot(X_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[3].plot(Y_sparse[:,j], linewidth=1, alpha=0.5, color="orange", linestyle='dashed') 
for i in unobserved:
    ax[3].plot(X_sparse[:,i], linewidth=1, alpha=0.5, color="green") 
plt.tight_layout()
plt.show()

    
########################## 
# Standard Kalman Filter #
##########################

from scripts import KalmanFilter

start_cpu_time  = time.process_time()
initial_rss     = psutil.Process(os.getpid()).memory_info().rss

X_KF            = KalmanFilter(Y1)

final_rss       = psutil.Process(os.getpid()).memory_info().rss
memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

end_cpu_time    = time.process_time()
cpu_time_taken  = end_cpu_time - start_cpu_time

print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_KF[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 





########################## 
# Extended Kalman Filter #
##########################

# from scripts import ExtendedKalmanFilter

"""
EKF - LG
"""
# X1_EKF          = ExtendedKalmanFilter(Y1)


"""
EKF - SV
"""
# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_EKF           = ExtendedKalmanFilter(Y, V=Cx, model="SV")

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

"""
EKF - Results
"""

# with open(pathdat+"res_EKF.pkl", "wb") as file:
#     pkl.dump({"res": X_EKF, 
#               "cpu": [cpu_time_taken, memory_increase_mib],
#               "res_testLG": X1_EKF}, 
#               file)
    
with open(pathdat+"res_EKF.pkl", 'rb') as file:
    res_EKF = pkl.load(file)
X_EKF           = res_EKF['res']
X1_EKF          = res_EKF['res_testLG']


fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X1_EKF[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_EKF[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 




########################### 
# Unscented Kalman Filter #
###########################

# from scripts import UnscentedKalmanFilter

"""
UKF - LG
"""
# X1_UKF          = UnscentedKalmanFilter(Y1)

"""
UKF - SV
"""
# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_UKF           = UnscentedKalmanFilter(Y, V=Cx, model="SV")

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

"""
UKF - Results
"""

# with open(pathdat+"res_UKF.pkl", "wb") as file:
#     pkl.dump({"res": X_UKF, 
#               "cpu": [cpu_time_taken, memory_increase_mib],
#               "res_testLG": X1_UKF}, file)

with open(pathdat+"res_UKF.pkl", 'rb') as file:
    res_UKF = pkl.load(file)
X_UKF           = res_UKF['res']
X1_UKF          = res_UKF['res_testLG']


fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_UKF[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X1_UKF[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 




############################ 
# Standard Particle Filter #
############################ 

from scripts import ParticleFilter
Np              = 100

X_PF_S_1, ess_PF_S_1, weights_PF_S_1, particles_PF_S_1, particles2_PF_S_1 = ParticleFilter(Y1, N=Np, resample="Soft")
# X_PF_S2_1, ess_PF_S2_1, weights_PF_S2_1, particles_PF_S2_1, particles2_PF_S2_1 = ParticleFilter(Y1, N=Np, resample="Soft", backpropagation=True)

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_PF1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='orange') 
    plt.plot(X_PF_S_1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
    # plt.plot(X_PF_S2_1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 


X_PF_OT_1, ess_PF_OT_1, weights_PF_OT_1, particles_PF_OT_1, particles2_PF_OT_1 = ParticleFilter(Y1, N=Np, resample="OT")

                  
fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_PF_OT_1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='orange')     
plt.show() 
                                                                                                                    

"""
PF - LG
"""

# X_PF1, ess_PF1, weights_PF1, particles_PF1, particles2_PF1 = ParticleFilter(Y1, N=Np)

"""
PF - SV
"""

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_PF, ess_PF, weights_PF, particles_PF, particles2_PF = ParticleFilter(Y, V=Cx, model="SV", N=Np)

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


"""
PF - Results
"""

# with open(pathdat+"res_PF.pkl", "wb") as file:
#     pkl.dump({"res": X_PF, 
#               "ess": ess_PF,
#               "weights": weights_PF,
#               "particles": particles_PF,
#               "particles2": particles2_PF,
#               "cpu": [cpu_time_taken, memory_increase_mib],
#               "res_testLG": X_PF1, 
#               "ess_testLG": ess_PF1,
#               "weights_testLG": weights_PF1,
#               "particles_testLG": particles_PF1,
#               "particles2_testLG": particles2_PF1}, file)

with open(pathdat+"res_PF.pkl", 'rb') as file:
    res_PF = pkl.load(file)

X_PF            = res_PF['res']
ess_PF          = res_PF['ess']
weights_PF      = res_PF['weights']
particles_PF    = res_PF['particles']
particles2_PF   = res_PF['particles2']

X_PF1           = res_PF['res_testLG']
ess_PF1         = res_PF['ess_testLG']
weights_PF1     = res_PF['weights_testLG']
particles_PF1   = res_PF['particles_testLG']
particles2_PF1  = res_PF['particles2_testLG']


fig, ax = plt.subplots(figsize=(6,4)) 
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_PF[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
    plt.show() 

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_PF1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
plt.show() 


j = 26

a1 = np.linspace(-6,4, 100)
a2 = np.linspace(-8,5, 100)
bx, by = np.meshgrid(a1,a2)
pos = np.dstack((bx, by))
rv = multivariate_normal([X_PF[j-1,0], X_PF[j-1,1]], Cx[0:2,0:2])
bz = rv.pdf(pos)

fig, ax = plt.subplots(figsize=(6,4))
plt.contour(bx,by,bz,levels=10, alpha=0.5)
plt.scatter(particles_PF[j,:,0], particles_PF[j,:,1], color='black',alpha=0.5)
plt.scatter(particles2_PF[j,:,0], particles2_PF[j,:,1], color='red')
plt.show()




fig, ax = plt.subplots(1,2, figsize=(12,4))
for i in range(nD):
    ax[0].plot(X[:,i], linewidth=1, alpha=0.5, color="green") 
    ax[0].plot(X_PF[:,i], linewidth=1, alpha=0.5, color="red", linestyle='dashed') 

ax[1].contour(bx,by,bz,levels=10, alpha=0.5)
ax[1].scatter(particles_PF[j,:,0], particles_PF[j,:,1], color='black',alpha=0.5)
ax[1].scatter(particles2_PF[j,:,0], particles2_PF[j,:,1], color='red')

plt.tight_layout()
# plt.savefig(pathfig+"PF.png")
plt.show()








#######
# EDH #
#######

# from scripts import EDH
# Np              = 100 

"""
EDH - LG 
"""

# X_EDH_EKF_1, ess_EDH_EKF_1, weights_EDH_EKF_1, Jx_EDH_EKF_1, Jw_EDH_EKF_1 = EDH(Y1, N=Np, method='EKF')
# X_EDH_1, ess_EDH_1, weights_EDH_1, Jx_EDH_1, Jw_EDH_1 = EDH(Y1, N=Np, method='UKF')

"""
EDH - SV 
"""

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_EDH_EKF, ess_EDH_EKF, weights_EDH_EKF, Jx_EDH_EKF, Jw_EDH_EKF = EDH(Y, V=Cx, model="SV", N=Np, method='EKF')

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_EDH, ess_EDH, weights_EDH, Jx_EDH, Jw_EDH = EDH(Y, V=Cx, model="SV", N=Np, method='UKF')

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib_EKF = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken_EKF = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib_EKF:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken_EKF:.3f} seconds")

with open(pathdat+"res_EDH.pkl", 'rb') as file:
    res_EDH = pkl.load(file)
    
X_EDH           = res_EDH['res']
ess_EDH         = res_EDH['ess']
weights_EDH     = res_EDH['weights']
Jx_EDH          = res_EDH['Jx']
Jw_EDH          = res_EDH['Jw']

X_EDH_EKF       = res_EDH['res_EKF']
ess_EDH_EKF     = res_EDH['ess_EKF']
weights_EDH_EKF = res_EDH['weights_EKF']
Jx_EDH_EKF      = res_EDH['Jx_EKF']
Jw_EDH_EKF      = res_EDH['Jw_EKF']

X_EDH_1        = res_EDH['res_testLG']
ess_EDH_1       = res_EDH['ess_testLG']
weights_EDH_1   = res_EDH['weights_testLG']
Jx_EDH_1        = res_EDH['Jx_testLG']
Jw_EDH_1        = res_EDH['Jw_testLG']

X_EDH_EKF_1     = res_EDH['res_testLG_EKF']
ess_EDH_EKF_1   = res_EDH['ess_testLG_EKF']
weights_EDH_EKF_1 = res_EDH['weights_testLG_EKF']
Jx_EDH_EKF_1    = res_EDH['Jx_testLG_EKF']
Jw_EDH_EKF_1    = res_EDH['Jw_testLG_EKF']

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_EDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red') 
    plt.plot(X_EDH_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange') 
    plt.show() 


for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_EDH_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red') 
    plt.plot(X_EDH_EKF_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange') 
plt.show() 
 
 
 

 
#######
# SDE #    
#######

Np              = 50

with open(pathdat+"res_SDE.pkl", 'rb') as file:
    res_SDE = pkl.load(file)

X_SDE           = res_SDE['res']
Cond_SDE        = res_SDE['cond']
stiff_SDE       = res_SDE['stiff']
beta_SDE        = res_SDE['beta']


with open(pathdat+"res_SDE_LG.pkl", 'rb') as file:
    res_SDE = pkl.load(file)

X_SDE_1         = res_SDE['res']
Cond_SDE_1      = res_SDE['cond']
stiff_SDE_1     = res_SDE['stiff']
beta_SDE_1      = res_SDE['beta']


with open(pathdat+"res_SDE_homo.pkl", 'rb') as file:
    res_SDE = pkl.load(file)

X_SDE_homo       = res_SDE['res']
Cond_SDE_homo   = res_SDE['cond']
stiff_SDE_homo  = res_SDE['stiff']
beta_SDE_homo   = res_SDE['beta']


with open(pathdat+"res_SDE_LG_homo.pkl", 'rb') as file:
    res_SDE = pkl.load(file)

X_SDE_1_homo    = res_SDE['res']
Cond_SDE_1_homo = res_SDE['cond']
stiff_SDE_1_homo= res_SDE['stiff']
beta_SDE_1_homo = res_SDE['beta']

for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_SDE_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red') 
    # plt.plot(X_SDE_1_homo[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange') 
plt.show() 

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_SDE[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red') 
    plt.plot(X_SDE_homo[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange') 
plt.show() 
    


########
# LEDH #
########

Np = 100


with open(pathdat+"res_LEDH.pkl", 'rb') as file:
    res_LEDH = pkl.load(file)
    
X_LEDH            = res_LEDH['res']
ess_LEDH          = res_LEDH['ess']
weights_LEDH      = res_LEDH['weights']
Jx_LEDH           = res_LEDH['Jx']
Jw_LEDH           = res_LEDH['Jw']

X_LEDH_EKF        = res_LEDH['res_EKF']
ess_LEDH_EKF      = res_LEDH['ess_EKF']
weights_LEDH_EKF  = res_LEDH['weights_EKF']
Jx_LEDH_EKF       = res_LEDH['Jx_EKF']
Jw_LEDH_EKF       = res_LEDH['Jw_EKF']

X_LEDH_1          = res_LEDH['res_testLG']
ess_LEDH_1        = res_LEDH['ess_testLG']
weights_LEDH_1    = res_LEDH['weights_testLG']
Jx_LEDH_1         = res_LEDH['Jx_testLG']
Jw_LEDH_1         = res_LEDH['Jw_testLG']

X_LEDH_EKF_1       = res_LEDH['res_testLG_EKF']
ess_LEDH_EKF_1     = res_LEDH['ess_testLG_EKF']
weights_LEDH_EKF_1 = res_LEDH['weights_testLG_EKF']
Jx_LEDH_EKF_1      = res_LEDH['Jx_testLG_EKF']
Jw_LEDH_EKF_1      = res_LEDH['Jw_testLG_EKF']


with open(pathdat+"res_LEDH_SDE_EKF_LG.pkl", 'rb') as file:
    res_LEDH = pkl.load(file)
    
X_LEDH_SDE_EKF_1       = res_LEDH['res']
ess_LEDH_SDE_EKF_1     = res_LEDH['ess']
weights_LEDH_SDE_EKF_1 = res_LEDH['weights']
Jx_LEDH_SDE_EKF_1      = res_LEDH['Jx']
Jw_LEDH_SDE_EKF_1      = res_LEDH['Jw']

with open(pathdat+"res_LEDH_SDE_UKF_LG.pkl", 'rb') as file:
    res_LEDH = pkl.load(file)
    
X_LEDH_SDE_1       = res_LEDH['res']
ess_LEDH_SDE_1     = res_LEDH['ess']
weights_LEDH_EKF_1 = res_LEDH['weights']
Jx_LEDH_SDE_1      = res_LEDH['Jx']
Jw_LEDH_SDE_1      = res_LEDH['Jw']

with open(pathdat+"res_LEDH_SDE_EKF.pkl", 'rb') as file:
    res_LEDH = pkl.load(file)
    
X_LEDH_SDE_EKF       = res_LEDH['res']
ess_LEDH_SDE_EKF     = res_LEDH['ess']
weights_LEDH_SDE_EKF = res_LEDH['weights']
Jx_LEDH_SDE_EKF      = res_LEDH['Jx']
Jw_LEDH_SDE_EKF      = res_LEDH['Jw']

with open(pathdat+"res_LEDH_SDE_UKF.pkl", 'rb') as file:
    res_LEDH = pkl.load(file)
    
X_LEDH_SDE          = res_LEDH['res']
ess_LEDH_SDE        = res_LEDH['ess']
weights_LEDH_SDE    = res_LEDH['weights']
Jx_LEDH_SDE         = res_LEDH['Jx']
Jw_LEDH_SDE         = res_LEDH['Jw']


for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_LEDH[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red') 
    plt.plot(X_LEDH_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange') 
plt.show() 

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75, color='green') 
    # plt.plot(X_LEDH_SDE[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='blue')
    plt.plot(X_LEDH_SDE_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='purple')
plt.show() 
    

for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_LEDH_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red') 
    plt.plot(X_LEDH_EKF_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange') 
plt.show() 

for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    plt.plot(X_LEDH_SDE_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='red')
    plt.plot(X_LEDH_SDE_EKF_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed', color='orange')
plt.show() 


fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(tf.reduce_mean((X - X_EDH_EKF)**2, axis=1), linewidth=1, alpha=0.5, color="green") 
ax[0].plot(tf.reduce_mean((X - X_LEDH_EKF)**2, axis=1), linewidth=1, alpha=0.5, color="red") 
ax[1].plot(tf.reduce_mean((X - X_EDH)**2, axis=1), linewidth=1, alpha=0.5, color="green") 
ax[1].plot(tf.reduce_mean((X - X_LEDH)**2, axis=1), linewidth=1, alpha=0.5, color="red") 
plt.tight_layout()
plt.show()
    
########
# KPFF #
########

# from scripts import KernelPFF
# Np              = 20
# Nl              = 30

"""
KPFF - LG
"""

# X_KPFF2_1, Jx_KPFF2_1, Jw_KPFF2_1, particles_KPFF2_1, particles2_KPFF2_1    = KernelPFF(Y1_sparse, Nx=nX, N=Np, B=B_sparse, method="scalar")
# X_KPFF_1, Jx_KPFF_1, Jw_KPFF_1, particles_KPFF_1, particles2_KPFF_1         = KernelPFF(Y1_sparse, Nx=nX, N=Np, B=B_sparse)

"""
KPFF - SV
"""

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss

# X_KPFF2, Jx_KPFF2, Jw_KPFF2, particles_KPFF2, particles2_KPFF2 = KernelPFF(Y_sparse, model="SV", Nx=nX, B=B_sparse, N=Np, Sigma0=tf.eye(nX, dtype=tf.float64), method="scalar")

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib2 = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken2 = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib2:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken2:.3f} seconds")

# start_cpu_time  = time.process_time()
# initial_rss     = psutil.Process(os.getpid()).memory_info().rss
    
# X_KPFF, Jx_KPFF, Jw_KPFF, particles_KPFF, particles2_KPFF = KernelPFF(Y_sparse, model="SV", Nx=nX, B=B_sparse, N=Np, Sigma0=tf.eye(nX, dtype=tf.float64))

# final_rss       = psutil.Process(os.getpid()).memory_info().rss
# memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

# end_cpu_time    = time.process_time()
# cpu_time_taken  = end_cpu_time - start_cpu_time

# print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
# print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


"""
KPFF - Results
"""

# with open(pathdat+"res_KPFF.pkl", "wb") as file:
#     pkl.dump({"res": X_KPFF,
#               "Jx": Jx_KPFF,
#               "Jw": Jw_KPFF, 
#               "particles": particles_KPFF, 
#               "particles2": particles2_KPFF,
#               "cpu": [cpu_time_taken, memory_increase_mib],

#               "res_2": X_KPFF2,
#               "Jx_2": Jx_KPFF2,
#               "Jw_2": Jw_KPFF2, 
#               "particles_2": particles_KPFF2, 
#               "particles2_2": particles2_KPFF2,
#               "cpu_2": [cpu_time_taken2, memory_increase_mib2],

#               "res_testLG": X_KPFF_1,
#               "Jx_testLG": Jx_KPFF_1, 
#               "Jw_testLG": Jw_KPFF_1, 
#               "particles_testLG": particles_KPFF_1,
#               "particles2_testLG": particles2_KPFF_1,  

#               "res_testLG2": X_KPFF2_1, 
#               "Jx_testLG2": Jx_KPFF2_1,
#               "Jw_testLG2": Jw_KPFF2_1,
#               "particles_testLG2": particles_KPFF2_1,
#               "particles2_testLG2": particles2_KPFF2_1}, file)


with open(pathdat+"res_KPFF.pkl", 'rb') as file:
    res_KPFF = pkl.load(file)
    
X_KPFF            = res_KPFF['res']
Jx_KPFF           = res_KPFF['Jx']
Jw_KPFF           = res_KPFF['Jw']
particles_KPFF    = res_KPFF['particles']
particles2_KPFF   = res_KPFF['particles2']

X_KPFF2           = res_KPFF['res_2']
Jx_KPFF2          = res_KPFF['Jx_2']
Jw_KPFF2          = res_KPFF['Jw_2']
particles_KPFF2   = res_KPFF['particles_2']
particles2_KPFF2  = res_KPFF['particles2_2']

X_KPFF_1          = res_KPFF['res_testLG']
Jx_KPFF_1         = res_KPFF['Jx_testLG']
Jw_KPFF_1         = res_KPFF['Jw_testLG']
particles_KPFF_1  = res_KPFF['particles_testLG']
particles2_KPFF_1 = res_KPFF['particles2_testLG']

X_KPFF2_1         = res_KPFF['res_testLG2']
Jx_KPFF2_1        = res_KPFF['Jx_testLG2']
Jw_KPFF2_1        = res_KPFF['Jw_testLG2']
particles_KPFF2_1  = res_KPFF['particles_testLG2']
particles2_KPFF2_1 = res_KPFF['particles2_testLG2']



for i in observed:
    plt.plot(X1_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_KPFF2_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

for i in unobserved:
    plt.plot(X1_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_KPFF2_1[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 



for i in observed:
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_KPFF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 

for i in unobserved:
    plt.plot(X_sparse[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KPFF2[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.plot(X_KPFF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
 



 



plt.scatter(particles_KPFF2_1[10,:,3],  particles_KPFF2_1[10,:,2])
plt.scatter(particles2_KPFF2_1[10,:,3],  particles2_KPFF2_1[10,:,2])
plt.show() 
 
 
plt.scatter(particles_KPFF_1[10,:,7],  particles_KPFF_1[10,:,8])
plt.scatter(particles2_KPFF_1[10,:,7],  particles2_KPFF_1[10,:,8])
plt.show() 
 
   

 

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(tf.reduce_mean((X_sparse - X_KPFF)**2, axis=1), linewidth=1, alpha=0.5, color="green") 
ax.plot(tf.reduce_mean((X_sparse - X_KPFF2)**2, axis=1), linewidth=1, alpha=0.5, color="red") 
    
plt.tight_layout()
plt.show()
    
 
 
 
 
 

#############################
# Plots, figures and tables # 
#############################


# def conditioning(J):
#     """Compute the conditioning number of a given Jacobian matrix using the 2-norm"""
#     norm        = tf.norm(J, ord=2)
#     Jinv        = tf.linalg.inv(J)
#     norminv     = tf.norm(Jinv, ord=2)
#     return norm * norminv

# def conditioning_asym(J):
#     """Compute the conditioning number of a given assymetric Jacobian matrix using the eigenvalues"""
#     _, s, _ = np.linalg.svd( J )
#     return np.max(s) / np.min(s)

# JxConds_KPFF_sparse = np.zeros((nT,Nl,Np))  
# for i in range(nT):
#     for j in range(Nl): 
#         for k in range(Np): 
#             JxConds_KPFF_sparse[i,j,k] = conditioning_asym(Jx_KPFF[i,j,k,:,:])

# JwConds_KPFF_sparse = np.zeros((nT,Nl,Np))  
# for i in range(nT):
#     for j in range(Nl): 
#         for k in range(Np): 
#             JwConds_KPFF_sparse[i,j,k] = conditioning_asym(Jw_KPFF[i,j,k,:,:])
            
# JxConds_KPFF2_sparse = np.zeros((nT,Nl,Np))  
# for i in range(nT):
#     for j in range(Nl): 
#         for k in range(Np): 
#             JxConds_KPFF2_sparse[i,j,k] = conditioning_asym(Jx_KPFF2[i,j,k,:,:])

# JwConds_KPFF2_sparse = np.zeros((nT,Nl,Np))  
# for i in range(nT):
#     for j in range(Nl): 
#         for k in range(Np): 
#             JwConds_KPFF2_sparse[i,j,k] = conditioning_asym(Jw_KPFF2[i,j,k,:,:])


# JxConds_KPFF_sparse2 = tf.where( tf.math.logical_and(tf.math.is_inf(JxConds_KPFF_sparse), JxConds_KPFF_sparse > 0.0) , tf.cast(1e9, tf.float64), JxConds_KPFF_sparse)
# JxConds_KPFF2_sparse2 = tf.where( tf.math.logical_and(tf.math.is_inf(JxConds_KPFF2_sparse), JxConds_KPFF2_sparse > 0.0) , tf.cast(1e9, tf.float64), JxConds_KPFF2_sparse)


# fig, ax = plt.subplots(1,2, figsize=(12,4))

# ax[0].plot(tf.reduce_mean(JxConds_KPFF_sparse2, axis=(1,2)), linewidth=1, alpha=0.5, color="green") 
# ax[0].plot(tf.reduce_mean(JxConds_KPFF2_sparse2, axis=(1,2)), linewidth=1, alpha=0.5, color="red") 

    
# ax[1].plot(tf.reduce_mean(JwConds_KPFF_sparse, axis=(1,2)), linewidth=1, alpha=0.5, color="green") 
# ax[1].plot(tf.reduce_mean(JwConds_KPFF2_sparse, axis=(1,2)), linewidth=1, alpha=0.5, color="red") 

# plt.tight_layout()
# plt.savefig(pathfig+"KPFF_J.png")

# plt.show()



# # with open(pathdat+"J_conds.pkl", "wb") as file:
# #     pkl.dump({"Jx_EDH_EKF": JxConds_EDH_EKF, 
# #               "Jw_EDH_EKF": JwConds_EDH_EKF,
# #               "Jx_LEDH_EKF": JxConds_LEDH_EKF,
# #               "Jw_LEDH_EKF": JwConds_LEDH_EKF,
# #               "Jw_EDH": JwConds_EDH, 
# #               "Jw_LEDH": JwConds_LEDH,
# #               "Jx_KPFF2_sparse": JxConds_KPFF2_sparse,
# #               "Jw_KPFF2_sparse": JwConds_KPFF2_sparse
# #               "Jx_KPFF2": JxConds_KPFF2,
# #               "Jw_KPFF2": JwConds_KPFF2}, file)

# with open(pathdat+"J_conds.pkl", 'rb') as file:
#     J_Conds = pkl.load(file)

# JxConds_EDH_EKF     = J_Conds["Jx_EDH_EKF"]
# JwConds_EDH_EKF     = J_Conds["Jw_EDH_EKF"]

# JxConds_LEDH_EKF    = J_Conds["Jx_LEDH_EKF"]
# JwConds_LEDH_EKF    = J_Conds["Jw_LEDH_EKF"]

# JwConds_EDH         = J_Conds["Jw_EDH"]
# JwConds_LEDH        = J_Conds["Jw_LEDH"]

# JxConds_KPFF_sparse = J_Conds["Jx_KPFF_sparse"]
# JwConds_KPFF_sparse = J_Conds["Jw_KPFF_sparse"]

# JxConds_KPFF2       = J_Conds["Jx_KPFF2"]
# JwConds_KPFF2       = J_Conds["Jw_KPFF2"]

# JxConds_KPFF2_sparse = J_Conds["Jx_KPFF2_sparse"]
# JwConds_KPFF2_sparse = J_Conds["Jw_KPFF2_sparse"]

    

# fig, ax = plt.subplots(1,2, figsize=(12,4))

# ax[0].plot(JxConds_EDH_EKF, linewidth=1, alpha=0.5, color="green") 
# ax[0].plot(tf.reduce_mean(JxConds_LEDH_EKF, axis=1), linewidth=1, alpha=0.5, color="red") 

    
# ax[1].plot(JwConds_EDH_EKF, linewidth=1, alpha=0.5, color="green") 
# ax[1].plot(tf.reduce_mean(JwConds_LEDH_EKF, axis=1),  linewidth=1, alpha=0.5, color="red") 

# plt.tight_layout()
# plt.savefig(pathfig+"EDH_J_EKF.png")

# plt.show()


# fig, ax = plt.subplots(figsize=(6,4))

# ax.plot(JwConds_EDH, linewidth=1, alpha=0.5, color="green") 
# ax.plot(tf.reduce_mean(JwConds_LEDH, axis=1), linewidth=1, alpha=0.5, color="red") 

    
# plt.tight_layout()
# plt.savefig(pathfig+"EDH_J_UKF.png")

# plt.show()




# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(np.repeat(1,10), np.arange(10), alpha=0.75, color=colors[0])
# ax.plot(np.repeat(2,10), np.arange(10), alpha=0.75, color=colors[1])
# ax.plot(np.repeat(3,10), np.arange(10), alpha=0.75, color=colors[2])
# ax.plot(np.repeat(4,10), np.arange(10),  alpha=0.75, color=colors[3])
# ax.plot(np.repeat(5,10), np.arange(10), alpha=0.75, color=colors[4])
# ax.plot(np.repeat(6,10), np.arange(10),  alpha=0.75, color=colors[5])
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(JxConds_EDH_EKF, alpha=0.75, color=colors[2])
# ax.plot(JxConds_LEDH_EKF.mean(axis=1), alpha=0.75, color=colors[3])
# ax.plot(JxConds_KPFF2.mean(axis=(1,2)), alpha=0.75, color=colors[4])
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(JwConds_EDH, alpha=0.75, color=colors[2])
# ax.plot(JwConds_LEDH.mean(axis=1), alpha=0.75, color=colors[3])
# ax.plot(JwConds_KPFF2.mean(axis=(1,2)), alpha=0.75, color=colors[4])
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(JxConds_KPFF2.mean(axis=(1,2)), alpha=0.75, color=colors[4])
# plt.tight_layout()
# plt.show()







# from itertools import permutations

# PermSets            = list(permutations(np.arange(nD).tolist()))
# PermList            = [list(l) for l in PermSets]
# def compute_OMT(x, xhat, n=nD, nPerm=len(PermList), Perm=PermList):
#     """Compute the OMAT error of between two vectors."""
#     dis             = np.zeros((nPerm,))
#     for i in range(nPerm):
#         idx         = Perm[i]
#         dis[i]      = np.sqrt(np.sum((x - xhat[idx])**2))
#     return np.min(dis) / n

# dis_EKF   = np.zeros((nT,))
# dis_UKF   = np.zeros((nT,))
# dis_PF    = np.zeros((nT,))
# dis_EDH   = np.zeros((nT,))
# dis_LEDH  = np.zeros((nT,))
# dis_KPFF2 = np.zeros((nT,))

# for i in range(nT):
#     dis_EKF[i] = compute_OMT(X_EKF[i,:].numpy(), X[i,:].numpy())
#     dis_UKF[i] = compute_OMT(X_UKF[i,:].numpy(), X[i,:].numpy())
#     dis_PF[i] = compute_OMT(X_PF[i,:].numpy(), X[i,:].numpy())
#     dis_EDH[i] = compute_OMT(X_EDH[i,:].numpy(), X[i,:].numpy())
#     dis_LEDH[i] = compute_OMT(X_LEDH[i,:].numpy(), X[i,:].numpy())
#     dis_KPFF2[i] = compute_OMT(X_KPFF2[i,:].numpy(), X[i,:].numpy())

# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(dis_UKF, alpha=0.75, color=colors[0])
# ax.plot(dis_PF, alpha=0.75, color=colors[1])
# ax.plot(dis_EDH, alpha=0.75, color=colors[2])
# ax.plot(dis_LEDH, alpha=0.75, color=colors[3])
# ax.plot(dis_KPFF2, alpha=0.75, color=colors[4])
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(dis_EKF, alpha=0.75, color=colors[5])
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot( tf.reduce_mean((X_UKF - X)**2, axis=1), alpha=0.75, color=colors[0])
# ax.plot( tf.reduce_mean((X_PF - X)**2, axis=1), alpha=0.75, color=colors[1])
# ax.plot( tf.reduce_mean((X_EDH - X)**2, axis=1), alpha=0.75, color=colors[2])
# ax.plot( tf.reduce_mean((X_LEDH - X)**2, axis=1), alpha=0.75, color=colors[3])
# ax.plot( tf.reduce_mean((X_KPFF2 - X)**2, axis=1), alpha=0.75, color=colors[4])
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot( tf.reduce_mean((X_EKF - X)**2, axis=1), alpha=0.75, color=colors[5])
# plt.tight_layout()
# plt.show()




# i = 1
# fig, ax = plt.subplots(figsize=(12,4))
# # ax.plot( (X_UKF[:,i] - X[:,i])**2, alpha=0.75, color=colors[0])
# ax.plot( (X_PF[:,i] - X[:,i])**2, alpha=0.75, color=colors[1])
# ax.plot( (X_EDH_EKF[:,i] - X[:,i])**2, alpha=0.75, color=colors[2])
# ax.plot( (X_LEDH_EKF[:,i] - X[:,i])**2, alpha=0.75, color=colors[3])
# ax.plot( (X_KPFF2[:,i] - X[:,i])**2, alpha=0.75, color=colors[4])
# plt.tight_layout()
# plt.show()

