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

import psutil
import time

from functions import SE_Cov_div
from functions import LGSSM, SVSSM
from functions import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from functions import ParticleFilter
from functions import EDH, LEDH, KernelPFF




nT = 100
nD = 4
Np = 500
Cx, DivCx = SE_Cov_div(nD, tf.random.normal((nD,), dtype=tf.float64), scale=1.25, length=2.0)

    
    

def get_current_process_ram_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mib = memory_info.rss / (1024 ** 2)  # Convert to MiB
    vms_mib = memory_info.vms / (1024 ** 2)  # Convert to MiB
    print(f"Current process RAM usage (RSS): {rss_mib:.2f} MiB")
    print(f"Current process RAM usage (VMS): {vms_mib:.2f} MiB")

get_current_process_ram_usage()


X, Y = LGSSM(nT,nD)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
    

start_cpu_time  = time.process_time()
initial_rss     = psutil.Process(os.getpid()).memory_info().rss

X_KF            = KalmanFilter(Y)

final_rss       = psutil.Process(os.getpid()).memory_info().rss
memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)
print(f"Memory increase during code block: {memory_increase_mib:.2f} MiB")

end_cpu_time = time.process_time()
cpu_time_taken = end_cpu_time - start_cpu_time
print(f"CPU time taken: {cpu_time_taken:.6f} seconds")

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_KF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 










A       = tf.linalg.diag(tf.constant(tf.linspace(0.5,0.95,nD).numpy(), dtype=tf.float64))
X, Y    = SVSSM(nT, nD, A=A, V=Cx)
for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(Y[:,i], linewidth=1, alpha=0.75)
    plt.show() 


start_cpu_time  = time.process_time()
initial_rss     = psutil.Process(os.getpid()).memory_info().rss

X_EKF           = ExtendedKalmanFilter(Y, A=A)

final_rss       = psutil.Process(os.getpid()).memory_info().rss
memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

end_cpu_time = time.process_time()
cpu_time_taken = end_cpu_time - start_cpu_time

print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_EKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    

start_cpu_time  = time.process_time()
initial_rss     = psutil.Process(os.getpid()).memory_info().rss

X_UKF           = UnscentedKalmanFilter(Y, A=A)

final_rss       = psutil.Process(os.getpid()).memory_info().rss
memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

end_cpu_time = time.process_time()
cpu_time_taken = end_cpu_time - start_cpu_time

print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
print(f"CPU time taken: {cpu_time_taken:.3f} seconds")


for i in range(nD):
    plt.plot(X[:,i], linewidth=1, alpha=0.75) 
    plt.plot(X_UKF[:,i], linewidth=1, alpha=0.75, linestyle='dashed') 
    plt.show() 
    
    
    
start_cpu_time  = time.process_time()
initial_rss     = psutil.Process(os.getpid()).memory_info().rss

X_PF, ess_PF, weights_PF, particles_PF = ParticleFilter(Y, N=Np, A=A)

final_rss       = psutil.Process(os.getpid()).memory_info().rss
memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

end_cpu_time = time.process_time()
cpu_time_taken = end_cpu_time - start_cpu_time

print(f"Memory increase during code block: {memory_increase_mib:.3f} MiB")
print(f"CPU time taken: {cpu_time_taken:.3f} seconds")

    
    
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

    
     
 

    