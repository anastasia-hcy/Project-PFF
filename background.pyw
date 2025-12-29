###################
# Set executables #
###################

import multiprocessing
import os, sys

pythonw_path = os.path.join(sys.prefix, 'pythonw.exe')
if os.path.exists(pythonw_path) and sys.executable != pythonw_path:
    multiprocessing.set_executable(pythonw_path)

############# 
# Libraries #
#############

import numpy as np
import pickle as pkl 
from scripts import norm_rvs, SDE

import psutil
import time

######################### 
# Datasets & parameters #
#########################

pathdat             = "C:/Users/anastasia/MyProjects/JPMorgan/data/"
# pathdat             = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/data/"

nT              = 100
nD              = 4
nX              = 10
Ny              = nX - 2
unobserved      = [3,7]
observed        = [0,1,2,4,5,6,8,9]
Np              = 50

with open(pathdat+"data_sim.pkl", 'rb') as file:
    data        = pkl.load(file)    

X1              = data['LG_States']
Y1              = data['LG_Obs']
Y               = data['SV_Obs']
A               = data['A']
Cx              = data['Cx']


def background_task_trial(nTimes, ndims, x, C, path):
    res = np.zeros((nTimes,ndims))
    for i in range(nTimes):
        res[i,:] = norm_rvs(ndims, x[i,:], C)
        
    with open(path+"bg_try.pkl", 'wb') as file:
        pkl.dump({'res':res}, file)         


def background_task(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X_SDE_1, Cond_SDE_1, stiff_SDE_1, beta_SDE_1 = SDE(y, N=N)
    
    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib_1 = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken_1 = end_cpu_time - start_cpu_time    
    
    with open(path+"res_SDE_LG.pkl", 'wb') as file:
        pkl.dump({"res": X_SDE_1, 
                  "cond": Cond_SDE_1,
                  "stiff": stiff_SDE_1,
                  "beta": beta_SDE_1, 
                  "cpu": [cpu_time_taken_1, memory_increase_mib_1]}, file)



def background_task2(y, N, A, V, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss
    
    X_SDE, Cond_SDE, stiff_SDE, beta_SDE = SDE(y, model="SV", N=N, A=A, V=V)
    
    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken  = end_cpu_time - start_cpu_time    
    
    with open(path+"res_SDE.pkl", 'wb') as file:
        pkl.dump({"res": X_SDE, 
                  "cond": Cond_SDE,
                  "stiff": stiff_SDE,
                  "beta": beta_SDE,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)


if __name__ == '__main__':
    
    p = multiprocessing.Process(target=background_task_trial, args=(nT, nD, X1, Cx, pathdat,))
    p.start()

    p2 = multiprocessing.Process(target=background_task, args=(Y1, Np, pathdat,))
    p2.start()
    
    p3 = multiprocessing.Process(target=background_task2, args=(Y, Np, A, Cx, pathdat,))
    p3.start()
