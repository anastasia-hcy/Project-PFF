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

pathdat         = "C:/Users/anastasia/MyProjects/JPMorgan/data/"

nT              = 100
nD              = 4
nX              = 10
Ny              = nX - 2
unobserved      = [3,7]
observed        = [0,1,2,4,5,6,8,9]

Np              = 100

with open(pathdat+"data_sim.pkl", 'rb') as file:
    data        = pkl.load(file)    

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





# def background_task0_EKF(y, N, T, path):
    
#     start_cpu_time  = time.process_time()
#     initial_rss     = psutil.Process(os.getpid()).memory_info().rss

#     X_LEDH_SDE_1, ess_LEDH_SDE_1, weights_LEDH_SDE_1, Jx_LEDH_SDE_1, Jw_LEDH_SDE_1 = LEDH(y, N=N, Nstep=T, method='EKF', stochastic=True)
    
#     final_rss       = psutil.Process(os.getpid()).memory_info().rss
#     memory_increase_mib_1 = (final_rss - initial_rss) / (1024 ** 2)

#     end_cpu_time    = time.process_time()
#     cpu_time_taken_1 = end_cpu_time - start_cpu_time    
    
#     with open(path+"res_LEDH_SDE_EKF_LG.pkl", 'wb') as file:
#         pkl.dump({"res": X_LEDH_SDE_1, 
#                   "ess": ess_LEDH_SDE_1,
#                   "weights": weights_LEDH_SDE_1,
#                   "Jx": Jx_LEDH_SDE_1, 
#                   "Jw": Jw_LEDH_SDE_1,
#                   "cpu": [cpu_time_taken_1, memory_increase_mib_1]}, file)


# def background_task0_UKF(y, N, T, path):
    
#     start_cpu_time  = time.process_time()
#     initial_rss     = psutil.Process(os.getpid()).memory_info().rss

#     X_LEDH_SDE_1, ess_LEDH_SDE_1, weights_LEDH_SDE_1, Jx_LEDH_SDE_1, Jw_LEDH_SDE_1 = LEDH(y, N=N, Nstep=T, method='UKF', stochastic=True)
    
#     final_rss       = psutil.Process(os.getpid()).memory_info().rss
#     memory_increase_mib_1 = (final_rss - initial_rss) / (1024 ** 2)

#     end_cpu_time    = time.process_time()
#     cpu_time_taken_1 = end_cpu_time - start_cpu_time    
    
#     with open(path+"res_LEDH_SDE_UKF_LG.pkl", 'wb') as file:
#         pkl.dump({"res": X_LEDH_SDE_1, 
#                   "ess": ess_LEDH_SDE_1,
#                   "weights": weights_LEDH_SDE_1,
#                   "Jx": Jx_LEDH_SDE_1, 
#                   "Jw": Jw_LEDH_SDE_1,
#                   "cpu": [cpu_time_taken_1, memory_increase_mib_1]}, file)




def background_task(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X_SDE, cond, stiff, beta = SDE(y, N=N, linear=True)
    
    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken  = end_cpu_time - start_cpu_time    
    
    with open(path+"res_SDE_LG.pkl", 'wb') as file:
        pkl.dump({"res": X_SDE,
                  "cond": cond,
                  "stiff": stiff,
                  "beta": beta,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)


def background_task2(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X_SDE, cond, stiff, beta = SDE(y, N=N, linear=False)
    
    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken  = end_cpu_time - start_cpu_time    
    
    with open(path+"res_SDE_LG_homo.pkl", 'wb') as file:
        pkl.dump({"res": X_SDE,
                  "cond": cond,
                  "stiff": stiff,
                  "beta": beta,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)




if __name__ == '__main__':
    
    p = multiprocessing.Process(target=background_task_trial, args=(nT, nD, Y, Cx, pathdat,))
    p.start()

    # p20 = multiprocessing.Process(target=background_task0_EKF, args=(Y1, Np, Ns, pathdat,))
    # p20.start()
    
    # p30 = multiprocessing.Process(target=background_task0_UKF, args=(Y1, Np, Ns, pathdat,))
    # p30.start()

    p2 = multiprocessing.Process(target=background_task, args=(Y1, Np, pathdat,))
    p2.start()
    
    p3 = multiprocessing.Process(target=background_task2, args=(Y1, Np, pathdat,))
    p3.start()



