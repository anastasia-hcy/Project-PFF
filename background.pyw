
import multiprocessing
import os, sys

from main import X1_EDH_EKF

pythonw_path = os.path.join(sys.prefix, 'pythonw.exe')
if os.path.exists(pythonw_path) and sys.executable != pythonw_path:
    multiprocessing.set_executable(pythonw_path)

import numpy as np
import pickle as pkl 
import psutil
import time
from scripts import norm_rvs
from scripts import ExactDH, LocalExactDH, KernelParticleFlow 

pathdat         = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/PPFResults/"

nT              = 100
nD              = 4
nX              = 10
Ny              = nX - 2
unobserved      = [3,7]
observed        = [0,1,2,4,5,6,8,9]
Np              = 100

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

EDH             = ExactDH(nTimes=nT, ndims=nD)
LEDH            = LocalExactDH(nTimes=nT, ndims=nD)
KernelPFF       = KernelParticleFlow(nTimes=nT, ndims=nD, nx=nX)

def background_task_trial(nTimes, ndims, x, C, path):
    res = np.zeros((nTimes,ndims))
    for i in range(nTimes):
        res[i,:] = norm_rvs(ndims, x[i,:], C)
    with open(path+"bg_try.pkl", 'wb') as file:
        pkl.dump({'res':res}, file)         

def background_edh_ekf(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X1_EDH_EKF, ess1_EDH_EKF, weights1_EDH_EKF, Jx1_EDH_EKF, Jw1_EDH_EKF = EDH.run(y=y, N=N)

    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken = end_cpu_time - start_cpu_time    
    
    with open(path+"res_EDH_EKF_LG.pkl", 'wb') as file:
        pkl.dump({"res": X1_EDH_EKF, 
                  "ess": ess1_EDH_EKF,
                  "weights": weights1_EDH_EKF,
                  "Jx": Jx1_EDH_EKF, 
                  "Jw": Jw1_EDH_EKF,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)

def background_edh_ukf(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X1_EDH, ess1_EDH, weights1_EDH, Jx1_EDH, Jw1_EDH = EDH.run(y=y, N=N, method='UKF')

    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken = end_cpu_time - start_cpu_time    
    
    with open(path+"res_EDH_LG.pkl", 'wb') as file:
        pkl.dump({"res": X1_EDH, 
                  "ess": ess1_EDH,
                  "weights": weights1_EDH,
                  "Jx": Jx1_EDH, 
                  "Jw": Jw1_EDH,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)

def background_ledh_ekf(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X1_LEDH_EKF, ess1_LEDH_EKF, weights1_LEDH_EKF, Jx1_LEDH_EKF, Jw1_LEDH_EKF = LEDH.run(y=y, N=N)

    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken = end_cpu_time - start_cpu_time    
    
    with open(path+"res_LEDH_EKF_LG.pkl", 'wb') as file:
        pkl.dump({"res": X1_LEDH_EKF, 
                  "ess": ess1_LEDH_EKF,
                  "weights": weights1_LEDH_EKF,
                  "Jx": Jx1_LEDH_EKF, 
                  "Jw": Jw1_LEDH_EKF,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)

def background_ledh_ukf(y, N, path):
    
    start_cpu_time  = time.process_time()
    initial_rss     = psutil.Process(os.getpid()).memory_info().rss

    X1_LEDH, ess1_LEDH, weights1_LEDH, Jx1_LEDH, Jw1_LEDH = LEDH.run(y=y, N=N, method='UKF')

    final_rss       = psutil.Process(os.getpid()).memory_info().rss
    memory_increase_mib = (final_rss - initial_rss) / (1024 ** 2)

    end_cpu_time    = time.process_time()
    cpu_time_taken = end_cpu_time - start_cpu_time    
    
    with open(path+"res_LEDH_LG.pkl", 'wb') as file:
        pkl.dump({"res": X1_LEDH, 
                  "ess": ess1_LEDH,
                  "weights": weights1_LEDH,
                  "Jx": Jx1_LEDH, 
                  "Jw": Jw1_LEDH,
                  "cpu": [cpu_time_taken, memory_increase_mib]}, file)



if __name__ == '__main__':
    
    p = multiprocessing.Process(target=background_task_trial, args=(nT, nD, Y, Cx, pathdat,))
    p.start()

    p2 = multiprocessing.Process(target=background_edh_ekf, args=(Y1, Np, pathdat,))
    p2.start()
    
    p3 = multiprocessing.Process(target=background_edh_ukf, args=(Y1, Np, pathdat,))
    p3.start()

    p4 = multiprocessing.Process(target=background_ledh_ekf, args=(Y1, Np, pathdat,))
    p4.start()
    
    p5 = multiprocessing.Process(target=background_ledh_ukf, args=(Y1, Np, pathdat,))
    p5.start()


