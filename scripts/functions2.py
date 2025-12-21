import os

# path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/"
# path                = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/scripts/"
# import sys
# os.chdir(path)
# cwd = os.getcwd()
# sys.path.append(cwd)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scipy.integrate import solve_ivp
from .model import norm_rvs, measurements_pred, measurements_Jacobi, measurements_covyHat, SE_Cov_div

############################
# Stochastic Particle Flow # 
############################

def Dai22eq28(mu):
    return 

# 1. Define the ODE function
# The function should accept (t, y) and return a list/array of derivatives
def vdp1(t, y):
    # y[0] is the position, y[1] is the velocity
    # Equations:
    # dy[0]/dt = y[1]
    # dy[1]/dt = (1 - y[0]**2) * y[1] - y[0]
    return [y[1], (1 - y[0]**2) * y[1] - y[0]]

# 2. Define the time span and initial conditions
t_span = [0, 20] # Integrate from t=0 to t=20
y0 = [2, 0]      # Initial conditions for [position, velocity]

# 3. Solve the ODE
sol = solve_ivp(vdp1, t_span, y0)