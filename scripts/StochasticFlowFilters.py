import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve 
from .model import norm_rvs, measurements_pred
from .ParticleFilter import StandardParticleFilter

class StochasticPFF: 
    
    def __init__(self, *, nTimes, ndims):
        self.nTimes     = nTimes
        self.ndims      = ndims
        self.PF         = StandardParticleFilter(nTimes=nTimes, ndims=ndims)
   
    def Dai22eq22(self, beta, mu, HessianLogPrior, HessianLogLikelihood, Q, U):
        """Compute and return the negative Hessian matrix, Jacobian matrix and stiffness, M, F, and kappa, using equation (22) in Dai et al. (2022)."""
        M               = - (HessianLogPrior + beta * HessianLogLikelihood) 
        kappa           = tf.linalg.trace(M) * tf.linalg.trace(tf.linalg.inv(M + U))
        d_beta          = tf.math.sqrt(2 * mu * kappa)
        F               = 1/2 * Q @ (-M) - d_beta/2 * tf.linalg.inv(-M - U) @ HessianLogLikelihood 
        return M, F, kappa

    def stiffness_ratio(self, F):
        """Compute and return the stiffness ratio using the Jacobian matrix F."""
        ei              = tf.constant(tf.math.real(tf.linalg.eigvals(F)), dtype=tf.float64)
        Ratio           = tf.reduce_max(ei) / tf.reduce_min(ei)
        return Ratio

    def posterior_Gaussian_mean(self, mx, my, Cx, Cy, U):
        Cx_inv = tf.linalg.inv(Cx + U)
        Cy_inv = tf.linalg.inv(Cy + U)
        return tf.linalg.matvec( tf.linalg.inv(Cx_inv + Cy_inv) , tf.linalg.matvec(Cx_inv, mx) + tf.linalg.matvec(Cy_inv, my) )

    def posterior_Gaussian_cov(self, Cx, Cy, U):
        Cx_inv = tf.linalg.inv(Cx + U)
        Cy_inv = tf.linalg.inv(Cy + U)
        return tf.linalg.inv(Cx_inv + Cy_inv) 

    def JacobiLogNormal(self, x, mu, P, U):
        """Compute and return the Jacobian of the log of a Normal distribution."""
        return tf.linalg.matvec(tf.linalg.inv(P + U), (x - mu))

    def HessianLogNormal(self, P, U):
        """Compute and return the Hessian of the log of a Normal distribution."""
        return - tf.linalg.inv(P + U) 

    def Dai22eq11eq12(self, M, HessianLogLikelihood, Q, U):
        """Compute and return constant terms, K1 and K2, specified by equations (11) and (12) of Dai et al. (2022)."""
        M_inv           = tf.linalg.inv(-M-U)
        K1              = Q/2 + 1/2 * M_inv @ HessianLogLikelihood @ M_inv 
        K2              = - M_inv
        return K1, K2

    def drift_f(self, K1, K2, JacobiLogP, JacobiLogLikelihood):
        """Compute and return the drift function specified by equation (10) of Dai et al. (2022)."""
        f               = tf.linalg.matvec(K1, JacobiLogP) + tf.linalg.matvec(K2, JacobiLogLikelihood)
        return f

    def sde_flow_dynamics(self, N, ndims, K1, K2, JLL,JLP, dL, w0, wI, q):
        """Compute and return the flow dynamics of the particles using the SDE."""
        dx              = tf.Variable(tf.zeros((N,ndims), dtype=tf.float64))    
        f               = self.drift_f(K1, K2, JLP, JLL)         
        for i in range(N):        
            dw          = norm_rvs(ndims, w0, dL * wI) 
            dx[i,:].assign( f * dL + tf.linalg.matvec(q, dw) )         
        return dx
    
    def run(self, *, y, model=None, A=None, B=None, V=None, W=None, N=None, Nstep=None, mu0=None, mu=None):
        """Run the EDH and return the filtered states, X_filtered.""" 

        model                       = "LG" if model is None else model      
        if model == "SV" and A is None : 
            A                       = tf.eye(self.ndims, dtype=tf.float64) * 0.5  
        if model != "SV" and A is None : 
            A                       = tf.eye(self.ndims, dtype=tf.float64) 
        B                           = tf.eye(self.ndims, dtype=tf.float64) if B is None else B
        V                           = tf.eye(self.ndims, dtype=tf.float64) if V is None else V
        W                           = tf.eye(self.ndims, dtype=tf.float64) if W is None else W

        mu0                         = tf.zeros((self.ndims,), dtype=tf.float64) if mu0 is None else mu0
        if model == "SV":
            Sigma0                  = V @ tf.linalg.inv(tf.eye(self.ndims, dtype=tf.float64) - A @ A)  
        if model != "SV": 
            Sigma0                  = V
        mu                          = tf.zeros((self.ndims,), dtype=tf.float64) if mu is None else mu
        U                           = tf.eye(self.ndims, dtype=tf.float64) * 1e-9
        I                           = tf.ones((self.ndims,), dtype=tf.float64)
        
        N                           = 1000 if N is None else N
        Np                          = N
        Nstep                       = 100 if Nstep is None else Nstep
        Rates                       = tf.constant(tf.linspace(0.0, 1.0, Nstep + 1).numpy(), dtype=tf.float64)
        
        CovPost                     = self.posterior_Gaussian_cov(Sigma0, W, U)
        Hess_LLike                  = self.HessianLogNormal(W, U)
        Hess_Lprior                 = self.HessianLogNormal(Sigma0, U)
        
        betas                       = Rates[1:]  
        # betas       = final_solve(Rates[1:], mc, Hess_Lprior, Hess_LLike, u) 
        mc                          = 0.1  
        Q                           = tf.eye(self.ndims, dtype=tf.float64) 
        q                           = tf.linalg.cholesky(Q)
        w0                          = tf.zeros((self.ndims,), dtype=tf.float64)

        Ms                          = tf.Variable(tf.zeros((Nstep,self.ndims,self.ndims), dtype=tf.float64))
        Fs                          = tf.Variable(tf.zeros((Nstep,self.ndims,self.ndims), dtype=tf.float64))
        cond_num                    = tf.Variable(tf.zeros((Nstep,), dtype=tf.float64))
        stiffness                   = tf.Variable(tf.zeros((Nstep,), dtype=tf.float64))
        K1s                         = tf.Variable(tf.zeros((Nstep,self.ndims,self.ndims), dtype=tf.float64))
        K2s                         = tf.Variable(tf.zeros((Nstep,self.ndims,self.ndims), dtype=tf.float64))
        
        for j in range(Nstep): 
            M, F, k                 = self.Dai22eq22(betas[j], mc, Hess_Lprior, Hess_LLike, Q, U)
            sr                      = self.stiffness_ratio(F)
            K1, K2                  = self.Dai22eq11eq12(M, Hess_LLike, Q, U)
            
            Ms[j,:,:].assign(M)
            Fs[j,:,:].assign(F)
            cond_num[j].assign(k)
            stiffness[j].assign(sr)
            K1s[j,:,:].assign(K1)
            K2s[j,:,:].assign(K2)

        x_filt                      = mu0
        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)

        for i in range(self.nTimes):
            
            x_prev                  = self.PF.initiate_particles(Np, self.ndims, x_filt, Sigma0)            
            gx                      = measurements_pred(model, self.ndims, mu, B, x_prev, W, U)
            mu_p                    = self.posterior_Gaussian_mean(mu0, gx, Sigma0, W, U)
            JLL                     = self.JacobiLogNormal(y[i,:], gx, W, U)
            JL0                     = self.JacobiLogNormal(x_filt, mu0, Sigma0, U)
            JL1                     = self.JacobiLogNormal(x_filt, mu_p, CovPost, U)
            
            for j in range(Nstep):      
                JLP                 = (1.0 - betas[j]) * JL0 + betas[j] * JL1  
                dL                  = Rates[j+1] - Rates[j]                   
                dx                  = self.sde_flow_dynamics(Np, self.ndims, K1s[j,:,:], K2s[j,:,:], JLL, JLP, dL, w0, I, q)
                x_prev.assign_add(dx)
                
            x_filt                  = tf.reduce_mean(x_prev, axis=0)
            X_filtered[i,:].assign(x_filt)
            
        return X_filtered, cond_num, stiffness, betas
    



def Dai22eq28(Lambda_span, beta, args):
    """Compute and return the anti-derivatives of the conditioning number given by negative Hessian, M, over the interval, Lambda_span, using equation (28) in Dai et al. (2022)."""
    mu, HessianLogPrior, HessianLogLikelihood, U = args
    M                = - (HessianLogPrior + beta * HessianLogLikelihood) 
    M_inv            = tf.linalg.inv(M + U)
    db2_dL2          = - mu * ( tf.linalg.trace(HessianLogLikelihood) * tf.linalg.trace(M_inv) + tf.linalg.trace(M) * tf.linalg.trace(M_inv @ HessianLogLikelihood @ M_inv) )
    return db2_dL2

def initial_solve_err(b0, args):
    """Compute the error of the boundary condition beta(1) = 1 at the initial condition beta(0) = 0 for the ODE solver."""
    sol              = solve_ivp(Dai22eq28, t_span=[0,1], y0=b0, args=(args,))
    return tf.constant([sol.y[0,-1] - 1], dtype=tf.float64) 
 
def final_solve(Lambda, mu, HessianLogPrior, HessianLogLikelihood, U): 
    """Solve the ODE system for beta sepcified by Dai22eq28 over the input interval, Lambdas, given the parameters."""
    args            = (mu, HessianLogPrior, HessianLogLikelihood, U)
    root            = fsolve(initial_solve_err, [1e-6], args=(args,))
    sol             = solve_ivp(Dai22eq28, [0,1], y0=root, args=(args,), t_eval=Lambda) 
    return tf.constant(sol.y[0], dtype=tf.float64)
