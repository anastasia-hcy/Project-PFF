import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

from .model import norm_rvs, measurements_pred, measurements_Jacobi, measurements_covyHat, SE_Cov_div
from .KalmanFilters import ExtendedKalmanFilter, UnscentedKalmanFilter
from .ParticleFilter import StandardParticleFilter

def Li17eq10(L, H, P, R, U):
    """Compute and return the first component, A, of the particle flow dynamics, Ax + b, under the assumption of linear Gaussian."""
    C           = tf.linalg.inv( L * H @ P @ tf.transpose(H) + R + U )
    return -1/2 * P @ tf.transpose(H) @ C @ H 

def Li17eq11(I, L, A, H, P, R, y, ei, e0i, U):
    """Compute and return the second component, b, of the particle flow dynamics, Ax + b, under the assumption of linear Gaussian."""
    M1          = (I + 2*L * A) 
    M2          = (I + L * A) @ P @ tf.transpose(H) @ tf.linalg.inv(R + U)
    v           = tf.linalg.matvec(M2, (y - ei)) + tf.linalg.matvec(A, e0i)
    return tf.linalg.matvec(M1, v)

class ExactDH: 

    def __init__(self, *, nTimes, ndims):
        self.nTimes     = nTimes
        self.ndims      = ndims
        self.EKF        = ExtendedKalmanFilter(nTimes=nTimes, ndims=ndims)
        self.UKF        = UnscentedKalmanFilter(nTimes=nTimes, ndims=ndims)
        self.PF         = StandardParticleFilter(nTimes=nTimes, ndims=ndims)

    def EDH_linearize_EKF(self, N, n, xprev, xhat, Pprev, A, V, U): 
        """Predict the pseudo particles, eta and eta0, by EKF using the previous state and its associated particles, xhat and xprev."""
        m_pred, P_pred  = self.EKF.Predict(xhat, Pprev, A, V) 
        eta             = tf.Variable(m_pred)
        eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        for i in range(N):
            eta0[i,:].assign( norm_rvs(n, xprev[i,:], P_pred + U) )    
        return eta, eta0, m_pred, P_pred 

    def EDH_linearize_UKF(self, N, n, xprev, xhat, Pprev, A, V, wm, wc, wi, L, U):
        """Predict the pseudo particles, eta and eta0, by UKF using the previous state and its associated particles, xhat and xprev."""
            
        SP              = self.UKF.SigmaPoints(n, xhat, Pprev, L)
        X_sp            = SP @ tf.transpose(A)
        
        m_pred          = self.UKF.Predict_mean(wm, wi, X_sp) 
        P_pred          = self.UKF.Predict_cov(n, wc, wi, X_sp, m_pred, U, Cov=V)  
        
        eta             = tf.Variable(m_pred)
        eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        for i in range(N): 
            eta0[i,:].assign( norm_rvs(n, xprev[i,:], P_pred + U) )               
        return eta, eta0, m_pred, P_pred 

    def EDH_flow_dynamics(self, N, n, Lamb, epsilon, I, e, e0, P, H, R, er, y, U):
        """Compute and return the flow dynamics of the pseudo particles under linear Gaussian for migration."""
        Ai              = Li17eq10(Lamb, H, P, R, U)
        bi              = Li17eq11(I, Lamb, Ai, H, P, R, y, er, e, U)
        move0           = tf.Variable(tf.zeros((N,n), dtype=tf.float64)) 
        for i in range(N): 
            move0[i,:].assign( epsilon * (tf.linalg.matvec(Ai, e0[i,:]) + bi) )  
        move            = epsilon * (tf.linalg.matvec(Ai, e) + bi) 
        return move0, move

    def EDH_flow_lp(self, N, eta0, eta1, xprev, y, SigmaX, muy, SigmaY, U):
        """Compute and return the log posterior of the migrated pseudo particles."""
        Lp              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        yLL             = self.PF.LogLikelihood(y, muy, SigmaY, U)
        for i in range(N):  
            Lp = Lp.write( i, yLL + self.PF.LogTarget(eta1[i,:], xprev[i,:], SigmaX) - self.PF.LogImportance(eta0[i,:], xprev[i,:], SigmaX) )  
        return Lp.stack()


    def run(self, *, y, model=None, A=None, B=None, V=None, W=None, N=None, Nstep=None, mu0=None, mu=None, method=None, stepsize=None):
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
        NT                          = N/2   
        method                      = "EKF" if method is None else method 
        if method == "UKF":
            weight0_m, weight0_c, weighti, L = self.UKF.SigmaWeights(self.ndims)
        stepsize                    = 1e-3 if stepsize is None else stepsize     
        Nstep                       = 30 if Nstep is None else Nstep
        Steps                       = tf.constant([stepsize * (1.2*i) for i in range(Nstep+1)], dtype=tf.float64)
        Rates                       = tf.math.cumsum(Steps) 

        x_filt                      = mu0
        P_prev                      = A @ Sigma0 @ tf.transpose(A) + V      
        x_prev                      = self.PF.initiate_particles(Np, self.ndims, x_filt, P_prev)
        w_prev                      = tf.ones((N,), dtype=tf.float64) / N 

        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        ESS                         = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        Weights                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        JacobiX                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        JacobiW                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)    

        for i in range(self.nTimes): 

            if method == "UKF":        
                eta, eta0, m_pred, P_pred   = self.EDH_linearize_UKF(Np, self.ndims, x_prev, x_filt, P_prev, A, V, weight0_m, weight0_c, weighti, L, U)
                y_pred, R, Rcross           = self.UKF.Predict(model, self.ndims, m_pred, P_pred, weight0_m, weight0_c, weighti, L, B, W, U)
                H, Hw                       = measurements_Jacobi(model, I, m_pred, y_pred, B)

            if method == "EKF":            
                eta, eta0, m_pred, P_pred   = self.EDH_linearize_EKF(Np, self.ndims, x_prev, x_filt, P_prev, A, V, U)
                y_pred                      = measurements_pred(model, self.ndims, mu, B, m_pred, W, U) 
                H, Hw                       = measurements_Jacobi(model, I, m_pred, y_pred, B)
                R                           = measurements_covyHat(model, Hw, W)

            el                              = y_pred - tf.linalg.matvec(H, m_pred) 
            eta1                            = eta0
            for j in range(1,Nstep+1): 
                eta1_move, eta_move         = self.EDH_flow_dynamics(Np, self.ndims, Rates[j], Steps[j], I, eta, eta1, P_pred, H, R, el, y[i,:], U)  
                eta.assign_add(eta_move)
                eta1.assign_add(eta1_move)
            
            lp                              = self.EDH_flow_lp(Np, eta0, eta1, x_prev, y[i,:], P_pred, y_pred, R, U)     
            w_pred                          = self.PF.compute_weights(w_prev, lp)
            w_norm                          = self.PF.normalize_weights(w_pred)        
            ness                            = self.PF.compute_ESS(w_norm)
             
            if ness < NT: 
                xbar, wbar                  = self.PF.multinomial_resample(Np, eta1, w_norm)
                x_filt                      = self.PF.compute_posterior(wbar, xbar)
                x_prev                      = xbar
            else: 
                x_filt                      = self.PF.compute_posterior(w_norm, eta1)
                x_prev                      = eta1
                
            if method == "UKF":
                K                           = self.UKF.Gain(Rcross, R, U)
                _, P_filt                   = self.UKF.Filter(m_pred, P_pred, R, y[i,:], y_pred, K)
            if method == "EKF":    
                K                           = self.EKF.Gain(P_pred, H, Hw, W, U)
                _, P_filt                   = self.EKF.Filter(m_pred, P_pred, y[i,:], y_pred, H, K)
            P_prev      = P_filt

            X_filtered                      = X_filtered.write(i, x_filt) 
            ESS                             = ESS.write(i, ness) 
            Weights                         = Weights.write(i, w_norm)    
            JacobiX                         = JacobiX.write(i, H)  
            JacobiW                         = JacobiW.write(i, Hw)
        
        return X_filtered.stack(), ESS.stack(), Weights.stack(), JacobiX.stack(), JacobiW.stack()



class LocalExactDH: 

    def __init__(self, *, nTimes, ndims):
        self.nTimes     = nTimes
        self.ndims      = ndims
        self.EKF        = ExtendedKalmanFilter(nTimes=nTimes, ndims=ndims)
        self.UKF        = UnscentedKalmanFilter(nTimes=nTimes, ndims=ndims)
        self.PF         = StandardParticleFilter(nTimes=nTimes, ndims=ndims)

    def LEDH_linearize_EKF(self, N, n, model, I1, xprev, Pprev, A, B, V, W, muy, U): 
        """Predict the pseudo particles, eta and eta0, by EKF using the previous state and its associated particles, xhat and xprev."""
        
        eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        eta             = tf.Variable(tf.zeros((N,n), dtype=tf.float64))

        m               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        P               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        y               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        H               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        Hw              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        Cy              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        err             = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        
        for i in range(N): 

            mi_pred, Pi_pred        = self.EKF.Predict(xprev[i,:], Pprev[i,:,:], A, V) 
            yi_pred                 = measurements_pred(model, n, muy, B, eta[i,:], W, U)            
            Jxi, Jwi                = measurements_Jacobi(model, I1, eta[i,:], yi_pred, B)
            Cyi                     = measurements_covyHat(model, Jwi, W)

            eta[i,:].assign(mi_pred)
            eta0[i,:].assign( norm_rvs(n, mi_pred, Pi_pred + U) )

            m           = m.write(i, mi_pred)
            P           = P.write(i, Pi_pred)
            y           = y.write(i, yi_pred)
            H           = H.write(i, Jxi)
            Hw          = Hw.write(i, Jwi)
            Cy          = Cy.write(i, Cyi)
            err         = err.write(i, yi_pred - tf.linalg.matvec(Jxi, eta[i,:]) )

        return eta, eta0, m.stack(), P.stack(), y.stack(), H.stack(), Hw.stack(), Cy.stack(), err.stack()

    def LEDH_update_EKF(self, N, m0, P0, y, yhat, Hx, Hw, W, U):
        """Compute the Extended Kalman Gain and return the updated covariance matricies of the particles."""
        P               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        for i in range(N): 
            K           = self.EKF.Gain(P0[i,:,:], Hx[i,:,:], Hw[i,:,:], W, U)
            _, Pi       = self.EKF.Filter(m0[i,:], P0[i,:,:], y, yhat[i,:], Hx[i,:,:], K)
            P           = P.write(i, Pi)
        return P.stack()

    def LEDH_linearize_UKF(self, N, n, model, I1, xprev, Pprev, A, B, V, W, wm, wc, wi, L, U):
        """Predict the pseudo particles, eta and eta0, by UKF using the previous state and its associated particles, xhat and xprev."""
        
        eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        eta             = tf.Variable(tf.zeros((N,n), dtype=tf.float64))

        m               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        P               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        y               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        H               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        Hw              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        Cy              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        CyCross         = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        err             = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)

        for i in range(N): 
            
            Xprev_sp    = self.UKF.SigmaPoints(n, xprev[i,:], Pprev[i,:,:], L)
            X_sp        = Xprev_sp @ tf.transpose(A) 

            mi_pred     = self.UKF.Predict_mean(wm, wi, X_sp) 
            Pi_pred     = self.UKF.Predict_cov(n, wc, wi, X_sp, mi_pred, U, Cov=V)              
            yi_pred, W_pred, Rcross = self.UKF.Predict(model, n, mi_pred, Pi_pred, wm, wc, wi, L, B, W, U)
            Jxi, Jwi    = measurements_Jacobi(model, I1, mi_pred, yi_pred, B)

            eta[i,:].assign(mi_pred)
            eta0[i,:].assign( norm_rvs(n, mi_pred, Pi_pred + U) )        

            m           = m.write(i, mi_pred)
            P           = P.write(i, Pi_pred)
            y           = y.write(i, yi_pred)
            Cy          = Cy.write(i, W_pred)
            CyCross     = CyCross.write(i, Rcross)
            H           = H.write(i, Jxi)
            Hw          = Hw.write(i, Jwi)
            err         = err.write(i, yi_pred - tf.linalg.matvec(Jxi, mi_pred) )        
        
        return eta, eta0, m.stack(), P.stack(), y.stack(), H.stack(), Hw.stack(), Cy.stack(), CyCross.stack(), err.stack() 


    def LEDH_update_UKF(self, N, m0, P0, y, yhat, W_pred, C_pred, U):
        """Compute the Unscented Kalman Gain and return the updated covariance matricies of the particles."""
        P               = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        for i in range(N): 
            K           = self.UKF.Gain(C_pred[i,:,:], W_pred[i,:,:], U)
            _, Pi       = self.UKF.Filter(m0[i,:], P0[i,:,:], W_pred[i,:,:], y, yhat[i,:], K)
            P           = P.write(i, Pi)
        return P.stack()

    def LEDH_flow_dynamics(self, N, n, Lamb, epsilon, I, eta, eta0, Pi, Hi, Ri, err, y, U):
        """Compute and return the flow dynamics of the pseudo particles for migration."""
        move0           = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        move            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        prod            = tf.Variable(tf.zeros((N,), dtype=tf.float64))    
        for i in range(N):                 
            Ai          = Li17eq10(Lamb, Hi[i,:,:], Pi[i,:,:], Ri[i,:,:], U)
            bi          = Li17eq11(I, Lamb, Ai, Hi[i,:,:], Pi[i,:,:], Ri[i,:,:], y, err[i,:], eta[i,:], U)
            move0[i,:].assign( epsilon * (tf.linalg.matvec(Ai, eta0[i,:]) + bi) )
            move[i,:].assign( epsilon * (tf.linalg.matvec(Ai, eta[i,:]) + bi) )
            prod[i].assign( tf.math.abs( tf.linalg.det(I + epsilon * Ai) ) )
        return move0, move, prod

    def LEDH_flow_lp(self, N, eta0, theta, eta1, xprev, y, SigmaX, muy, SigmaY, U):
        """Compute and return the log posterior of the migrated pseudo particles."""
        Lp              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        for i in range(N):  
            Lp = Lp.write(i, tf.math.log(theta[i]) + self.PF.LogLikelihood(y, muy[i,:], SigmaY[i,:,:], U) + self.PF.LogTarget(eta1[i,:], xprev[i,:], SigmaX[i,:,:]) - self.PF.LogImportance(eta0[i,:], xprev[i,:], SigmaX[i,:,:]) )  
        return Lp.stack() 

    def run(self, y, model=None, A=None, B=None, V=None, W=None, N=None, Nstep=None, mu0=None, mu=None, method=None, stepsize=None):
        """Run the LEDH and return the filtered states, X_filtered."""
    
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
        I                           = tf.eye(self.ndims, dtype=tf.float64)
        I1                          = tf.ones((self.ndims,), dtype=tf.float64)
        
        N                           = 1000 if N is None else N
        Np                          = N
        NT                          = N/2   
        method                      = "EKF" if method is None else method 
        if method == "UKF":
            weight0_m, weight0_c, weighti, L = self.UKF.SigmaWeights(self.ndims)
        stepsize                    = 1e-3 if stepsize is None else stepsize     
        Nstep                       = 30 if Nstep is None else Nstep
        Steps                       = tf.constant([stepsize * (1.2*i) for i in range(Nstep+1)], dtype=tf.float64)
        Rates                       = tf.math.cumsum(Steps) 

        P0                          = A @ Sigma0 @ tf.transpose(A) + V
        P_prev                      = tf.Variable(tf.zeros((Np, self.ndims, self.ndims), dtype=tf.float64))
        for i in range(Np):
            P_prev[i,:,:].assign(P0)

        x_filt                      = mu0
        x_prev                      = self.PF.initiate_particles(Np, self.ndims, mu0, P0)
        w_prev                      = tf.ones((Np,), dtype=tf.float64) / Np 

        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        ESS                         = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        Weights                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        JacobiX                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        JacobiW                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)    

        for i in range(self.nTimes): 

            if method == "UKF":        
                eta, eta0, m_pred, P_pred, y_pred, H, Hiw, R, Rcross, el = self.LEDH_linearize_UKF(Np, self.ndims, model, I1, x_prev, P_prev, A, B, V, W, weight0_m, weight0_c, weighti, L, U)
            if method == "EKF":            
                eta, eta0, m_pred, P_pred, y_pred, H, Hiw, R, el = self.LEDH_linearize_EKF(Np, self.ndims, model, I1, x_prev, P_prev, A, B, V, W, mu, U)
                    
        eta1        = eta0
        theta       = tf.Variable(tf.ones((Np,), dtype=tf.float64))
        for j in range(1,Nstep+1): 
            eta1_move, eta_move, theta_prod = self.LEDH_flow_dynamics(Np, self.ndims, Rates[j], Steps[j], I, eta, eta1, P_pred, H, R, el, y[i,:], U)                   
            theta2 = theta * theta_prod
            eta.assign_add(eta_move)
            eta1.assign_add(eta1_move)
            theta.assign(theta2)
            
        lp          = self.LEDH_flow_lp(Np, eta0, theta, eta1, x_prev, y[i,:], P_pred, y_pred, R, U)       
        w_pred      = self.PF.compute_weights(w_prev, lp)
        w_norm      = self.PF.normalize_weights(w_pred)        
        ness        = self.PF.compute_ESS(w_norm)

        if ness < NT: 
            xbar, wbar  = self.PF.multinomial_resample(Np, eta1, w_norm)
            x_filt      = self.PF.compute_posterior(wbar, xbar)
            x_prev      = xbar
            w_prev      = wbar
        else: 
            x_filt      = self.PF.compute_posterior(w_norm, eta1)
            x_prev      = eta1
            w_prev      = w_norm
        
        if method == "UKF":
            P_filt  = self.LEDH_update_UKF(Np, m_pred, P_pred, y[i,:], y_pred, R, Rcross, U) 
        if method == "EKF":    
            P_filt  = self.LEDH_update_EKF(Np,  m_pred, P_pred, y[i,:], y_pred, H, Hiw, W, U)
        
        P_prev          = P_filt
        
        X_filtered                      = X_filtered.write(i, x_filt) 
        ESS                             = ESS.write(i, ness) 
        Weights                         = Weights.write(i, w_norm)    
        JacobiX                         = JacobiX.write(i, H)  
        JacobiW                         = JacobiW.write(i, Hiw)
        
        return X_filtered.stack(), ESS.stack(), Weights.stack(), JacobiX.stack(), JacobiW.stack()



class KernelParticleFlow: 

    def __init__(self, *, nTimes, ndims, nx=None):
        self.nTimes     = nTimes
        self.ndims      = ndims
        self.nx         = nx if nx is not None else ndims
        self.PF         = StandardParticleFilter(nTimes=nTimes, ndims=self.nx)

    def Hu21eq13(self, model, y, ypred, Jx, Jw, W, U):
        R               = measurements_covyHat(model, Jw, W)
        Rinv            = tf.linalg.inv(R + U)  
        return tf.linalg.matvec( tf.transpose(Jx) @ Rinv, y - ypred)  

    def Hu21eq15(self, xpred, x0, Sigma0, U):
        return tf.linalg.matvec( tf.linalg.inv(Sigma0 + U), xpred - x0)

    def KPFF_LP(self, N, nx, n, model, I, x, y, muy, B, W, mu0, Sigma0, U, Uy):
        JxC             = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        JwC             = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        LP              = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        for i in range(N):            
            yi_pred     = measurements_pred(model, n, muy, B, x[i,:], W, Uy)      
            Hx, Hw      = measurements_Jacobi(model, I, x[i,:], yi_pred, B)     
            Hxj         = tf.reshape(tf.linalg.diag_part(Hx), [n,1])
            Jx          = tf.tile(Hxj, [1,nx])          
            lpi         = self.Hu21eq13(model, y, yi_pred, Jx, Hw, W, Uy) - self.Hu21eq15(x[i,:], mu0, Sigma0, U)            
            JxC         = JxC.write(i, Jx)
            JwC         = JwC.write(i, Hw)
            LP          = LP.write(i, lpi)            
        return LP.stack(), JxC.stack(), JwC.stack()

    def KPFF_RKHS(self, N, n, x, Lp, Sigma0):
        In              = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
        for i in range(n): 
            K, Kc       = SE_Cov_div(N, x[:,i], length=Sigma0[i,i])
            for j in range(N): 
                In[j,i].assign( tf.reduce_sum(1/N * ( Lp[j,i] * K[j,:] + Kc[j,:] * K[j,:] )) )
        return In

    def KPFF_flow(self, N, epsilon, integral, Sigma0):
        xadd            = tf.TensorArray(tf.float64, size=N, dynamic_size=True, clear_after_read=False)
        for i in range(N):
            field       = tf.linalg.matvec( Sigma0, integral[i,:] )
            xadd        = xadd.write(i, epsilon * field)
        return xadd.stack()
    
    def run(self, *, y, model=None, A=None, B=None, V=None, W=None, N=None, Nstep=None, mu0=None, mu=None, method=None, stepsize=None):

        model                       = "LG" if model is None else model      
        if model == "SV" and A is None : 
            A                       = tf.eye(self.nx, dtype=tf.float64) * 0.5  
        if model != "SV" and A is None : 
            A                       = tf.eye(self.nx, dtype=tf.float64) 
        if self.nx is not None and B is None:
            B           = tf.ones((self.ndims, self.nx), dtype=tf.float64)
        if self.nx is None and B is None: 
            B           = tf.eye(self.ndims, dtype=tf.float64) 
        V                           = tf.eye(self.nx, dtype=tf.float64) if V is None else V
        W                           = tf.eye(self.ndims, dtype=tf.float64) if W is None else W

        mu0                         = tf.zeros((self.nx,), dtype=tf.float64) if mu0 is None else mu0
        if model == "SV":
            Sigma0                  = V @ tf.linalg.inv(tf.eye(self.nx, dtype=tf.float64) - A @ A)  
        if model != "SV": 
            Sigma0                  = V
        mu                          = tf.zeros((self.ndims,), dtype=tf.float64) if mu is None else mu
        U                           = tf.eye(self.nx, dtype=tf.float64) * 1e-9
        Uy                          = tf.eye(self.ndims, dtype=tf.float64) * 1e-9
        I                           = tf.ones((self.ndims,), dtype=tf.float64)

        N                           = 1000 if N is None else N
        Np                          = N
        method                      = "scalar" if method is None else method 
        stepsize                    = 1e-3 if stepsize is None else stepsize     
        Nstep                       = 30 if Nstep is None else Nstep
        Rates                        = [stepsize] + [stepsize * (1.2*i) for i in range(1,Nstep)]
    
        x_hat                       = mu0

        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        X_part                      = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        X_part2                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        JacobiX                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        JacobiW                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)    

        for i in range(self.nTimes): 
        
            x_prev      = self.PF.initiate_particles(Np, self.nx, x_hat, Sigma0)
            X_part      = X_part.write(i, x_prev)

            for j in range(Nstep): 
                
                grad, Jx, Jw = self.KPFF_LP(Np, self.nx, self.ndims, model, I, x_prev, y[i,:], mu, B, W, x_hat, Sigma0, U, Uy)
                
                if method == "scalar":
                    II  = grad / Np
                if method == "kernel":
                    II  = self.KPFF_RKHS(Np, self.nx, x_prev, grad, Sigma0)

                x_move  = self.KPFF_flow(Np, Rates[j], II, Sigma0)
                x_prev.assign_add(x_move)

            x_hat       = tf.Variable( tf.reduce_mean(x_prev, axis=0) )    

            X_filtered  = X_filtered.write(i, x_hat)
            X_part2     = X_part2.write(i, x_prev)    
            JacobiX     = JacobiX.write(i, Jx)
            JacobiW     = JacobiW.write(i, Jw)

        return X_filtered.stack(), X_part.stack(), X_part2.stack(), JacobiX.stack(), JacobiW.stack() 



