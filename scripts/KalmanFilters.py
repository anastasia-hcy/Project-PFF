import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)
from .model import norm_rvs, measurements_pred, measurements_Jacobi

##########################
# Standard Kalman Filter # 
##########################

class KalmanFilter: 
    """
    Compute the estimated states using the Kalman Filters given the measurements. 

    Keyword args:
    -------------
    A : tf.Tensor of float64 with shape (ndims,ndims), optional. The transition matrix. Defaults to identity matrix if not provided.
    B : tf.Tensor of float64 with shape (ndims,ndims), optional. The output matrix. Defaults to identity matrix if not provided.
    V : tf.Tensor of float64 with shape (ndims,ndims), optional. The system noise matrix. Defaults to identity matrix if not provided.
    W : tf.Tensor of float64 with shape (ndims,ndims), optional. The measurement noise matrix. Defaults to identity matrix if not provided.
    
    Returns:
    --------
    X_filtered : tf.Variable of float64 with dimension (nTimes,ndims). The filtered states given by the standard Kalman Filter. 
    """
    def __init__(self, *, nTimes, ndims):
        self.nTimes, self.ndims     = nTimes, ndims

    def Predict(self, x_prev, P_prev, A, V):
        "Predict and return the system state and covariance matrix, x and P, given the previous, x_prev and P_prev."
        x               = tf.linalg.matvec(A, x_prev)
        P               = A @ P_prev @ tf.transpose(A) + V
        return x, P 

    def Gain(self, P, B, W):
        "Compute and return the standard Kalman gain."
        M               = P @ tf.transpose(B) 
        Minv            = tf.linalg.inv(B @ M + W) 
        return M @ Minv
    
    def Filter(self, x_prev, P_prev, y_obs, y_prev, M, K):
        "Filter the predicted system state and covariance matrix, x_prev and P_prev, using the Kalman gain and observed measurments, K and y_obs."
        x               = x_prev + tf.linalg.matvec(K, y_obs - y_prev)
        P               = P_prev - P_prev @ tf.transpose(M) @ tf.transpose(K)
        return x, P
    
    def run(self, y, model=None, A=None, B=None, V=None, W=None, mu0=None, mu=None):       

        model                       = "LG" if model is None else model    
        A                           = tf.eye(self.ndims, dtype=tf.float64) if A is None else A
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

        x_prev                      = norm_rvs(self.ndims, mu0, Sigma0) 
        P_prev                      = A @ Sigma0 @ tf.transpose(A) + V 
        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)

        for i in range(self.nTimes):
            x_pred, P_pred          = self.Predict(x_prev, P_prev, A, V)
            y_pred                  = measurements_pred(model, self.ndims, mu, B, x_pred, W, U)
            K                       = self.Gain(P_pred, B, U)
            x_filt, P_filt          = self.Filter(x_pred, P_pred, y[i,:], y_pred, B, K)

            x_prev, P_prev          = x_filt, P_filt        
            X_filtered              = X_filtered.write(i, x_prev)  

        return X_filtered.stack()



##########################
# Extended Kalman Filter # 
##########################

class ExtendedKalmanFilter: 
    """
    Compute the estimated states using the Extended Kalman Filters given the measurements. 

    Keyword args:
    -------------
    A : tf.Tensor of float64 with shape (ndims,ndims), optional. The transition matrix. Defaults to identity matrix if not provided.
    B : tf.Tensor of float64 with shape (ndims,ndims), optional. The output matrix. Defaults to identity matrix if not provided.
    V : tf.Tensor of float64 with shape (ndims,ndims), optional. The system noise matrix. Defaults to identity matrix if not provided.
    W : tf.Tensor of float64 with shape (ndims,ndims), optional. The measurement noise matrix. Defaults to identity matrix if not provided.
    
    Returns:
    --------
    X_filtered : tf.Variable of float64 with dimension (nTimes,ndims). The filtered states given by the standard Kalman Filter. 
    """
    def __init__(self, *, nTimes, ndims):
        self.nTimes, self.ndims     = nTimes, ndims

    def Predict(self, x_prev, P_prev, A, V):
        "Predict and return the system state and covariance matrix, x and P, given the previous, x_prev and P_prev."
        x               = tf.linalg.matvec(A, x_prev)
        P               = A @ P_prev @ tf.transpose(A) + V
        return x, P 

    def Gain(self, P, Jx, Jw, W, U):
        """Compute and return the Extended Kalman gain."""
        Mx              = P @ tf.transpose(Jx)
        J               = Jx @ Mx + Jw @ W @ tf.transpose(Jw)
        Minv            = tf.linalg.inv(J + U) 
        return Mx @ Minv
    
    def Filter(self, x_prev, P_prev, y_obs, y_prev, M, K):
        "Filter the predicted system state and covariance matrix, x_prev and P_prev, using the Kalman gain and observed measurments, K and y_obs."
        x               = x_prev + tf.linalg.matvec(K, y_obs - y_prev)
        P               = P_prev - P_prev @ tf.transpose(M) @ tf.transpose(K)
        return x, P
    
    def run(self, y, model=None, A=None, B=None, V=None, W=None, mu0=None, mu=None):        
        "Run the Kalman Filter and return the filtered states."
        
        model                       = "LG" if model is None else model        
        A                           = tf.eye(self.ndims, dtype=tf.float64) if A is None else A
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
        I1                          = tf.ones((self.ndims,), dtype=tf.float64)

        x_prev                      = norm_rvs(self.ndims, mu0, Sigma0) 
        P_prev                      = A @ Sigma0 @ tf.transpose(A) + V 
        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)

        for i in range(self.nTimes):
            x_pred, P_pred          = self.Predict(x_prev, P_prev, A, V)
            y_pred                  = measurements_pred(model, self.ndims, mu, B, x_pred, W, U)        
            Jx, Jw                  = measurements_Jacobi(model, I1, x_pred, y_pred, B)
            K                       = self.Gain(P_pred, Jx, Jw, W, U)
            x_filt, P_filt          = self.Filter(x_pred, P_pred, y[i,:], y_pred, Jx, K)

            x_prev, P_prev          = x_filt, P_filt        
            X_filtered              = X_filtered.write(i, x_prev)  

        return X_filtered.stack()


###########################
# Unscented Kalman Filter # 
###########################

class UnscentedKalmanFilter: 
    """
    Compute the estimated states using the Unscented Kalman Filters given the measurements. 
    Keyword args:
    -------------
    A : tf.Tensor of float64 with shape (ndims,ndims), optional. The transition matrix. Defaults to identity matrix if not provided.
    B : tf.Tensor of float64 with shape (ndims,ndims), optional. The output matrix. Defaults to identity matrix if not provided.
    V : tf.Tensor of float64 with shape (ndims,ndims), optional. The system noise matrix. Defaults to identity matrix if not provided.
    W : tf.Tensor of float64 with shape (ndims,ndims), optional. The measurement noise matrix. Defaults to identity matrix if not provided.
    alpha : 
    kappa : 
    beta : 
    Returns:
    --------
    X_filtered : tf.Variable of float64 with dimension (nTimes,ndims). The filtered states given by the standard Kalman Filter. 
    """
    def __init__(self, *, nTimes, ndims, alpha=None, kappa=None, beta=None):
        self.nTimes, self.ndims     = nTimes, ndims        
        alpha                       = 1.0 if alpha is None else alpha
        kappa                       = 3.0 * ndims / 2.0 if kappa is None else kappa
        beta                        = 2.0 if beta is None else beta
        self.alpha, self.kappa, self.beta = alpha, kappa, beta

    def SigmaWeights(self, ndims, alpha=None, kappa=None, beta=None):
        """Compute and return the weights of sigma-points."""
        alpha           = 1.0 if alpha is None else alpha
        kappa           = 3.0 * ndims / 2.0 if kappa is None else kappa
        beta            = 2.0 if beta is None else beta

        Lambda          = (alpha**2) * kappa
        w0m             = (Lambda - ndims) / Lambda 
        w0c             = w0m + (1 - alpha**2 + beta)
        wi              = 1 / (2*Lambda)    
        return w0m, w0c, wi, Lambda

    def SigmaPoints(self, ndims, xhat, Phat, Lambda):  
        """Compute and return a set of 2*ndims+1 sigma-points for each state."""
        sqrtMat         = tf.linalg.cholesky(Lambda * Phat) 
        SP              = tf.Variable(tf.zeros((2*ndims+1, ndims), dtype=tf.float64))
        SP[0,:].assign(xhat)
        for i in range(1,ndims+1):
            SP[i,:].assign( xhat + sqrtMat[:,i-1] )
            SP[ndims+i,:].assign( xhat - sqrtMat[:,i-1] ) 
        return SP

    def UKF_Propagate(self, model, ndims, X_sp, B):
        """Propagate the sigma points of states, X_sp, through the specified measurement model."""
        if model =="LG" :
            return X_sp @ tf.transpose(B)  
        if model == "SV" : 
            return tf.math.exp(X_sp[:,:ndims]/2) @ tf.transpose(B) * X_sp[:,ndims:]

    def UKF_Predict_mean(self, weight0m, weighti, SP):
        """Compute and return the estimated mean of the transformed variables."""
        return weight0m * SP[0,:] + tf.reduce_sum( weighti * SP[1:,:], axis=0 )

    def UKF_Predict_cov(self, ndims, weight0c, weighti, SP, mean, u, Cov=None):
        """Compute and return the estimated covariance of the transformed variables."""
        Cov             = tf.zeros((len(mean),len(mean)), dtype=tf.float64) if Cov is None else Cov 
        diffs           = SP - mean 
        cov1            = weight0c * tf.tensordot(diffs[0,:], diffs[0,:], axes=0)
        for i in range(1,ndims+1):
            cov1        += weighti * tf.tensordot(diffs[i,:], diffs[i,:], axes=0)
            cov1        += weighti * tf.tensordot(diffs[ndims+i,:], diffs[ndims+i,:], axes=0)
        return cov1 + Cov + u
        
    def UKF_Predict_crosscov(self, ndims, weight0c, weighti, SP, mean, SP2, mean2, u):
        """Compute and return the estimated cross-covariance of the transformed variables."""
        diffs           = SP - mean
        diffs2          = SP2 - mean2
        cov             = weight0c * tf.tensordot(diffs[0,:], diffs2[0,:], axes=0)
        for i in range(1,ndims+1):       
            cov         += weighti * tf.tensordot(diffs[i,:], diffs2[i,:], axes=0)
            cov         += weighti * tf.tensordot(diffs[ndims+i,:], diffs2[ndims+i,:], axes=0)
        return cov + u

    def Predict(self, model, ndims, xpred, Ppred, w0m, w0c, wi, Lamb, B, W, U):
        """Compute and return the estimated means, covariance and cross-covariance of the transformed variables with additive noises."""
        
        if model == "LG":
            ndims2  = ndims 
            x_pred0 = xpred 
            P_pred0 = Ppred
            
        elif model == "SV":
            ndims2  = ndims * 2
            x_pred0 = tf.Variable(tf.zeros((ndims2,), dtype=tf.float64))
            P_pred0 = tf.Variable(tf.zeros((ndims2,ndims2), dtype=tf.float64))
            x_pred0[:ndims].assign(xpred)
            P_pred0[:ndims,:ndims].assign(Ppred)
            P_pred0[ndims:ndims2,ndims:ndims2].assign(W) 

        Xpred_sp    = self.SigmaPoints(ndims2, x_pred0, P_pred0, Lamb)
        Y_sp        = self.UKF_Propagate(model, ndims, Xpred_sp, B) 
        y_pred      = self.UKF_Predict_mean(w0m, wi, Y_sp)
        W_pred      = self.UKF_Predict_cov(ndims2, w0c, wi, Y_sp, y_pred, U, Cov=W)
        C_pred      = self.UKF_Predict_crosscov(ndims2, w0c, wi, Xpred_sp[:,:ndims], x_pred0[:ndims], Y_sp, y_pred, U) 
        
        return y_pred, W_pred, C_pred

    def Gain(self, Cn, Wn, u):
        """Compute and return the unscented Kalman gain."""
        M           =  tf.linalg.inv(Wn + u)
        return Cn @ M

    def Filter(self,x1, P1, Wn, y_obs, y_pred, K):
        """Filter the predicted system state and covariance matrix, x1 and P1, using the Kalman gain and observed measurments, K and y_obs."""
        x               = x1 + tf.linalg.matvec(K, y_obs - y_pred)
        P               = P1 - K @ Wn @ tf.transpose(K)
        return x, P

    def run(self, y, model=None, A=None, B=None, V=None, W=None, mu0=None, mu=None):        
        "Run the Kalman Filter and return the filtered states."
        
        model                       = "LG" if model is None else model
        A                           = tf.eye(self.ndims, dtype=tf.float64) if A is None else A
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
        weight0_m, weight0_c, weighti, L = self.SigmaWeights(self.ndims) 

        x_prev                      = norm_rvs(self.ndims, mu0, Sigma0) 
        P_prev                      = A @ Sigma0 @ tf.transpose(A) + V 
        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)

        for i in range(self.nTimes):
            
            Xprev_sp                = self.SigmaPoints(self.ndims, x_prev, P_prev, L)
            X_sp                    = Xprev_sp @ tf.transpose(A) 
            x_pred                  = self.UKF_Predict_mean(weight0_m, weighti, X_sp) 
            P_pred                  = self.UKF_Predict_cov(self.ndims, weight0_c, weighti, X_sp, x_pred, U, Cov=V)  

            y_pred, W_pred, C_pred  = self.Predict(model, self.ndims, x_pred, P_pred, weight0_m, weight0_c, weighti, L, B, W, U)
            K                       = self.Gain(C_pred, W_pred, U)
            x_filt, P_filt          = self.Filter(x_pred, P_pred, W_pred, y[i,:], y_pred, K)

            x_prev, P_prev          = x_filt, P_filt        
            X_filtered              = X_filtered.write(i, x_prev)  

        return X_filtered.stack()
