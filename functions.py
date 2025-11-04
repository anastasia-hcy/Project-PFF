import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions




def SE_kernel(x1, x2, scale, length):
    return (scale**2) * tf.math.exp( - (x1-x2)**2 / (2*length) )

def SE_Cov(ndims, x, scale=None, length=None):
    scale           = 1.0 if scale is None else scale 
    length          = 1.0 if length is None else length 
    M               = tf.Variable( tf.zeros((ndims,ndims), dtype=tf.float32) )
    for i in range(ndims): 
        for j in range(i,ndims): 
            v       = SE_kernel(x[i], x[j], scale, length) 
            M[i,j].assign(v)       
            M[j,i].assign(v)       
    return M 

def norm_rvs(mean, Sigma):
    n               = len(mean)
    nr, nc          = Sigma.shape 
    if nr != nc or nr != n:
        raise ValueError("Dimensions mismatch.")
    chol            = tf.linalg.cholesky(Sigma)
    x_par           = tf.random.normal((n,), 0, 1)
    x               = tf.linalg.matvec(chol, x_par) + mean
    return x


def LGSSM(nTimes, ndims, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None):
    
    mu0             = tf.zeros((ndims,)) if mu0 is None else mu0
    Sigma0          = tf.eye(ndims) if Sigma0 is None else Sigma0
    A               = tf.eye(ndims) if A is None else A
    B               = tf.eye(ndims) if B is None else B
    V               = tf.eye(ndims) if V is None else V
    W               = tf.eye(ndims) if W is None else W
    
    x0              = norm_rvs(mu0, Sigma0)
    y0              = norm_rvs(x0, W)
    
    X               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))
    Y               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))
    
    X[0,:].assign(x0)    
    Y[0,:].assign(y0)
    for i in tf.range(1, nTimes):
        xi          = norm_rvs(tf.linalg.matvec(A, X[i-1,:]), V)  
        yi          = norm_rvs(tf.linalg.matvec(B, xi), W)      
        X[i,:].assign(xi)
        Y[i,:].assign(yi)
        
    return X, Y




def SV_transform(m, B, x, W, U):
    z = tf.linalg.matvec(B, tf.math.exp(x/2)) 
    try:
        r = norm_rvs(m, z * W * z)
    except tf.errors.InvalidArgumentError:
        r = norm_rvs(m, z * W * z + U)
    return r 

def SVSSM(nTimes, ndims, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None, muy=None):
    
    A               = tf.eye(ndims) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) > 1.0:
        raise ValueError("The matrix A out of range [-1,1].")
    if A is not None and tf.reduce_min(A) < -1.0:
        raise ValueError("The matrix A out of range [-1,1].")
        
    B               = tf.eye(ndims) if B is None else B
    V               = tf.eye(ndims) if V is None else V 
    W               = tf.eye(ndims) if W is None else W
    
    mu0             = tf.zeros((ndims,)) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,)) if muy is None else muy
    u               = tf.eye(ndims) * 1e-5
    
    x0              = norm_rvs(mu0, Sigma0)
    y0              = SV_transform(muy, B, x0, W, u)

    X               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))
    Y               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))

    X[0,:].assign(x0)    
    Y[0,:].assign(y0)
    for i in range(1, nTimes):
        xi          = norm_rvs(tf.linalg.matvec(A, X[i-1,:]), V)
        yi          = SV_transform(muy, B, xi, W, u)     
        X[i,:].assign(xi)
        Y[i,:].assign(yi)

    return X, Y






def Initialize(mu0, Sigma0, A, V):
    x0              = norm_rvs(mu0, Sigma0) 
    P0              = A @ Sigma0 @ tf.transpose(A) + V
    return x0, P0








def KF_Predict(x_prev, P_prev, A, V):
    x               = tf.linalg.matvec(A, x_prev)
    P               = A @ P_prev @ tf.transpose(A) + V
    return x, P 

def KF_Gain(P, B, W):
    M               = P @ tf.transpose(B) 
    Minv            = tf.linalg.inv(B @ M + W) 
    return M * Minv

def KF_Filter(x_prev, P_prev, y_obs, B, K):
    x               = x_prev + tf.linalg.matvec(K, y_obs - tf.linalg.matvec(B, x_prev))
    P               = P_prev - P_prev @ tf.transpose(B) @ tf.transpose(K)
    return x, P

def KalmanFilter(y, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None):
    
    nTimes, ndims   = y.shape 
    mu0             = tf.zeros((ndims,)) if mu0 is None else mu0
    Sigma0          = tf.eye(ndims) if Sigma0 is None else Sigma0
    A               = tf.eye(ndims) if A is None else A
    B               = tf.eye(ndims) if B is None else B
    V               = tf.eye(ndims) if V is None else V
    W               = tf.eye(ndims) if W is None else W
    
    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))
    x_prev, P_prev  = Initialize(mu0, Sigma0, A, V)
    
    for i in range(nTimes):
        x_pred, P_pred     = KF_Predict(x_prev, P_prev, A, V)
        K                  = KF_Gain(P_pred, B, W)
        x_filt, P_filt     = KF_Filter(x_pred, P_pred, y[i,:], B, K)
        x_prev, P_prev     = x_filt, P_filt        
        X_filtered[i,:].assign(x_prev)
        
    return X_filtered







def EKF_Predict(x_prev, P_prev, A, B, V, W, m, U):
    x               = tf.linalg.matvec(A, x_prev)
    P               = A @ P_prev @ tf.transpose(A) + V
    y               = SV_transform(m, B, x, W, U)
    return x, y, P 

def EKF_Jacobi(x, y, B):
    Jx              = tf.linalg.diag(y/2)
    Jw              = tf.linalg.diag(tf.linalg.matvec(B, tf.math.exp(x/2)))
    return Jx, Jw

def EKF_Gain(P, Jx, Jw, W, U):
    Mx              = P @ tf.transpose(Jx) 
    J               = Jx @ Mx + Jw @ W @ tf.transpose(Jw)
    try:
        Minv        = tf.linalg.inv(J)
    except tf.errors.InvalidArgumentError:
        Minv        = tf.linalg.inv(J + U) 
    return Mx @ Minv

def EKF_Filter(x_prev, P_prev, y_obs, y_prev, Jx, K):
    x               = x_prev + tf.linalg.matvec(K, y_obs - y_prev)
    P               = P_prev - P_prev @ tf.transpose(Jx) @ tf.transpose(K)
    return x, P

def ExtendedKalmanFilter(y, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None, muy=None):
    
    nTimes, ndims   = y.shape 
    
    A               = tf.eye(ndims) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) > 1.0:
        raise ValueError("The matrix A out of range [-1,1].")
    if A is not None and tf.reduce_min(A) < -1.0:
        raise ValueError("The matrix A out of range [-1,1].")
    
    B               = tf.eye(ndims) if B is None else B
    V               = tf.eye(ndims) if V is None else V 
    W               = tf.eye(ndims) if W is None else W
    
    mu0             = tf.zeros((ndims,)) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,)) if muy is None else muy
    u               = tf.eye(ndims) * 1e-5

    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))
    x_prev, P_prev  = Initialize(mu0, Sigma0, A, V)
    
    for i in range(nTimes):
        x_pred, y_pred, P_pred      = EKF_Predict(x_prev, P_prev, A, B, V, W, muy, u)
        Jx, Jw                      = EKF_Jacobi(x_pred, y_pred, B)
        K                           = EKF_Gain(P_pred, Jx, Jw, W, u)
        x_filt, P_filt              = EKF_Filter(x_pred, P_pred, y[i,:], y_pred, Jx, K)
        x_prev, P_prev              = x_filt, P_filt        
        X_filtered[i,:].assign(x_prev) 
        
    return X_filtered







def SigmaWeights(ndims, alpha=None, kappa=None, beta=None):
    alpha           = 1.0 if alpha is None else alpha
    kappa           = 3.0 * ndims / 2.0 if kappa is None else kappa
    beta            = 2.0 if beta is None else beta
    
    Lambda          = (alpha**2) * kappa
    
    w0m             = (Lambda - ndims) / Lambda 
    w0c             = w0m + (1 - alpha**2 + beta)
    wi              = 1 / (2*Lambda)
    return w0m, w0c, wi, Lambda

def SigmaPoints(ndims, xhat, Phat, Lambda, u):  
    
    sqrtMat         = tf.linalg.sqrtm(Lambda * Phat)
    SP              = tf.Variable(tf.zeros((2*ndims+1, ndims), dtype=tf.float32))
    SP[0,:].assign(xhat)
    for i in range(1, ndims):
        SP[i,:].assign( xhat + sqrtMat[:,i] )
        SP[ndims+i,:].assign( xhat - sqrtMat[:,i] )
        
    return SP






def UKF_Predict_mean(weight0m, weighti, SP):
    return weight0m * SP[0,:] + tf.reduce_sum( weighti * SP[1:,:], axis=0 )

def UKF_Predict_cov(ndims, weight0c, weighti, SP, mean):
    diffs           = SP - mean 
    cov1            = weight0c * tf.tensordot(diffs[0,:], diffs[0,:], axes=0)
    for i in range(1, ndims):
        cov1        += weighti * tf.tensordot(diffs[i,:], diffs[i,:], axes=0)
    return cov1
    
def UKF_Predict_crosscov(ndims, weight0c, weighti, SP, mean, SP2, mean2):
    diffs          = SP - mean
    diffs2         = SP2 - mean2
    cov            = weight0c * tf.tensordot(diffs[0,:], diffs2[0,:], axes=0)
    for i in range(1, ndims):       
        cov        += weighti * tf.tensordot(diffs[i,:], diffs2[i,:], axes=0)
    return cov

def UKF_Gain(Cn, Wn, u):
    Winv        = tf.linalg.inv(Wn + u)
    return Cn @ Winv

def UKF_Filter(x1, P1, Wn, y_obs, y_pred, K):
    x = x1 + tf.linalg.matvec(K, y_obs - y_pred)
    P = P1 - K @ Wn @ tf.transpose(K)
    return x, P







def UnscentedKalmanFilter(y, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None, muy=None):
    
    nTimes, ndims   = y.shape 
    
    A               = tf.eye(ndims) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) > 1.0:
        raise ValueError("The matrix A out of range [-1,1].")
    if A is not None and tf.reduce_min(A) < -1.0:
        raise ValueError("The matrix A out of range [-1,1].")
    
    B               = tf.eye(ndims) if B is None else B
    V               = tf.eye(ndims) if V is None else V 
    W               = tf.eye(ndims) if W is None else W
    
    mu0             = tf.zeros((ndims,)) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,)) if muy is None else muy
    u               = tf.eye(ndims) * 1e-5
    u2              = tf.eye(ndims*2) * 1e-5
    
    weight0_m, weight0_c, weighti, L = SigmaWeights(ndims) 

    x_pred0         = tf.Variable(tf.zeros((ndims*2,), dtype=tf.float32))
    P_pred0         = tf.Variable(tf.zeros((ndims*2,ndims*2), dtype=tf.float32))
    P_pred0[ndims:ndims*2,ndims:ndims*2].assign(W) 
        
    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float32))
    x_prev, P_prev  = Initialize(mu0, Sigma0, A, V)
    
    for i in range(nTimes):

        Xprev_sp    = SigmaPoints(ndims, x_prev, P_prev, L, u)
        X_sp        = Xprev_sp @ tf.transpose(A) 

        x_pred      = UKF_Predict_mean(weight0_m, weighti, X_sp) 
        P_pred      = UKF_Predict_cov(ndims, weight0_c, weighti, X_sp, x_pred) + V
        
        x_pred0[:ndims].assign(x_pred)
        P_pred0[0:ndims,0:ndims].assign(P_pred)
        
        Xpred_sp    = SigmaPoints(ndims*2, x_pred0, P_pred0, L, u2)
        Y_sp        = tf.math.exp(Xpred_sp[:,:ndims]/2) @ tf.transpose(B) * Xpred_sp[:,ndims:]
        
        y_pred      = UKF_Predict_mean(weight0_m, weighti, Y_sp)
        W_pred      = UKF_Predict_cov(ndims, weight0_c, weighti, Y_sp, y_pred)
        C_pred      = UKF_Predict_crosscov(ndims, weight0_c, weighti, Xpred_sp[:,:ndims], x_pred, Y_sp, y_pred)
         
        K           = UKF_Gain(C_pred, W_pred, u)
            
        x_filt, P_filt              = UKF_Filter(x_pred, P_pred, W_pred, y[i,:], y_pred, K)
        x_prev, P_prev              = x_filt, P_filt        
        X_filtered[i,:].assign(x_prev) 
        
    return X_filtered







