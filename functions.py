import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions







def SE_kernel_div(x1, x2, length):
    return - (x1-x2) / length

def SE_kernel(x1, x2, scale=None, length=None):
    scale           = 1.0 if scale is None else scale 
    length          = 1.0 if length is None else length 
    return (scale**2) * tf.math.exp( - (x1-x2)**2 / (2*length) )

def SE_Cov(ndims, x, scale=None, length=None):
    scale           = 1.0 if scale is None else scale 
    length          = 1.0 if length is None else length 
    M               = tf.Variable( tf.zeros((ndims,ndims), dtype=tf.float64) )
    for i in range(ndims): 
        for j in range(i,ndims): 
            v       = SE_kernel(x[i], x[j], scale=scale, length=length)
            M[i,j].assign(v)                     
            M[j,i].assign(v)                     
    return M









def norm_rvs(n, mean, Sigma):
    chol            = tf.linalg.cholesky(Sigma)
    x_par           = tf.random.normal((n,), dtype=tf.float64)
    x               = tf.linalg.matvec(chol, x_par) + mean
    return x








def LGSSM(nTimes, ndims, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None):
    
    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = tf.eye(ndims, dtype=tf.float64) if Sigma0 is None else Sigma0
    A               = tf.eye(ndims, dtype=tf.float64) if A is None else A
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W
    
    x0              = norm_rvs(ndims, mu0, Sigma0)
    y0              = norm_rvs(ndims, x0, W)
    
    X               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    Y               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    
    X[0,:].assign(x0)    
    Y[0,:].assign(y0)
    for i in tf.range(1, nTimes):
        xi          = norm_rvs(ndims, tf.linalg.matvec(A, X[i-1,:]), V)  
        yi          = norm_rvs(ndims, tf.linalg.matvec(B, xi), W)      
        X[i,:].assign(xi)
        Y[i,:].assign(yi)
        
    return X, Y

def SV_transform(n, m, B, x, W, U=None):
    U               = tf.zeros((n,n), dtype=tf.float64) if U is None else U
    z               = tf.linalg.matvec(B, tf.math.exp(x/2)) 
    C               = tf.linalg.diag(z) @ W @ tf.linalg.diag(z) 
    return norm_rvs(n, m, C + U)


def SVSSM(nTimes, ndims, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None, muy=None):
    
    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) > 1.0:
        raise ValueError("The matrix A out of range [-1,1].")
    if A is not None and tf.reduce_min(A) < -1.0:
        raise ValueError("The matrix A out of range [-1,1].")
        
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W
    
    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy

    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9

    x0              = norm_rvs(ndims, mu0, Sigma0)
    y0              = SV_transform(ndims, muy, B, x0, W)

    X               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    Y               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))

    X[0,:].assign(x0)    
    Y[0,:].assign(y0)
    for i in range(1, nTimes):
        xi          = norm_rvs(ndims, tf.linalg.matvec(A, X[i-1,:]), V)
        yi          = SV_transform(ndims, muy, B, xi, W)     
        X[i,:].assign(xi)
        Y[i,:].assign(yi)

    return X, Y

















def KF_Predict(x_prev, P_prev, A, V):
    x               = tf.linalg.matvec(A, x_prev)
    P               = A @ P_prev @ tf.transpose(A) + V
    return x, P 

def KF_Gain(P, B, W):
    M               = P @ tf.transpose(B) 
    Minv            = tf.linalg.inv(B @ M + W) 
    return M @ Minv

def KF_Filter(x_prev, P_prev, y_obs, B, K):
    x               = x_prev + tf.linalg.matvec(K, y_obs - tf.linalg.matvec(B, x_prev))
    P               = P_prev - P_prev @ tf.transpose(B) @ tf.transpose(K)
    return x, P

def KalmanFilter(y, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None):
    
    nTimes, ndims   = y.shape 
    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = tf.eye(ndims, dtype=tf.float64) if Sigma0 is None else Sigma0
    A               = tf.eye(ndims, dtype=tf.float64) if A is None else A
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W
    
    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))

    x_prev          = norm_rvs(ndims, mu0, Sigma0) 
    P_prev          = A @ Sigma0 @ tf.transpose(A) + V

    for i in range(nTimes):
        x_pred, P_pred     = KF_Predict(x_prev, P_prev, A, V)
        K                  = KF_Gain(P_pred, B, W)
        x_filt, P_filt     = KF_Filter(x_pred, P_pred, y[i,:], B, K)
        x_prev, P_prev     = x_filt, P_filt        
        X_filtered[i,:].assign(x_prev)
        
    return X_filtered










def EKF_Predict(x_prev, P_prev, A, V):
    x               = tf.linalg.matvec(A, x_prev)
    P               = A @ P_prev @ tf.transpose(A) + V
    return x, P 

def EKF_Jacobi(x, y, B):
    Jx              = tf.linalg.diag(y/2)
    Jw              = tf.linalg.diag(tf.linalg.matvec(B, tf.math.exp(x/2)))
    return Jx, Jw

def EKF_Gain(P, Jx, Jw, W, U):
    Mx              = P @ tf.transpose(Jx)
    J               = Jx @ Mx + Jw @ W @ tf.transpose(Jw)
    Minv            = tf.linalg.inv(J + U) 
    return Mx @ Minv

def EKF_Filter(x_prev, P_prev, y_obs, y_prev, Jx, K):
    x               = x_prev + tf.linalg.matvec(K, y_obs - y_prev)
    P               = P_prev - P_prev @ tf.transpose(Jx) @ tf.transpose(K) 
    return x, P

def ExtendedKalmanFilter(y, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None, muy=None):
    
    nTimes, ndims   = y.shape 
    
    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) >= 1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    if A is not None and tf.reduce_min(A) <= -1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W
    
    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy
    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9

    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))

    x_prev          = norm_rvs(ndims, mu0, Sigma0) 
    P_prev          = A @ Sigma0 @ tf.transpose(A) + V
    
    for i in range(nTimes):
        x_pred, P_pred              = EKF_Predict(x_prev, P_prev, A, V)
        y_pred                      = SV_transform(ndims, muy, B, x_pred, W, u)
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

def SigmaPoints(ndims, xhat, Phat, Lambda):  
    sqrtMat         = tf.linalg.sqrtm(Lambda * Phat)
    SP              = tf.Variable(tf.zeros((2*ndims+1, ndims), dtype=tf.float64))
    SP[0,:].assign(xhat)
    for i in range(1, ndims):
        SP[i,:].assign( xhat + sqrtMat[:,i] )
        SP[ndims+i,:].assign( xhat - sqrtMat[:,i] ) 
    return SP

def UKF_Predict_mean(weight0m, weighti, SP):
    return weight0m * SP[0,:] + tf.reduce_sum( weighti * SP[1:,:], axis=0 )

def UKF_Predict_cov(ndims, weight0c, weighti, SP, mean, u, Cov=None):
    Cov             = tf.zeros((ndims,ndims), dtype=tf.float64) if Cov is None else Cov 
    diffs           = SP - mean 
    cov1            = weight0c * tf.tensordot(diffs[0,:], diffs[0,:], axes=0)
    for i in range(1, ndims):
        cov1        += weighti * tf.tensordot(diffs[i,:], diffs[i,:], axes=0)
    return cov1 + Cov + u
    
def UKF_Predict_crosscov(ndims, weight0c, weighti, SP, mean, SP2, mean2, u):
    diffs           = SP - mean
    diffs2          = SP2 - mean2
    cov             = weight0c * tf.tensordot(diffs[0,:], diffs2[0,:], axes=0)
    for i in range(1, ndims):       
        cov         += weighti * tf.tensordot(diffs[i,:], diffs2[i,:], axes=0)
    return cov + u

def UKF_Gain(Cn, Wn, u):
    M           =  tf.linalg.inv(Wn + u)
    return Cn @ M

def UKF_Filter(x1, P1, Wn, y_obs, y_pred, K):
    x               = x1 + tf.linalg.matvec(K, y_obs - y_pred)
    P               = P1 - K @ Wn @ tf.transpose(K)
    return x, P

def UnscentedKalmanFilter(y, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None):
    
    nTimes, ndims   = y.shape 
    
    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) >= 1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    if A is not None and tf.reduce_min(A) <= -1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W
    
    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0

    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9
    
    weight0_m, weight0_c, weighti, L = SigmaWeights(ndims) 

    x_pred0         = tf.Variable(tf.zeros((ndims*2,), dtype=tf.float64))
    P_pred0         = tf.Variable(tf.zeros((ndims*2,ndims*2), dtype=tf.float64))
    P_pred0[ndims:ndims*2,ndims:ndims*2].assign(W) 
        
    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    
    x_prev          = norm_rvs(ndims, mu0, Sigma0) 
    P_prev          = A @ Sigma0 @ tf.transpose(A) + V

    for i in range(nTimes):

        Xprev_sp    = SigmaPoints(ndims, x_prev, P_prev, L)
        X_sp        = Xprev_sp @ tf.transpose(A) 

        x_pred      = UKF_Predict_mean(weight0_m, weighti, X_sp) 
        P_pred      = UKF_Predict_cov(ndims, weight0_c, weighti, X_sp, x_pred, u, Cov=V)  
        
        x_pred0[:ndims].assign(x_pred)
        P_pred0[0:ndims,0:ndims].assign(P_pred)
        
        Xpred_sp    = SigmaPoints(ndims*2, x_pred0, P_pred0, L)
        Y_sp        = tf.math.exp(Xpred_sp[:,:ndims]/2) @ tf.transpose(B) * Xpred_sp[:,ndims:]
        
        y_pred      = UKF_Predict_mean(weight0_m, weighti, Y_sp)
        W_pred      = UKF_Predict_cov(ndims, weight0_c, weighti, Y_sp, y_pred, u)
        
        C_pred      = UKF_Predict_crosscov(ndims, weight0_c, weighti, Xpred_sp[:,:ndims], x_pred, Y_sp, y_pred, u) 
        K           = UKF_Gain(C_pred, W_pred, u)
            
        x_filt, P_filt      = UKF_Filter(x_pred, P_pred, W_pred, y[i,:], y_pred, K)
        x_prev, P_prev      = x_filt, P_filt        
        X_filtered[i,:].assign(x_prev) 
        
    return X_filtered













def initiate_particles(N, n, mu0, Sigma0):
    x0              = tf.Variable(tf.zeros((N,n), dtype=tf.float64)) 
    for i in range(N):  
        x0[i,:].assign( norm_rvs(n, mu0, Sigma0) )
    return x0 
















def draw_particles(N, n, y, xprev, SigmaX, muy, SigmaY, Sigma0, U):
    xn              = tf.Variable(tf.zeros((N,n), dtype=tf.float64)) 
    Lp              = tf.Variable(tf.zeros((N,), dtype=tf.float64)) 
    for i in range(N):  
        xi          = norm_rvs(n, xprev[i,:], Sigma0)
        xn[i,:].assign(xi)
        Lp[i].assign( LogLikelihood(xi, y, muy, SigmaY, U) + LogTarget(xi, xprev[i,:], SigmaX) - LogImportance(xi, xprev[i,:], Sigma0) )      
    return xn, Lp 

def LogImportance(x, mu0, Sigma0):
    InvSigma0       = tf.linalg.inv(Sigma0)
    diff            = x - mu0
    return - 1/2 * tf.math.log(tf.linalg.det(Sigma0)) - 1/2 * tf.linalg.tensordot( tf.linalg.matvec(InvSigma0, diff), diff, axes=1) 

def LogLikelihood(x, y, muy, Sigma, U):
    
    xe              = tf.math.exp(x/2)
    Ci              = tf.linalg.diag(xe) @ Sigma @ tf.linalg.diag(xe) + U 
    
    InvCi           = tf.linalg.inv(Ci)
    detCi           = tf.linalg.det(Ci)
    return - 1/2 * tf.math.log(detCi) - 1/2 * tf.linalg.tensordot( tf.linalg.matvec(InvCi, (y - muy)), (y - muy), axes=1) 

def LogTarget(x, xprev, Sigma0):
    InvSigma0       = tf.linalg.inv(Sigma0)
    diff            = x - xprev
    return - 1/2 *  tf.math.log(tf.linalg.det(Sigma0)) - 1/2 * tf.linalg.tensordot( tf.linalg.matvec(InvSigma0, diff), diff, axes=1) 

def compute_weights(w0, Lp):
    return tf.math.exp( tf.math.log(w0) + Lp )

def normalize_weights(w):
    return w / tf.reduce_sum(w)

def compute_ESS(w):
    return 1 / tf.reduce_sum(w**2)

def multinomial_resample(N, x, w):
    dist            = tfd.Categorical(probs=w)
    indices         = dist.sample(N)
    xbar            = tf.gather(x, indices)
    wbar            = tf.ones((N,), dtype=tf.float64) / N 
    return xbar, wbar 


def compute_posterior(w, x):
    return tf.linalg.matvec( tf.transpose(x), w ) 
    # Ex              = tf.Variable(tf.zeros((N,n))) 
    # for i in range(N):
    #     Ex[i,:].assign(w[i] * x[i,:])
    # return tf.reduce_sum(Ex, axis=0)



def ParticleFilter(y, A=None, B=None, V=None, W=None, N=None, resample=None, mu0=None, Sigma0=None, muy=None):
    nTimes, ndims   = y.shape 

    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) >= 1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    if A is not None and tf.reduce_min(A) <= -1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W

    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy

    N               = 1000 if N is None else N
    resample        = False if resample is None else resample
    NT              = N/2
    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9

    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    ESS             = tf.Variable(tf.zeros((nTimes,), dtype=tf.float64))

    x_prev          = initiate_particles(N, ndims, mu0, Sigma0)
    w_prev          = tf.Variable(tf.ones((N,), dtype=tf.float64) / N) 

    for i in range(nTimes):
        
        x_pred, lp  = draw_particles(N, ndims, y[i,:], x_prev, V, muy, W, Sigma0, u)

        w_pred      = compute_weights(w_prev, lp)
        w_norm      = normalize_weights(w_pred)
        
        ness        = compute_ESS(w_norm)
        if resample == True and ness < NT: 
            xbar, wbar  = multinomial_resample(N, x_pred, w_norm)
            x_filt      = compute_posterior(wbar, xbar)
            x_prev      = xbar
        else: 
            x_filt      = compute_posterior(w_norm, x_pred)
            x_prev      = x_pred

        ESS[i].assign(ness)
        X_filtered[i,:].assign(x_filt) 

    return X_filtered, ESS














def Li17eq10(L, H, P, R, U):
    C               = tf.linalg.inv( L * H @ P @ tf.transpose(H) + R + U )
    return -1/2 * P @ tf.transpose(H) @ C @ H 

def Li17eq11(I, L, A, H, P, R, y, ei, e0i, U):
    M1          = (I + 2*L * A) 
    M2          = (I + L * A) @ P @ tf.transpose(H) @ tf.linalg.inv(R + U)
    v           = tf.linalg.matvec(M2, (y - ei)) + tf.linalg.matvec(A, e0i)
    return tf.linalg.matvec(M1, v)














def EDH_linearize_EKF(N, n, xprev, xhat, Pprev, A, B, V, W, muy, U): 
    
    m_pred, P_pred  = EKF_Predict(xhat, Pprev, A, V) 
    
    eta             = tf.Variable(m_pred)
    eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    for i in range(N):
        eta0[i,:].assign( norm_rvs(n, xprev[i,:], P_pred + U) )
    
    y_pred          = SV_transform(n, muy, B, m_pred, W, U) 
    Jx, Jw          = EKF_Jacobi(m_pred, y_pred, B)
    Cy              = Jw @ tf.transpose(Jw)
    err             = y_pred - tf.linalg.matvec(Jx, m_pred) 
    
    return eta, eta0, m_pred, P_pred, y_pred, Jx, Jw, Cy, err 


def EDH_linearize_UKF(N, n, xprev, xhat, Pprev, A, B, V, W, wm, wc, wi, L, U):
        
    Xprev_sp        = SigmaPoints(n, xhat, Pprev, L)
    X_sp            = Xprev_sp @ tf.transpose(A) 
    
    m_pred          = UKF_Predict_mean(wm, wi, X_sp) 
    P_pred          = UKF_Predict_cov(n, wc, wi, X_sp, m_pred, U, Cov=V)  
    
    x_pred0         = tf.Variable(tf.zeros((n*2,), dtype=tf.float64))
    x_pred0[:n].assign(m_pred)
    P_pred0         = tf.Variable(tf.zeros((n*2,n*2), dtype=tf.float64))
    P_pred0[n:n*2,n:n*2].assign(W)     
    P_pred0[0:n,0:n].assign(P_pred)
    
    Xpred_sp        = SigmaPoints(n*2, x_pred0, P_pred0, L)
    Y_sp            = tf.math.exp(Xpred_sp[:,:n]/2) @ tf.transpose(B) * Xpred_sp[:,n:]
    
    y_pred          = UKF_Predict_mean(wm, wi, Y_sp)
    W_pred          = UKF_Predict_cov(n, wc, wi, Y_sp, y_pred, U)
    Jx, _           = EKF_Jacobi(m_pred, y_pred, B)
    err             = y_pred - tf.linalg.matvec(Jx, m_pred) 
    
    eta             = tf.Variable(m_pred) 
    eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    for i in range(N): 
        eta0[i,:].assign( norm_rvs(n, xprev[i,:], P_pred + U) )   
        
    return eta, eta0, m_pred, P_pred, y_pred, Jx, W_pred, err, Xpred_sp, Y_sp



def EDH_flow_dynamics(N, n, Lamb, epsilon, I, e, e0, P, H, R, er, y, U):
    Ai              = Li17eq10(Lamb, H, P, R, U)
    bi              = Li17eq11(I, Lamb, Ai, H, P, R, y, er, e, U)
    move0           = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    for i in range(N): 
        move0[i,:].assign( epsilon * (tf.linalg.matvec(Ai, e0[i,:]) + bi) )  
    move            = epsilon * (tf.linalg.matvec(Ai, e) + bi) 
    return move0, move

def EDH_flow_lp(N, eta0, eta1, xprev, y, SigmaX, muy, SigmaY, U):
    Lp              = tf.Variable(tf.zeros((N,), dtype=tf.float64)) 
    for i in range(N):  
        Lp[i].assign( LogLikelihood(eta1[i,:], y, muy, SigmaY, U) + LogTarget(eta1[i,:], xprev[i,:], SigmaX) - LogImportance(eta0[i,:], xprev[i,:], SigmaX) )  
    return Lp

def EDH(y, A=None, B=None, V=None, W=None, N=None, mu0=None, Sigma0=None, muy=None, method=None, stepsize=None):

    nTimes, ndims   = y.shape 

    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) >= 1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    if A is not None and tf.reduce_min(A) <= -1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W

    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy

    N               = 1000 if N is None else N
    Np              = N

    method          = "UKF" if method is None else method 
    weight0_m, weight0_c, weighti, L = SigmaWeights(ndims)
    
    stepsize        = 1e-3 if stepsize is None else stepsize 
    if stepsize is not None and stepsize >= 1.0:
        raise ValueError("Step-size out of range [0,1).")
    if stepsize is not None and stepsize  < 0.0:
        raise ValueError("Step-size out of range [0,1).")
    
    Rates           = tf.range(0.0, 1.0, stepsize, dtype=tf.float64)
    Nl              = len(Rates)

    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9
    I               = tf.eye(ndims, dtype=tf.float64)

    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    ESS             = tf.Variable(tf.zeros((nTimes,), dtype=tf.float64))
    
    x_filt          = tf.Variable(mu0, dtype=tf.float64)
    P_prev          = A @ Sigma0 @ tf.transpose(A) + V 
    x_prev          = initiate_particles(Np, ndims, x_filt, P_prev)
    w_prev          = tf.Variable(tf.ones((N,), dtype=tf.float64) / N) 

    for i in range(nTimes): 

        if method == "UKF":        
            eta, eta0, m_pred, P_pred, y_pred, H, R, el, xsp, ysp       = EDH_linearize_UKF(Np, ndims, x_prev, x_filt, P_prev, A, B, V, W, weight0_m, weight0_c, weighti, L, u)
        if method == "EKF":            
            eta, eta0, m_pred, P_pred, y_pred, H, Hw, R, el             = EDH_linearize_EKF(Np, ndims, x_prev, x_filt, P_prev, A, B, V, W, muy, u)
            
        eta1        = eta0
        for j in range(Nl): 
            # try:
            eta1_move, eta_move                                         = EDH_flow_dynamics(Np, ndims, Rates[j], stepsize, I, eta, eta1, P_pred, H, R, el, y[i,:], u)  
            # except: 
                # return x_prev, eta0, m_pred, P_pred, y_pred, H, R, el 
            eta.assign_add(eta_move)
            eta1.assign_add(eta1_move)
            
        lp          = EDH_flow_lp(Np, eta0, eta1, x_prev, y[i,:], P_pred, muy, W, u)     
        w_pred      = compute_weights(w_prev, lp)
        w_norm      = normalize_weights(w_pred)        
        ness        = compute_ESS(w_norm)

        x_filt      = compute_posterior(w_norm, eta1)
        x_prev      = eta1

        ESS[i].assign(ness)
        X_filtered[i,:].assign(x_filt) 
        
        if method == "UKF":
            C_pred          = UKF_Predict_crosscov(ndims, weight0_c, weighti, xsp[:,:ndims], m_pred, ysp, y_pred, u) 
            K               = UKF_Gain(C_pred, R, u)
            _, P_filt       = UKF_Filter(m_pred, P_pred, R, y[i,:], y_pred, K)
        if method == "EKF":    
            K               = EKF_Gain(P_pred, H, Hw, W, u)
            _, P_filt       = EKF_Filter(m_pred, P_pred, y[i,:], y_pred, H, K)
        P_prev      = P_filt
        
    return X_filtered, ESS























def LEDH_linearize_EKF(N, n, xprev, Pprev, A, B, V, W, muy, U): 

    eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    eta             = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    m               = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    P               = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))

    y               = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    H               = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    Hw              = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    Cy              = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    err             = tf.Variable(tf.zeros((N,n), dtype=tf.float64))

    for i in range(N): 

        mi_pred, Pi_pred        = EKF_Predict(xprev[i,:], Pprev[i,:,:], A, V) 
        
        eta[i,:].assign(mi_pred)
        eta0[i,:].assign( norm_rvs(n, mi_pred, Pi_pred + U) )
        m[i,:].assign(mi_pred)
        P[i,:,:].assign(Pi_pred)
        
        yi_pred     = SV_transform(n, muy, B, eta[i,:], W, U)
        y[i,:].assign(yi_pred)
        
        Jxi, Jwi    = EKF_Jacobi(eta[i,:], yi_pred, B)
        H[i,:,:].assign( Jxi )
        Hw[i,:,:].assign( Jwi )
        
        Cy[i,:,:].assign( Jwi @ W @ tf.transpose(Jwi) )
        err[i,:].assign( yi_pred - tf.linalg.matvec(Jxi, eta[i,:]) )

    return eta, eta0, m, P, y, H, Hw, Cy, err

def LEDH_update_EKF(N, n, m0, P0, y, yhat, Hx, Hw, W, U):
    P               = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    for i in range(N): 
        K           = EKF_Gain(P0[i,:,:], Hx[i,:,:], Hw[i,:,:], W, U)
        _, Pi       = EKF_Filter(m0[i,:], P0[i,:,:], y, yhat[i,:], Hx[i,:,:], K)
        P[i,:,:].assign(Pi)
    return P

def LEDH_linearize_UKF(N, n, xprev, Pprev, A, B, V, W, wm, wc, wi, L, U):
    
    eta0            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    eta             = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    m               = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    P               = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))

    y               = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    H               = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    Cy              = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    err             = tf.Variable(tf.zeros((N,n), dtype=tf.float64))

    x_pred0         = tf.Variable(tf.zeros((n*2,), dtype=tf.float64))
    P_pred0         = tf.Variable(tf.zeros((n*2,n*2), dtype=tf.float64))
    P_pred0[n:n*2,n:n*2].assign(W) 
    
    XSP             = tf.Variable(tf.zeros((N,2*(n*2)+1, n*2), dtype=tf.float64))    
    YSP             = tf.Variable(tf.zeros((N,2*(n*2)+1, n), dtype=tf.float64))
    
    for i in range(N): 
        
        Xprev_sp    = SigmaPoints(n, xprev[i,:], Pprev[i,:,:], L)
        X_sp        = Xprev_sp @ tf.transpose(A) 

        mi_pred     = UKF_Predict_mean(wm, wi, X_sp) 
        Pi_pred     = UKF_Predict_cov(n, wc, wi, X_sp, mi_pred, U, Cov=V)  
        
        eta[i,:].assign(mi_pred)
        eta0[i,:].assign( norm_rvs(n, mi_pred, Pi_pred + U) )        
        m[i,:].assign(mi_pred)
        P[i,:,:].assign(Pi_pred)
        
        x_pred0[:n].assign(mi_pred)
        P_pred0[0:n,0:n].assign(Pi_pred)
        
        Xpred_sp    = SigmaPoints(n*2, x_pred0, P_pred0, L)
        XSP[i,:,:].assign(Xpred_sp)
        
        Y_sp        = tf.math.exp(Xpred_sp[:,:n]/2) @ tf.transpose(B) * Xpred_sp[:,n:]
        YSP[i,:,:].assign(Y_sp)
        
        yi_pred     = UKF_Predict_mean(wm, wi, Y_sp)
        y[i,:].assign(yi_pred)
        
        W_pred      = UKF_Predict_cov(n, wc, wi, Y_sp, yi_pred, U)
        Jxi, _      = EKF_Jacobi(eta[i,:], yi_pred, B)
        H[i,:,:].assign( Jxi )
        
        Cy[i,:,:].assign( W_pred )
        err[i,:].assign( yi_pred - tf.linalg.matvec(Jxi, eta[i,:]) )        
    
    return eta, eta0, m, P, y, H, Cy, err, XSP, YSP


def LEDH_update_UKF(N, n, m0, P0, y, yhat, Hw, Xsp, Ysp, wc, wi, U):
    P               = tf.Variable(tf.zeros((N,n,n), dtype=tf.float64))
    for i in range(N): 
        C_pred      = UKF_Predict_crosscov(n, wc, wi, Xsp[i,:,:n], m0[i,:], Ysp[i,:,:], yhat[i,:], U) 
        K           = UKF_Gain(C_pred, Hw[i,:,:], U)
        _, Pi       = UKF_Filter(m0[i,:], P0[i,:,:], Hw[i,:,:], y, yhat[i,:], K)
        P[i,:,:].assign(Pi)
    return P

def LEDH_flow_dynamics(N, n, Lamb, epsilon, I, eta, eta0, Pi, Hi, Ri, err, y, U):
    
    move0           = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    move            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    prod            = tf.Variable(tf.zeros((N,), dtype=tf.float64))
    
    for i in range(N):         
        
        Ai          = Li17eq10(Lamb, Hi[i,:,:], Pi[i,:,:], Ri[i,:,:], U)
        bi          = Li17eq11(I, Lamb, Ai, Hi[i,:,:], Pi[i,:,:], Ri[i,:,:], y, err[i,:], eta0[i,:], U)
        
        move0[i,:].assign( epsilon * (tf.linalg.matvec(Ai, eta0[i,:]) + bi) )
        move[i,:].assign( epsilon * (tf.linalg.matvec(Ai, eta[i,:]) + bi) )
        prod[i].assign( tf.math.abs( tf.linalg.det(I + epsilon * Ai) ) )
        
    return move0, move, prod

def LEDH_flow_lp(N, eta0, theta, eta1, xprev, y, SigmaX, muy, SigmaY, U):
    Lp              = tf.Variable(tf.zeros((N,), dtype=tf.float64)) 
    for i in range(N):  
        Lp[i].assign( tf.math.log(theta[i]) + LogLikelihood(eta1[i,:], y, muy, SigmaY, U) + LogTarget(eta1[i,:], xprev[i,:], SigmaX[i,:,:]) - LogImportance(eta0[i,:], xprev[i,:], SigmaX[i,:,:]) )  
    return Lp

def LEDH(y, A=None, B=None, V=None, W=None, N=None, resample=None, mu0=None, Sigma0=None, muy=None, method=None, stepsize=None):

    nTimes, ndims   = y.shape 

    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) >= 1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    if A is not None and tf.reduce_min(A) <= -1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W

    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy

    N               = 1000 if N is None else N
    Np              = N
    
    resample        = False if resample is None else resample
    NT              = Np / 2

    method          = "UKF" if method is None else method 
    weight0_m, weight0_c, weighti, L = SigmaWeights(ndims)
    
    stepsize        = 1e-3 if stepsize is None else stepsize 
    if stepsize is not None and stepsize >= 1.0:
        raise ValueError("Step-size out of range [0,1).")
    if stepsize is not None and stepsize  < 0.0:
        raise ValueError("Step-size out of range [0,1).")
    
    Rates           = tf.range(0.0, 1.0, stepsize, dtype=tf.float64)
    Nl              = len(Rates)

    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9
    I               = tf.eye(ndims, dtype=tf.float64)

    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    ESS             = tf.Variable(tf.zeros((nTimes,), dtype=tf.float64))

    P0              = A @ Sigma0 @ tf.transpose(A) + V
    P_prev          = tf.Variable(tf.zeros((Np,ndims,ndims), dtype=tf.float64))
    for i in range(Np):
        P_prev[i,:,:].assign(P0)

    x_prev          = initiate_particles(Np, ndims, mu0, P0)
    w_prev          = tf.Variable(tf.ones((N,), dtype=tf.float64) / N) 

    for i in range(nTimes): 

        if method == "UKF":        
            eta, eta0, m_pred, P_pred, y_pred, H, R, el, xsp, ysp       = LEDH_linearize_UKF(Np, ndims, x_prev, P_prev, A, B, V, W, weight0_m, weight0_c, weighti, L, u)
        if method == "EKF":            
            eta, eta0, m_pred, P_pred, y_pred, H, Hiw, R, el            = LEDH_linearize_EKF(Np, ndims, x_prev, P_prev, A, B, V, W, muy, u)
              
        eta1        = eta0
        theta       = tf.Variable(tf.ones((Np,), dtype=tf.float64))
        for j in range(Nl): 
            # try:
            eta1_move, eta_move, theta_prod     = LEDH_flow_dynamics(Np, ndims, Rates[j], stepsize, I, eta, eta1, P_pred, H, R, el, y[i,:], u)  
            # except: 
                # return x_prev, eta, eta0, m_pred, P_pred, y_pred, H, R, el 
            
            eta.assign_add(eta_move)
            eta1.assign_add(eta1_move)
            theta2 = theta * theta_prod
            theta.assign(theta2)
            
        lp          = LEDH_flow_lp(Np, eta0, theta, eta1, x_prev, y[i,:], P_pred, muy, W, u)       
        w_pred      = compute_weights(w_prev, lp)
        w_norm      = normalize_weights(w_pred)        
        ness        = compute_ESS(w_norm)

        x_filt      = compute_posterior(w_norm, eta1)        
        x_prev      = eta1

        ESS[i].assign(ness)
        X_filtered[i,:].assign(x_filt) 
        
        if method == "UKF":
            P_filt  = LEDH_update_UKF(Np, ndims, m_pred, P_pred, y[i,:], y_pred, R, xsp, ysp, weight0_c, weighti, u) 
        if method == "EKF":    
            P_filt  = LEDH_update_EKF(Np, ndims, m_pred, P_pred, y[i,:], y_pred, H, Hiw, W, u)
        P_prev      = P_filt
        
    return X_filtered, ESS
















def Hu21eq13(y, ypred, Jx, Jw, W, U):
    Rinv = tf.linalg.inv( Jw @ W @ tf.transpose(Jw) + U )
    yhat = y - ypred
    return tf.linalg.matvec( tf.transpose(Jx) @ Rinv, yhat)

def Hu21eq15(xpred, x0, Sigma0):
    return tf.linalg.matvec( tf.linalg.inv(Sigma0), xpred - x0)

def KPFF_LP(N, n, x, y, muy, B, W, mu0, Sigma0, U):
    LP              = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    for i in range(N):
        yi_pred     = SV_transform(n, muy, B, x[i,:], W, U)
        Hx, Hw      = EKF_Jacobi(x[i,:], yi_pred, B)
        lpi         = Hu21eq13(y, yi_pred, Hx, Hw, W, U) - Hu21eq15(x[i,:], mu0, Sigma0)
        LP[i,:].assign(lpi)
    return LP

def KPFF_RKHS(N, n, x, Lp, Sigma0):
    In              = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    for i in range(n): 
        for j in range(N):
            val = 0.0
            for k in range(N):
                K   = SE_kernel(x[j,i], x[k,i], length=Sigma0[i,i])
                Kd  = SE_kernel_div(x[j,i], x[k,i], length=Sigma0[i,i]) 
                val += 1/N * ( Lp[j,i] * K + Kd ) 
            In[j,i].assign(val) 
    return In

def KPFF_flow(N, n, epsilon, integral, Sigma0):
    xadd            = tf.Variable(tf.zeros((N,n), dtype=tf.float64))
    for i in range(N):
        field       = tf.linalg.matvec( Sigma0, integral[i,:] )
        xadd[i,:].assign( epsilon * field )
    return xadd 


def KernelPFF(y, A=None, B=None, V=None, W=None, N=None, mu0=None, Sigma0=None, muy=None, stepsize=None):

    nTimes, ndims   = y.shape 

    A               = tf.eye(ndims, dtype=tf.float64) * 0.5 if A is None else A
    if A is not None and tf.reduce_max(A) >= 1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    if A is not None and tf.reduce_min(A) <= -1.0:
        raise ValueError("The matrix A out of range (-1,1).")
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W

    mu0             = tf.zeros((ndims,), dtype=tf.float64) if mu0 is None else mu0
    Sigma0          = (V @ V) @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A) if Sigma0 is None else Sigma0
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy

    N               = 1000 if N is None else N
    Np              = N
    
    stepsize        = 1e-3 if stepsize is None else stepsize 
    if stepsize is not None and stepsize >= 1.0:
        raise ValueError("Step-size out of range [0,1).")
    if stepsize is not None and stepsize  < 0.0:
        raise ValueError("Step-size out of range [0,1).")
    
    Nl              = int(1/stepsize)    
    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9

    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))  
      
    for i in range(nTimes): 
        
        x_prev      = initiate_particles(N, ndims, mu0, Sigma0)
        x_hat       = tf.Variable( tf.reduce_mean(x_prev, axis=0) )
        
        for _ in range(Nl): 
            grad    = KPFF_LP(Np, ndims, x_prev, y[i,:], muy, B, W, x_hat, V, u)
            II      = KPFF_RKHS(Np, ndims, x_prev, grad, V)
            x_move  = KPFF_flow(Np, ndims, stepsize, II, V)
            x_prev.assign_add(x_move) 
            
        x_hat       = tf.Variable( tf.reduce_mean(x_prev, axis=0) )
        X_filtered[i,:].assign(x_hat)

    return X_filtered














