import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

#############################################
#  Squared Exponential Kernels & Divergence # 
#############################################

def SE_kernel(x1, x2, scale, length):
    """The Squared Exponential (SE) kernel given two points, x1 and x2, and the two parameters, scale and length."""
    return (scale**2) * tf.math.exp( - (x1-x2)**2 / (2*length) ) 


def SE_kernel_divC(x1, x2, length):
    """Compute and return the constant part of the derivative of Squared Exponential (SE) kernel given two points, x1 and x2, and the length parameter."""
    return - (x1-x2) / length


def SE_Cov_div(ndims, x, scale=None, length=None):
    """
    Compute the covariance matrix and the constant part of its derivatives using the Squared Exponential (SE) kernel given a vector, x

    Keyword args:
    -------------
    ndims : int32. Dimension of input vector, x. 
    x : tf.Tensor/array/list of float64. Input vector x to be evaluated. 
    scale : float64, optional. The scale parameter for the SE kernel. Defaults to 1.0 if not provided.
    length : float64, optional. The length parameter for the SE kernel. Defaults to 1.0 if not provided.

    Returns:
    --------
    M : tf.Variable of float64 with shape (ndims,ndims). The covariance matrix. 
    Md : tf.Variable of float64 with shape (ndims,ndims). The constant part of the derivatives of the covariance matrix.
    """
    scale           = 1.0 if scale is None else scale 
    length          = 1.0 if length is None else length 

    M               = tf.Variable( tf.zeros((ndims,ndims), dtype=tf.float64) )
    Md              = tf.Variable( tf.zeros((ndims,ndims), dtype=tf.float64) )

    for i in range(ndims): 
        for j in range(i,ndims): 

            v       = SE_kernel(x[i], x[j], scale=scale, length=length)
            M[i,j].assign(v)                     
            M[j,i].assign(v)                     

            vd      = SE_kernel_divC(x[i], x[j], length=length)
            Md[i,j].assign(vd)
            Md[j,i].assign(-vd)

    return M, Md

##############################
#  Gaussian random variables # 
############################## 

def norm_rvs(n, mean, Sigma):
    """
    Generate and return Gaussian random variables using Cholesky decomposition and standard Normal given the expectation and covariance matrix, mean and Sigma.

    Keyword args:
    -------------
    n : int32. Dimension of input expectation vector, mean. 
    mean : tf.Tensor/array/list of float64 with shape (n,). Expectation of the Gaussian random variables to be generated. 
    Sigma : tf.Tensor/array of float64 with shape (n,n). Covariance of the Gaussian random variables to be generated. 

    Returns:
    --------
    x : tf.Tensor of float64 with shape (n,). A Gaussian random vector. 
    """
    chol            = tf.linalg.cholesky(Sigma)
    x_par           = tf.random.normal((n,), dtype=tf.float64)
    x               = tf.linalg.matvec(chol, x_par) + mean
    return x

######################################
#  Linear Gaussian State Space Model # 
######################################

def LG_Jacobi(Ones, B):
    """Compute and return the Jacobian matrices of the linear Gaussian SSM."""
    Jx              = tf.linalg.diag( tf.reduce_sum(B, axis=1) )
    Jw              = tf.linalg.diag( Ones )
    return Jx, Jw

###############################
# Stochastic Volatility Model # 
###############################

def SV_transform(B, x):
    """Compute and return the transformation of the stochastic volatility model given the system states, x."""
    return tf.linalg.matvec(B, tf.math.exp(x/2))   

def SV_measurements(n, m, gx, W, U):
    """Generate and return the measurements of the stochastic volatility model given the transformed system states, x."""
    z               = gx  
    C               = tf.linalg.diag(z) @ W @ tf.linalg.diag(z) 
    y               = norm_rvs(n, m, C + U)
    y2              = tf.where( tf.math.is_nan(y), tf.cast(0.0, tf.float64), y )   
    return y2

def SV_Jacobi(gx, y):
    """Compute and return the pseudo-estimate Jacobian matrices of the stochastic volatility model."""
    Jx              = tf.linalg.diag(y/2)
    Jw              = tf.linalg.diag(gx)
    Jw2             = tf.where( tf.math.logical_and(tf.math.is_inf(Jw), Jw > 0.0), tf.cast(1e9, tf.float64), Jw )
    Jw2             = tf.where( tf.math.logical_and(tf.math.is_inf(Jw), Jw < 0.0), tf.cast(-1e9, tf.float64), Jw2 )
    Jw2             = tf.where( tf.math.is_nan(Jw), tf.cast(0.0, tf.float64), Jw2 )
    return Jx, Jw2

############################
# Location Sensoring Model # 
############################

def sensor_transform(x):
    """Compute and return the transformation of the location sensoring model given the system states, x."""
    sensor1         = (3.5, 0.0) 
    sensor2         = (-3.5, 0.0)
    d               = tf.Variable(tf.zeros((2,), dtype=tf.float64))
    d[0].assign( tf.math.atan((x[1] - sensor1[1]) / (x[0] - sensor1[0])) )
    d[1].assign( tf.math.atan((x[1] - sensor2[1]) / (x[0] - sensor2[0])) )
    return d

def sensor_measurements(gx, W):
    """Generate and return the measurements of the location sensoring model given the transformed system states, x."""
    return norm_rvs(2, gx, W)

def sensor_Jacobi(x): 
    """Compute and return the Jacobian matrices of the location sensoring model."""
    J               = tf.Variable(tf.zeros((2,2), dtype=tf.float64))
    sensor1         = (3.5, 0.0) 
    sensor2         = (-3.5, 0.0)    
    
    z1              = (x[1] - sensor1[1]) / (x[0] - sensor1[0])
    z1part          = 1 / (z1**2 + 1)
    z2              = (x[1] - sensor2[1]) / (x[0] - sensor2[0])
    z2part          = 1 / (z2**2 + 1)
    
    J[0,0].assign( - z1part * (x[1] - sensor1[1]) / ( (x[0] - sensor1[0])**2 ) )
    J[0,1].assign( z1part / (x[0] - sensor1[0]) )    
    J[1,0].assign( - z2part * (x[1] - sensor2[1]) / ( (x[0] - sensor2[0])**2 ) )
    J[1,1].assign( z2part / (x[0] - sensor2[0]) )
    
    return J
    
##############################################
# Non-linear non-Gaussian State Space Model # 
##############################################

def measurements(model, n, m, B, x, W, U=None):
    """Compute and return the measurements of the specified model given the states, x, and other parameters."""
    if model == "LG" : 
        return norm_rvs(n, m + tf.linalg.matvec(B, x), W)
    if model == "SV" :
        U           = tf.zeros((n,n), dtype=tf.float64) if U is None else U
        g           = SV_transform(B, x)
        return SV_measurements(n, m, g, W, U)
    if model == "sensor": 
        g           = sensor_transform(x)
        return sensor_measurements(m + g, W)
        
def measurements_Jacobi(model, ones, x, y, B):
    """Compute and return the (pseudo-estimate) Jacobian matrices of the specified model."""
    if model == "LG" : 
        return LG_Jacobi(ones, B)
    if model == "SV" :
        g           = SV_transform(B, x)
        return SV_Jacobi(g, y)
    if model == "sensor":
        return sensor_Jacobi(x)

def measurements_pred(model, n, m, B, x, W, U=None):
    """Compute and return the predicted measurements given the states, x, and other parameters using the specified model."""
    if model == "LG" : 
        return m + tf.linalg.matvec(B, x)
    if model == "SV" :
        U           = tf.zeros((n,n), dtype=tf.float64) if U is None else U
        g           = SV_transform(B, x)
        return SV_measurements(n, m, g, W, U)
    if model == "sensor":
        return m + sensor_transform(x)
    
def measurements_covyHat(model, Jw, W):
    """Compute and return the (pseudo-estimate) covariance matrix of the specified model."""
    if model == "LG" or model == "sensor": 
        return W
    if model == "SV" :
        return Jw @ W @ tf.transpose(Jw) 

def SSM(nTimes, ndims, n_sparse=None, model=None, A=None, B=None, V=None, W=None, mu0=None, Sigma0=None, muy=None):
    """
    Generate random variables from a state space model (SSM) for the given time and state dimensions, nTimes and ndims.

    Keyword args:
    -------------
    nTimes : int32. Dimension of discrete time step of SSM.
    ndims : int32. Dimension of state space. 
    n_sparse : int32, optional. Dimension of measurements in case of observation sparsity. Defaults to ndims if not provided.
    model: string, optional. The name of the measurement model. Defaults to linear Gaussian "LG" if not provided.
    A : tf.Tensor of float64 with shape (ndims,ndims), optional. The transition matrix. Defaults to identity matrix if not provided.
    B : tf.Tensor of float64 with shape (n_sparse,ndims), optional. The output matrix. Defaults to identity matrix if not provided.
    V : tf.Tensor of float64 with shape (ndims,ndims), optional. The system noise matrix. Defaults to identity matrix if not provided.
    W : tf.Tensor of float64 with shape (n_sparse,n_sparse), optional. The measurement noise matrix. Defaults to identity matrix if not provided.
    mu0 : tf.Tensor of float64 with shape (ndims,), optioanl. The prior mean for initial state. Defaults to vector of zeros if not provided.
    Sigma0 : tf.Tensor of float64 with shape (ndims,ndims). The prior covariance for initial state. Defaults to diagonal matrix if not provided.
    muy : tf.Tensor of float64 with shape (n_sparse,), optioanl. The expectation of the measurements. Defaults to vector of zeros if not provided.

    Returns:
    --------
    X : tf.Variable of float64 with dimension (nTimes,ndims). The states generated by the linear Gaussian SSM.
    Y : tf.Variable of float64 with dimension (nTimes,n_sparse). The measurements generated by the non-linear, non-Gaussian transformation of the states.
    """     
    n_sparse        = ndims if n_sparse is None else n_sparse 
    
    model           = "LG" if model is None else model
    if model == "sensor" and ndims != 2:
        raise ValueError("The state space dimension must be 2 for the location sensoring model.")
        
    if model == "SV" and A is None: 
        A           = tf.eye(ndims, dtype=tf.float64) * 0.5  
    elif model == "SV" and A is not None:
        if tf.reduce_max(A) > 1.0 or tf.reduce_min(A) < -1.0:
            raise ValueError("The matrix A out of range [-1,1].")
    if model != "SV" and A is None: 
        A           = tf.eye(ndims, dtype=tf.float64) 
    
    if n_sparse != ndims and B is None: 
        B           = tf.ones((n_sparse, ndims), dtype=tf.float64) 
    if n_sparse == ndims and B is None:
        B           = tf.eye(ndims, dtype=tf.float64) 

    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(n_sparse, dtype=tf.float64) if W is None else W

    if model == "sensor" and mu0 is None:
        mu0         = tf.constant([3.0,5.0], dtype=tf.float64) 
    else:
        mu0         = tf.zeros((ndims,), dtype=tf.float64)     
        
    if model == "SV" and Sigma0 is None :
        Sigma0      = V @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A)  
    elif model != "SV" and Sigma0 is None: 
        Sigma0      = V
        
    muy             = tf.zeros((n_sparse,), dtype=tf.float64) if muy is None else muy
    
    X               = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    Y               = tf.Variable(tf.zeros((nTimes, n_sparse), dtype=tf.float64))

    x0              = norm_rvs(ndims, mu0, Sigma0)
    y0              = measurements(model, n_sparse, muy, B, x0, W)
    X[0,:].assign(x0)    
    Y[0,:].assign(y0)
    
    for i in range(1, nTimes):

        xi          = norm_rvs(ndims, tf.linalg.matvec(A, X[i-1,:]), V)
        yi          = measurements(model, n_sparse, muy, B, xi, W)  
        
        X[i,:].assign(xi)
        Y[i,:].assign(yi)

    return X, Y
