import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)
from .model import norm_rvs, measurements_pred, measurements_Jacobi, measurements_covyHat



class StandardParticleFilter: 

    def __init__(self, *, nTimes, ndims):
        self.nTimes     = nTimes
        self.ndims      = ndims

    def initiate_particles(self, N, n, mu0, Sigma0):
        """Draw and return a set of initial particles, x0, from the importance distribution, Gaussian with mean and covariance, m0 and Sigma0."""
        x0              = tf.Variable(tf.zeros((N,n), dtype=tf.float64)) 
        for i in range(N):  
            x0[i,:].assign( norm_rvs(n, mu0, Sigma0) )
        return x0 

    def LogImportance(self, x, mu0, Sigma0):
        """Compute and return the log probability of the state vector, x, using the importance distribution given the mean and covariance, mu0 and Sigma0."""
        InvSigma0       = tf.linalg.inv(Sigma0)
        diff            = x - mu0
        return - 1/2 * tf.math.log(tf.linalg.det(Sigma0)) - 1/2 * tf.linalg.tensordot( tf.linalg.matvec(InvSigma0, diff), diff, axes=1) 

    def LogLikelihood(self, y, muy, Sigma, U):
        """Compute and return the log likelihood of the measurement model given the measurement mean and covariance, muy and Sigma. """
        InvCi           = tf.linalg.inv(Sigma + U)
        detCi           = tf.linalg.det(Sigma)
        return - 1/2 * tf.math.log(detCi) - 1/2 * tf.linalg.tensordot( tf.linalg.matvec(InvCi, (y - muy)), (y - muy), axes=1) 

    def LogTarget(self, x, xprev, Sigma0):
        """Compute and return the log probability of the state vector, x, using the target distribution given the previous state and covariance, xprev and Sigma0."""
        InvSigma0       = tf.linalg.inv(Sigma0)
        diff            = x - xprev
        return - 1/2 *  tf.math.log(tf.linalg.det(Sigma0)) - 1/2 * tf.linalg.tensordot( tf.linalg.matvec(InvSigma0, diff), diff, axes=1) 

    def draw_particles(self, N, n, model, I, y, xprev, SigmaX, muy, SigmaY, Sigma0, B, U):
        """Draw particles, xn, from the importance distribution and compute the log posterior probability, Lp."""
        xn              = tf.Variable(tf.zeros((N,n), dtype=tf.float64)) 
        Lp              = tf.Variable(tf.zeros((N,), dtype=tf.float64)) 
        for i in range(N):  
            xi          = norm_rvs(n, xprev[i,:], Sigma0) 
            xn[i,:].assign(xi)
            _, Jw       = measurements_Jacobi(model, I, xprev[i,:], y, B)
            Cy          = measurements_covyHat(model, Jw, SigmaY)
            mu          = measurements_pred(model, n, muy, B, xi, SigmaY, U)
            Lp[i].assign( self.LogLikelihood(y, mu, Cy, U) + self.LogTarget(xi, xprev[i,:], SigmaX) - self.LogImportance(xi, xprev[i,:], Sigma0) )      
        return xn, Lp 

    def compute_weights(self, w0, Lp):
        """Compute and return the importance weights using the previous weights, w0, and the log posterior probability, Lp."""
        return tf.math.exp( tf.math.log(w0) + Lp )

    def normalize_weights(self, w):
        """Normalize the weights, w, and return the normalized weights that sum up to one."""
        return w / tf.reduce_sum(w)

    def compute_ESS(self, w):
        """Compute and return the effective sample size using the normalized weights, w."""
        return 1 / tf.reduce_sum(w**2)

    def multinomial_resample(self, N, x, w):
        """Resample from the set of particles, x, using the weights, w, as multinomial probabilities and return the new set of particles, xbar, and the new weights, wbar."""
        indices         = tfd.Categorical(probs=w).sample(N)
        xbar            = tf.gather(x, indices)
        wbar            = tf.ones((N,), dtype=tf.float64) / N 
        return xbar, wbar 

    def soft_resample(self, N, x, w):
        """Resample from the set of particles, x, using the weights, w, as multinomial probabilities and return the new set of particles, xbar, and the new weights, wbar.""" 
        Lamb            = tf.cast(tfd.Uniform().sample(), tf.float64)
        what            = Lamb * w + (1-Lamb) / N 
        indices         = tfd.Categorical(probs=what).sample(N)
        xbar            = tf.gather(x, indices)
        wbar            = w / what
        return xbar, wbar 

    def compute_posterior(self, w, x):
        """Compute and return the posterior estimates of the state using the weights, w, and the particles, x."""
        return tf.linalg.matvec(tf.transpose(x), w)         

    def run(self, *, y, model=None, A=None, B=None, V=None, W=None, N=None, resample=None, mu0=None, mu=None):
        """Run the Particle Filter and return the filtered states, X_filtered.""" 
        
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

        resample                    = "Multinomial" if resample is None else resample            
        N                           = 1000 if N is None else N
        NT                          = N/2      

        x_prev                      = self.initiate_particles(N, self.ndims, mu0, Sigma0) 
        w_prev                      = tf.Variable(tf.ones((N,), dtype=tf.float64) / N) 

        X_filtered                  = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        ESS                         = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        Weights                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        X_part                      = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
        X_part2                     = tf.TensorArray(tf.float64, size=self.nTimes, dynamic_size=True, clear_after_read=False)
    
        for i in range(self.nTimes):
            
            x_pred, lp              = self.draw_particles(N, self.ndims, model, I, y[i,:], x_prev, V, mu, W, Sigma0, B, U)
            w_pred                  = self.compute_weights(w_prev, lp)
            w_norm                  = self.normalize_weights(w_pred) 
            ness                    = self.compute_ESS(w_norm)
            
            if resample == "Multinomial" and ness < NT: 
                xbar, wbar          = self.multinomial_resample(N, x_pred, w_norm)                
            elif resample == "Soft" and ness < NT:
                xbar, what          = self.soft_resample(N, x_pred, w_norm)
                wbar                = self.normalize_weights(w_norm / what)                
            elif ness >= NT:
                xbar                = x_pred
                wbar                = w_norm
                
            x_filt                  = self.compute_posterior(wbar, xbar)
            x_prev                  = xbar        
            w_prev                  = wbar

            ESS                     = ESS.write(i, ness) 
            X_part                  = X_part.write(i, x_pred)  
            Weights                 = Weights.write(i, w_norm) 
            X_part2                 = X_part2.write(i, x_prev) 
            X_filtered              = X_filtered.write(i, x_filt)  

        return X_filtered.stack(), ESS.stack(), Weights.stack(), X_part.stack(), X_part2.stack()



