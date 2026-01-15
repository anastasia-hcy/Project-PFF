import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

from .functions import initiate_particles, draw_particles, compute_weights, normalize_weights, compute_ESS, multinomial_resample, compute_posterior, ParticleFilter
from .functions2 import soft_resample, ot_resample, cost_matrix
import keras
from keras import layers, ops



def DifferentialParticleFilter(y, model=None, A=None, B=None, V=None, W=None, N=None, resample=None,  backpropagation=False, mu0=None, Sigma0=None, muy=None):
    """
    Compute the estimated states using the standard Particle Filter given the measurements. 

    Keyword args:
    -------------
    y : tf.Variable of float64 with dimension (nTimes,ndims). The observed measurements. 
    model: string, optional. The name of the measurement model. Defaults to linear Gaussian "LG" if not provided.A : tf.Tensor of float64 with shape (ndims,ndims), optional. The transition matrix. Defaults to diagonal matrix of 0.5 if not provided.
    A : tf.Tensor of float64 with shape (ndims,ndims), optional. The transition matrix. Defaults to diagonal matrix if not provided.
    B : tf.Tensor of float64 with shape (ndims,ndims), optional. The output matrix. Defaults to identity matrix if not provided.
    V : tf.Tensor of float64 with shape (ndims,ndims), optional. The system noise matrix. Defaults to identity matrix if not provided.
    W : tf.Tensor of float64 with shape (ndims,ndims)., optional. The measurement noise matrix. Defaults to identity matrix if not provided.
    N : int32, optional. Defaults to 1000 if not provided.
    resample : string, optional. The resampling scheme. Defaults to Multinomial. 
    mu0 : tf.Tensor of float64 with shape (ndims,), optioanl. The prior mean for initial state. Defaults to zeros if not provided.
    Sigma0 : tf.Tensor of float64 with shape (ndims,ndims). The prior covariance for initial state. Defaults to predefined covariance if not provided.
    muy : tf.Tensor of float64 with shape (ndims,), optioanl. The scalar means of the measurements. Defaults to zeros if not provided.
    
    Returns:
    --------
    X_filtered : tf.Variable of float64 with dimension (nTimes,ndims). The filtered states given by the standard Particle Filter. 
    ESS : tf.Variable of float64 with dimension (nTimes,). The effective sample sizes before resampling. 
    Weights : tf.Variable of float64 with dimension (nTimes,N). The normalized weights of particles before resampling.
    X_part : tf.Variable of float64 with dimension (nTimes,N,ndims). The particles before resampling.
    X_part2 : tf.Variable of float64 with dimension (nTimes,N,ndims). THe particles after resampling.
    """
    
    nTimes, ndims   = y.shape 
    
    model           = "LG" if model is None else model
    if model == "sensor" and ndims != 2:
        raise ValueError("The state space dimension must be 2 for the location sensoring model.")
    
    if model == "SV" and A is None : 
        A           = tf.eye(ndims, dtype=tf.float64) * 0.5  
    if model == "SV" and A is not None :
        if tf.reduce_max(A) > 1.0:
            raise ValueError("The matrix A out of range [-1,1].")
        if tf.reduce_min(A) < -1.0:
            raise ValueError("The matrix A out of range [-1,1].")
    if model != "SV" and A is None : 
        A           = tf.eye(ndims, dtype=tf.float64) 
    
    B               = tf.eye(ndims, dtype=tf.float64) if B is None else B
    V               = tf.eye(ndims, dtype=tf.float64) if V is None else V 
    W               = tf.eye(ndims, dtype=tf.float64) if W is None else W
    

    mu0             = tf.zeros((ndims,), dtype=tf.float64)  
    if model == "SV" and Sigma0 is None :
        Sigma0      = V @ tf.linalg.inv(tf.eye(ndims, dtype=tf.float64) - A @ A)  
    if model != "SV" and Sigma0 is None: 
        Sigma0      = V
        
    muy             = tf.zeros((ndims,), dtype=tf.float64) if muy is None else muy
    resample        = "Multinomial" if resample is None else resample
    N               = 1000 if N is None else N
    NT              = N/2
    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9
    I               = tf.ones((ndims,), dtype=tf.float64)
    
    if resample == "OT":
        inds        = tf.reshape(tf.range(N, dtype=tf.float32), (-1,1))
        C           = cost_matrix(inds,inds)
        
    if backpropagation == "Multi-Head": 
        _, _, train_weights, _, train_particles2 = ParticleFilter(y, N=N)
        DPT = MultiHeadPT(h=8, k=32, SigmaX=tf.cast(V, dtype=tf.float32))
        DPT.compile(optimizer=keras.optimizers.SGD(1e-3))
        DPT.fit(x=train_particles2, y=train_weights, epochs=10)
        
    X_filtered      = tf.Variable(tf.zeros((nTimes, ndims), dtype=tf.float64))
    ESS             = tf.Variable(tf.zeros((nTimes,), dtype=tf.float64))
    Weights         = tf.Variable(tf.zeros((nTimes,N), dtype=tf.float64))
    X_part          = tf.Variable(tf.zeros((nTimes,N,ndims), dtype=tf.float64))
    X_part2         = tf.Variable(tf.zeros((nTimes,N,ndims), dtype=tf.float64))
    
    x_prev          = initiate_particles(N, ndims, mu0, Sigma0)
    w_prev          = tf.Variable(tf.ones((N,), dtype=tf.float64) / N) 

    for i in range(nTimes):
        
        x_pred, lp  = draw_particles(N, ndims, model, I, y[i,:], x_prev, V, muy, W, Sigma0, B, u)
        w_pred      = compute_weights(w_prev, lp)
        w_norm      = normalize_weights(w_pred) 
        
        X_part[i,:,:].assign(x_pred)
        Weights[i,:].assign(w_norm)
        
        ness        = compute_ESS(w_norm)
        ESS[i].assign(ness)
        
        if resample == "Soft" and ness < NT:
            xbar, what  = soft_resample(N, x_pred, w_norm) 
            wbar        = normalize_weights(what)
        
        elif resample == "OT" and ness < NT:
            xbar, what, pot = ot_resample(N, x_pred, w_norm, C)
            wbar        = normalize_weights(what)
            
        elif ness >= NT:
            xbar        = x_pred
            wbar        = w_norm
        
        if backpropagation == "Multi-Head": 
            xhat        = tf.cast(xbar, dtype=tf.float32)
            xbar, _, _  = DPT.predict(xhat)
            xbar        = tf.cast(xbar, tf.float64)
            wbar        = tf.ones((N,), dtype=tf.float64) / N 
            
        x_filt      = compute_posterior(wbar, xbar)
        x_prev      = xbar        
        w_prev      = wbar
        X_part2[i,:,:].assign(x_prev)
        X_filtered[i,:].assign(x_filt)

    return X_filtered, ESS, Weights, X_part, X_part2 




class Sampling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.seed_generator = keras.random.SeedGenerator(123)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class EncoderMHPT(layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(EncoderMHPT, self).__init__()
        self.ffn1 = layers.Dense(embedding_dim)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn2 = keras.Sequential([
            layers.Dense(embedding_dim),  # Feed Forward layer
            layers.Dense(embedding_dim)  # Output to match embedding dimension
        ])
        self.mean = layers.Dense(embedding_dim)
        self.log_var = layers.Dense(embedding_dim)
        self.sample = Sampling()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        out = self.ffn1(x)
        attn_output = self.attention(out, out)
        out1 = self.layernorm1(x + attn_output)  
        ffn_output = self.ffn2(out1)
        output = self.layernorm2(ffn_output)
        z_mu = self.mean(output) 
        z_logvar = self.log_var(output)
        z = self.sample((z_mu, z_logvar))
        return (z_mu, z_logvar, z)
    
class DecoderMHPT(layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(DecoderMHPT, self).__init__()
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = keras.Sequential([
            layers.Dense(embedding_dim),  # Feed Forward layer
            layers.Dense(embedding_dim)  # Output to match embedding dimension
        ])
        self.output_layer = layers.Dense(units=1)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output):
    
        attn1 = self.attention1(x, x)
        out1 = self.layernorm1(x + attn1)
        
        attn2 = self.attention2(out1, enc_output)
        out2 = self.layernorm2(enc_output + attn2) 
        
        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(ffn_output)
        
        output = self.output_layer(out3)
        return ops.squeeze(output, axis=-1) 
    
class MultiHeadPT(keras.Model):
    
    def __init__(self, h, k, SigmaX):
        super().__init__()
        self.k = k
        self.SigmaX = SigmaX
        self.SigmaX_inv = ops.linalg.inv(SigmaX)
        self.SigmaX_det = ops.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )
        self.encoder = EncoderMHPT(embedding_dim=k, num_heads=h)
        self.decoder = DecoderMHPT(embedding_dim=k, num_heads=h)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def LogSumExp(self, x):
        c               = tf.reduce_max(x) 
        return c + tf.math.log(tf.reduce_sum(tf.math.exp(x - c), axis=0))
    
    @property
    def metrics(self):
        return (
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        )
        
    def call(self, x):   
        inputs = ops.expand_dims(x, axis=-1)
        z_mu, z_lv, z = self.encoder(inputs)
        reconstruction = self.decoder(z, z)        
        return reconstruction, z_mu, z_lv

    def train_step(self, data):
        x, y = data      
        with tf.GradientTape() as tape:      
                  
            reconstruction, z_mu, z_lv = self(x)         
             
            diff = reconstruction - x
            diffSum = ops.log(y) - 1/2 * tf.reduce_sum( diff @ self.SigmaX_inv * diff, axis=-1) 
            reconstruction_loss =  - ops.mean(- self.SigmaX_det + self.LogSumExp(diffSum))
            
            kl_loss = - 0.5 * (1 + z_lv - ops.square(z_mu) - ops.exp(z_lv))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=-1))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
        
        
