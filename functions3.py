import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

from .functions import initiate_particles, draw_particles, compute_weights, normalize_weights, compute_ESS, compute_posterior
from .functions2 import soft_resample, cost_matrix
import keras
from keras import layers, ops
from keras_tuner import HyperModel

####################################################
# Feed-forward neural network with backpropagation # 
#################################################### 


class SimpleFNN(keras.Model):
    
    def __init__(self, SigmaX, ndim):
        super().__init__()
        self.SigmaX = SigmaX
        self.SigmaX_inv = ops.linalg.inv(SigmaX)
        self.SigmaX_det = ops.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )
        self.fnn = keras.Sequential([
            layers.Dense(32, activation="tanh"), 
            layers.Dense(64, activation="relu"), 
            layers.Dense(32), 
            layers.Dense(ndim)
        ])
        
    def LogSumExp(self, x):
        c               = ops.max(x) 
        return c + ops.log(ops.sum(ops.exp(x - c), axis=0))
        
    def call(self, x):   
        return self.fnn(x[0])

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:      
            y_pred = self(x)  
            diff = y_pred - y
            diffSum =  - 1/2 * ops.sum( diff @ self.SigmaX_inv * diff, axis=-1) 
            loss =  - ops.sum(x[1] * ( - self.SigmaX_det + self.LogSumExp(diffSum)))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

#####################################################
# Variational autoencoder with multi-head attention #
##################################################### 

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
            layers.Dense(embedding_dim), 
            layers.Dense(embedding_dim) 
        ])
        self.mean = layers.Dense(embedding_dim)
        self.log_var = layers.Dense(embedding_dim)
        self.sample = Sampling()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        out = self.ffn1(x)
        attn_output = self.attention(out, out)
        out1 = self.layernorm1(attn_output)  
        ffn_output = self.ffn2(out1)
        output = self.layernorm2(ffn_output)
        z_mu = self.mean(output) 
        z_logvar = self.log_var(output)
        z = self.sample((z_mu, z_logvar))
        return (z_mu, z_logvar, z)
    
class DecoderMHPT(layers.Layer):
    
    def __init__(self, embedding_dim, num_heads, ndim):
        super(DecoderMHPT, self).__init__()
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = keras.Sequential([
            layers.Dense(embedding_dim), 
            layers.Dense(embedding_dim)  
        ])
        self.output_layer = layers.Dense(units=ndim)
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
        return output 
    
class MultiHeadPT(keras.Model):
    
    def __init__(self, h, k, SigmaX, ndims):
        super().__init__()
        self.SigmaX = SigmaX
        self.SigmaX_inv = ops.linalg.inv(SigmaX)
        self.SigmaX_det = ops.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )
        self.encoder = EncoderMHPT(embedding_dim=k, num_heads=h)
        self.decoder = DecoderMHPT(embedding_dim=k, num_heads=h, ndim=ndims)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def LogSumExp(self, x):
        c               = ops.max(x) 
        return c + ops.log(ops.sum(tf.math.exp(x - c), axis=0))
    
    @property
    def metrics(self):
        return (
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        )
        
    def call(self, x):   
        inputs = x[0] 
        z_mu, z_lv, z = self.encoder(inputs)
        reconstruction = self.decoder(z*0.0, z)        
        return reconstruction, z_mu, z_lv

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:      
            
            reconstruction, z_mu, z_lv = self(x)  
            diff = reconstruction - y
            diffSum =  - 1/2 * ops.sum( diff @ self.SigmaX_inv * diff, axis=-1) 
            reconstruction_loss =  - ops.sum(x[1] * ( - self.SigmaX_det + self.LogSumExp(diffSum)))
            
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
        
################################
# Differential Particle Filter #
################################


def DifferentialParticleFilter(y, model=None, A=None, B=None, V=None, W=None, N=None, backpropagation=None, mu0=None, Sigma0=None, muy=None):
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
    backpropagation : string, optional. Defaults to simple feed-forward neural network. 
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
    backpropagation = "Simple" if backpropagation is None else backpropagation
    N               = 1000 if N is None else N
    NT              = N/2
    u               = tf.eye(ndims, dtype=tf.float64) * 1e-9
    I               = tf.ones((ndims,), dtype=tf.float64)
    
    w_multi         = tf.ones((N,), dtype=tf.float64) / N 
    _, _, _, _, parts2_train = ParticleFilter(y, N=N, model=model)
    _, _, weights_train, parts_train, _ = ParticleFilter(y, N=N, model=model, resample="Soft")
    
    if backpropagation == "Simple": 
        Transform = SimpleFNN(SigmaX=tf.cast(Sigma0, dtype=tf.float32), ndim=ndims)
    elif backpropagation == "Multi-Head": 
        Transform = MultiHeadPT(h=8, k=32, SigmaX=tf.cast(Sigma0, dtype=tf.float32), ndims=ndims)   
    Transform.compile(optimizer=keras.optimizers.SGD(1e-3))
    Transform.fit(x=(parts_train, weights_train), y=parts2_train)
        
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
        
        if ness < NT:       
            _, what     = soft_resample(N, x_pred, w_norm)
            wbar        = normalize_weights(what)
        elif ness >= NT: 
            wbar        = w_norm
                    
        inputs      = (tf.reshape(x_pred,(1,N,ndims)), tf.reshape(w_norm,(1,N)))
        if backpropagation == "Simple": 
            xhat    = Transform.predict(x=inputs)            
        elif backpropagation == "Multi-Head":
            xhat, _, _  = Transform.predict(x=inputs)                
        xbar        = tf.cast(tf.reshape(xhat, (N,ndims)), dtype=tf.float64)
            
        x_prev      = xbar        
        w_prev      = wbar        
        x_filt      = compute_posterior(w_multi, xbar)
        X_part2[i,:,:].assign(x_prev)
        X_filtered[i,:].assign(x_filt)

    return X_filtered, ESS, Weights, X_part, X_part2 













#######################################
# Feed-forward neural network with OT # 
#######################################

class SimpleFNN_OT(HyperModel):
    
    def __init__(self, SigmaX, ndim):
        super().__init__()
        self.ndim   = ndim
        self.SigmaX = SigmaX
        self.SigmaX_inv = ops.linalg.inv(SigmaX)
        self.SigmaX_det = ops.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )
        self.fnn = keras.Sequential([
            layers.Dense(32, activation="tanh"), 
            layers.Dense(64, activation="relu"), 
            layers.Dense(32), 
            layers.Dense(ndim)
        ])
        
    def LogSumExp(self, x):
        c               = ops.max(x) 
        return c + ops.log(ops.sum(ops.exp(x - c), axis=0))
    
    def Sinkhorn(self, a, b, C, reg, num_iter):
        log_K = - C / reg
        
        u = tf.ones_like(a)
        v = tf.ones_like(b)
        log_u = tf.zeros_like(a) 
        log_v = tf.zeros_like(b) 
        
        for _ in range(num_iter):
            log_u = tf.math.log(a) - self.LogSumExp(log_K + log_v[:,tf.newaxis]) 
            log_v = tf.math.log(b) - self.LogSumExp(log_K + log_u[:, tf.newaxis] + log_v[:, tf.newaxis])
            u = tf.math.exp(log_u)
            v = tf.math.exp(log_v)
            
        P = tf.linalg.diag(u) @ tf.math.exp(log_K) @ tf.linalg.diag(v)
        return P
            
    def build(self, hp):
        inputs = keras.Input(shape=(self.ndim,))     
        x = self.fnn(inputs)
        return keras.Model(inputs=inputs, outputs=x)
    
    def fit(self, hp, model, x, y, validation_data, callbacks=None):
        
        inds = ops.reshape(ops.arange(len(x[1]), dtype=tf.float32), (-1,1))
        C = cost_matrix(inds,inds)
        
        niter = hp.Int("num_iter", min_value=100, max_value=1000, default=100)
        reg = hp.Float("reg", min_value=1e-3, max_value=1.0, default=0.1)
        POT = self.Sinkhorn(x[1], x[2], C, reg=reg, num_iter=niter)   
        
        optimizer = keras.optimizers.SGD(1e-3)
        epoch_loss_metric = keras.metrics.MeanSquaredError()
        
        def run_train_step(x, y, P):
            with tf.GradientTape() as tape:
                y_pred = model(P @ x[0])
                diff = y_pred - y
                diffSum =  - 1/2 * ops.sum( diff @ self.SigmaX_inv * diff, axis=-1) 
                loss =  - ops.sum(x[1] * ( - self.SigmaX_det + self.LogSumExp(diffSum)))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        def run_val_step(x, y, P):
            y_pred = model(P @ x[0])
            diff = y_pred - y
            diffSum = - 1/2 * ops.sum( diff @ self.SigmaX_inv * diff, axis=-1) 
            loss = - ops.sum(x[1] * ( - self.SigmaX_det + self.LogSumExp(diffSum)))
            epoch_loss_metric.update_state(loss)

        for callback in callbacks:
            callback.set_model(model)
            
        best_epoch_loss = float("inf")

        for epoch in range(10):
            print(f"Epoch: {epoch}")
            
            run_train_step(x, y, POT)
            for xv, yv in validation_data:
                run_val_step(xv, yv, POT)

            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_state()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        return best_epoch_loss
