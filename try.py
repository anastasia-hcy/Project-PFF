#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM"

import os, sys
os.chdir(path)
cwd = os.getcwd()
print(f"Current working directory is: {cwd}")
sys.path.append(cwd)

############# 
# Libraries #
#############

import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

import keras
from keras import layers, Loss, ops
from scripts import initiate_particles, soft_resample, LogSumExp, normalize_weights

nT              = 100
nD              = 4
Np              = 100
mu0             = tf.zeros(nD, dtype=tf.float64) + 2.0
Sigma0          = tf.eye(nD, dtype=tf.float64) 

w0              = tf.random.uniform((Np,), minval=0, maxval=1, dtype=tf.float64)
w0              = normalize_weights(w0) 
parts           = tf.cast(initiate_particles(Np, nD, mu0, Sigma0), tf.float32)

parts1, w1      = soft_resample(Np, parts, w0)
w1              = normalize_weights(w1) 
parts2, w2      = soft_resample(Np, parts1, w1)
w2              = normalize_weights(w2)

w0              = tf.cast(w0, tf.float32)
w1              = tf.cast(w1, tf.float32)
w2              = tf.cast(w2, tf.float32)
parts1          = tf.cast(parts1, tf.float32)
parts2          = tf.cast(parts2, tf.float32)
Sigma0          = tf.cast(Sigma0, tf.float32)


plt.hist(parts2.numpy().flatten(), bins=30, alpha=0.5, label='Particles 2')
plt.hist(parts1.numpy().flatten(), bins=30, alpha=0.5, label='Particles 1')
plt.show() 

class particles_loss(Loss):
    
    def __init__(self, w, SigmaX, reduction="sum_over_batch_size", name='particles_loss'):
        super().__init__(reduction=reduction, name=name)
        self.w = w
        self.SigmaX = SigmaX
        self.SigmaX_inv = tf.linalg.inv(SigmaX)
        self.SigmaX_det = - tf.math.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )

    def LogSumExp(self, x):
        c               = tf.reduce_max(x) 
        return c + tf.math.log(tf.reduce_sum(tf.math.exp(x - c), axis=0))
    
    def call(self, y_true, y_pred):
        xi          = y_true - y_pred
        xiSum       =  - 1/2 * tf.reduce_sum( xi @ self.SigmaX_inv * xi , axis=-1)      
        loss        = - self.w * ( self.SigmaX_det + LogSumExp(xiSum) ) 
        return loss

class ParticleTransform(keras.Model):
    
    def __init__(self, ndims):
        super().__init__()
        self.ndims = ndims
        self.dense_layer1 = layers.Dense(units=32, activation="linear")  
        self.dense_layer2 = layers.Dense(units=64, activation="linear")
        self.output_layer = layers.Dense(units=self.ndims, activation="linear")
        
    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        return self.output_layer(x)


model = ParticleTransform(nD)
model.compile(optimizer='adam', loss=particles_loss(w1, Sigma0))
model.fit(x=parts1, y=parts2, verbose=1)
new_parts = model.predict(parts2)

plt.hist(parts2.numpy().flatten(), bins=30, alpha=0.5, label='Particles')
plt.hist(new_parts.flatten(), bins=30, alpha=0.5, label='Transformed Particles')
plt.show() 
    
    
    
    
x = tf.linspace(1e-3,1,10)
y = tf.linspace(100,1000,10)
    
    
    

class weights_loss(Loss):
    
    def __init__(self, N, reduction="sum_over_batch_size", name='particles_loss'):
        super().__init__(reduction=reduction, name=name)
        self.N = N

    def LogSumExp(self, x):
        c               = tf.reduce_max(x) 
        return c + tf.math.log(tf.reduce_sum(tf.math.exp(x - c), axis=0))
    
    def call(self, y_true, y_pred):
        y_pred_log = tf.math.log(y_pred  + 1e-9)
        loss = LogSumExp(y_pred_log) - tf.math.log(self.N)
        return - loss


class WeightsBP(keras.Model):
       
    def __init__(self):
        super().__init__()
        self.dense_layer1 = layers.Dense(units=32, activation="relu")  
        self.dense_layer2 = layers.Dense(units=64, activation="relu")
        self.output_layer = layers.Dense(units=1, activation="relu")
        
    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        return self.output_layer(x)

model2 = WeightsBP()
model2.compile(optimizer='adam', loss=weights_loss(Np))




