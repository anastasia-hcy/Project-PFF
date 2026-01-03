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

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

import keras
from keras import layers, Loss, ops

from scripts import initiate_particles, soft_resample, LogSumExp

nT              = 100
nD              = 4
Np              = 100
mu0             = tf.zeros(nD, dtype=tf.float64) + 2.0
Sigma0          = tf.eye(nD, dtype=tf.float64) 

w0              = tf.random.uniform((Np,), minval=0, maxval=1, dtype=tf.float64)
w0              = w0 / tf.reduce_sum(w0)
parts           = tf.cast(initiate_particles(Np, nD, mu0, Sigma0), tf.float32)

parts1, w1      = soft_resample(Np, parts, w0)
parts2, w2      = soft_resample(Np, parts1, w1)

w0              = tf.cast(w0, tf.float32)
w1              = tf.cast(w1, tf.float32)
w2              = tf.cast(w2, tf.float32)
parts1          = tf.cast(parts1, tf.float32)
parts2          = tf.cast(parts2, tf.float32)
Sigma0          = tf.cast(Sigma0, tf.float32)


class particles_loss(Loss):
    
    def __init__(self, N, w, SigmaX, reduction='sum_over_batch_size', name='particles_loss'):
        super().__init__(reduction=reduction, name=name)
        self.N = N
        self.w = w
        self.SigmaX = SigmaX
        self.SigmaX_inv = tf.linalg.inv(SigmaX)
        self.SigmaX_det = - tf.math.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )

    def LogSumExp(self, x):
        c               = tf.reduce_max(x) 
        return c + tf.math.log(tf.reduce_sum(tf.math.exp(x - c), axis=0))
    
    def call(self, y_true, y_pred):
        loss            = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for i in range(self.N):
            xi          = y_true[i,:] - y_pred
            xiSum       =  - 1/2 * tf.reduce_sum( xi @ self.SigmaX_inv * xi , axis=-1)      
            loss        = loss.write(i, - self.w[i] * ( self.SigmaX_det + LogSumExp(xiSum) ) )
        return loss.stack().numpy()     
    
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
model.compile(optimizer='adam', loss=particles_loss(Np, w0, Sigma0)) 
model.fit(x=parts, y=parts2, verbose=1)
new_parts = model.predict(parts2)

plt.hist(parts2.numpy().flatten(), bins=30, alpha=0.5, label='Particles')
plt.hist(new_parts.flatten(), bins=30, alpha=0.5, label='Transformed Particles')
plt.show() 
    
    
    
    
    
    
    
    
    
    
    
    
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
model2.compile(optimizer='adam', loss=weightsBP_loss(Np, parts, parts2, Sigma0))
model2.fit(w0, what, epochs=1, verbose=1)

new_w = tf.reshape(model2.call(tf.reshape(what, (-1,1))), (-1,))








##########################
# Neural network example #
########################## 

inputs = keras.Input(shape=(Np,nD))
x = layers.Dense(64, activation="linear", name="dense_1")(inputs)
x = layers.Dense(64, activation="linear", name="dense_2")(x)
outputs = layers.Dense(nD, activation="linear", name="predictions")(x)

custom_loss = soft_weights_loss(Np, tf.cast(w0,tf.float32), tf.cast(Sigma0, tf.float32))
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=custom_loss
)

hist = model.fit(parts, parts2, batch_size=64, epochs=1)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)





def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)

model.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=custom_mean_squared_error)

# We need to one-hot encode the labels to use MSE
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
