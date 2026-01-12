#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/" 
pathdat             = "C:/Users/anastasia/MyProjects/JPMorgan/data/" 

import os, sys
os.chdir(path)
cwd = os.getcwd()
print(f"Current working directory is: {cwd}")
sys.path.append(cwd)

############# 
# Libraries #
#############

import matplotlib.pyplot as plt
import pickle as pkl

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

import keras
from keras import layers, Loss, ops
from scripts import ParticleFilter, normalize_weights, compute_posterior


with open(pathdat+"data_sim.pkl", 'rb') as file:
    data        = pkl.load(file)    
X1              = data['LG_States']
Y1              = data['LG_Obs']

Np              = 100
nT, nD          = X1.shape

X_PF_1, ess_PF_1, weights_PF_1, particles_PF_1, particles2_PF_1 = ParticleFilter(Y1, N=Np)
# X_PF_S_1, ess_PF_S_1, weights_PF_S_1, particles_PF_S_1, particles2_PF_S_1 = ParticleFilter(Y1, N=Np, resample="Soft")
# X_PF_OT_1, ess_PF_OT_1, weights_PF_OT_1, particles_PF_OT_1, particles2_PF_OT_1 = ParticleFilter(Y1, N=Np, resample="OT")

fig, ax = plt.subplots(figsize=(6,4))
for i in range(nD):
    plt.plot(X1[:,i], linewidth=1, alpha=0.75, color='green') 
    # plt.plot(X_PF_S_1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='orange') 
    plt.plot(X_PF_1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='red') 
    # plt.plot(X_PF_OT_1[:,i], linewidth=1, alpha=0.5, linestyle='dashed', color='yellow') 
plt.show() 



class particles_loss(Loss):
    
    def __init__(self, w, SigmaX, reduction="sum_over_batch_size", name='particles_loss'):
        super(particles_loss, self).__init__(reduction=reduction, name=name)
        self.w = w
        self.SigmaX = SigmaX
        self.SigmaX_inv = tf.linalg.inv(SigmaX)
        self.SigmaX_det = - tf.math.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )

    def LogSumExp(self, x):
        c               = tf.reduce_max(x) 
        return c + tf.math.log(tf.reduce_sum(tf.math.exp(x - c), axis=0))
    
    def call(self, y, y_pred):
        xi          = y - y_pred
        xiSum       =  - 1/2 * tf.reduce_sum( xi @ self.SigmaX_inv * xi , axis=-1)      
        loss        = - self.w * ( self.SigmaX_det + self.LogSumExp(xiSum) ) 
        return loss




class Sampling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.seed_generator = keras.random.SeedGenerator(123)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class EncoderPT(layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(EncoderPT, self).__init__()
        self.ffn1 = layers.Dense(embedding_dim, activation='tanh')
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn2 = keras.Sequential([
            layers.Dense(embedding_dim, activation='tanh'),  # Feed Forward layer
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
    
class DecoderPT(layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(DecoderPT, self).__init__()
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = keras.Sequential([
            layers.Dense(embedding_dim, activation='tanh'),  # Feed Forward layer
            layers.Dense(embedding_dim)  # Output to match embedding dimension
        ])
        self.output_layer = layers.Dense(units=1)
        self.reshape_layer = layers.Reshape((-1,))
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
        return self.reshape_layer(output)
    
class ParticleTransform(keras.Model):
    
    def __init__(self, h, k, N, SigmaX):
        super().__init__()
        self.N = N
        self.k = k
        self.SigmaX = SigmaX
        self.SigmaX_inv = tf.linalg.inv(SigmaX)
        self.SigmaX_det = tf.math.log( tf.math.sqrt(tf.linalg.det(SigmaX)) )
        self.encoder = EncoderPT(embedding_dim=k, num_heads=h)
        self.decoder = DecoderPT(embedding_dim=k, num_heads=h)
        self.reshape_layer = layers.Reshape((-1,1))
        
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
        inputs = self.reshape_layer(x)
        z_mu, z_lv, z = self.encoder(inputs)
        reconstruction = self.decoder(z*0.0, z)        
        return reconstruction, z_mu, z_lv

    def train_step(self, data):
        x, y = data      
        with tf.GradientTape() as tape:      
                  
            reconstruction, z_mu, z_lv = self(x)          
            diff = reconstruction - x
            diffSum = - 1/2 * tf.reduce_sum( diff @ self.SigmaX_inv * diff , axis=-1) 
            reconstruction_loss = - y * ( self.SigmaX_det + self.LogSumExp(diffSum) ) 
            
            kl_loss = -0.5 * (1 + z_lv - ops.square(z_mu) - ops.exp(z_lv))
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
        
    # def call(self, inputs):
    #     inputs = self.reshape_layer(inputs)
    #     x = self.encoder(inputs)
    #     x = self.decoder(x*0.0, x)
    #     return x
    
    # def train_step(self, data):
    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x)
    #         loss = self.compute_loss(y=y, y_pred=y_pred) 
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     return {"loss": loss}
    
    
i = 25
Sigma0 = tf.eye(nD)
w0 = tf.cast(normalize_weights(weights_PF_1[i,:]), tf.float32)

optimizer = keras.optimizers.SGD(1e-3)
model = ParticleTransform(h=8, k=32, N=tf.cast(Np,tf.float32), SigmaX=Sigma0)

model.compile(optimizer=optimizer)
model.fit(x=particles_PF_1[i,:,:], y=w0, verbose=1, epochs=10)

new_parts, latent_mu, laten_lv = model.predict(particles_PF_1[i,:,:])

plt.hist(particles2_PF_1[i,:,:].numpy().flatten(), bins=30, alpha=0.5, label='Particles')
plt.hist(new_parts.flatten(), bins=30, alpha=0.5, label='Transformed Particles', color="red")
plt.show() 

compute_posterior(tf.ones(Np)/Np, new_parts)
compute_posterior( tf.cast(w0, tf.float64), particles_PF_1[i,:,:])




embedding_dim = 32
num_heads = 8

encoder = EncoderPT(embedding_dim, num_heads)
decoder = DecoderPT(embedding_dim, num_heads)

x = tf.random.uniform((10,100,4)) 
x = layers.Reshape((1,-1))(x)

ffn = layers.Dense(embedding_dim, activation='tanh')
ffn_output = ffn(x) # OUtput from first ff layer : (Batch size, Input sequence length, embedding_dim)
ffn_output.shape

# Input for encoder : (Batch Size, Input Sequence Length, 1)
enc_output = encoder(x)  # Output from encoder : (Batch size, Input sequence length, embedding_dim)

# Input for decoder (Batch Size, Sequence Length, Embedding Size)
dec_input = tf.zeros((100, 4, embedding_dim))
dec_output = decoder(dec_input, enc_output[-1])  # Output from decoder : (Batch size, Input sequence length, embedding_dim)






attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
attn_output, attn_weights = attention(ffn_output, ffn_output, return_attention_scores=True)
# Weights from attention : (Batch size, heads, Input sequence length, Input sequence length)
# Output from attnetion : (Batch size, Input sequence length, embedding_dim)

query, _, key, _, val, _, out, _ = attention.get_weights()
# queury (32, 8, 32) , key (32, 8, 32) , value (32,8,32) , output (8,32,32)

    
latent_dim = 2
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()   
    
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
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

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)
    


class DotProductAttention(layers.Layer):
    
    def __init__(self, d_k):
        super(DotProductAttention, self).__init__()
        self.d_k = d_k
        
    def call(self, queries, keys, values, mask=None):       
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = tf.linalg.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(self.d_k)
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
        # Computing the weights by a softmax operation
        weights = tf.nn.softmax(scores) 
        # Computing the attention by a weighted sum of the value vectors
        return tf.linalg.matmul(weights, values)
 
 
# Implementing the Multi-Head Attention
class Custom_MHA(layers.Layer):
    
    def __init__(self, h, d_k):
        super(Custom_MHA, self).__init__()
        self.attention = DotProductAttention(d_k) # Scaled dot product attention
        self.heads = h # Number of attention heads to use
        self.d_k = d_k # Dimensionality of the linearly projected queries and keys
        self.d_v = d_k # Dimensionality of the linearly projected values
        d_model = h * d_k
        #units of projection matrices should be d_model instead of d_k/d_v
        self.W_q = layers.Dense(d_model) # Learned projection matrix for the queries
        self.W_k = layers.Dense(d_model) # Learned projection matrix for the keys
        self.W_v = layers.Dense(d_model) # Learned projection matrix for the values
        self.W_o = layers.Dense(d_model) # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = tf.reshape(x, shape=(x.shape[0], x.shape[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(x.shape[0], x.shape[1], self.d_k))
        return x
    
    # def reshape_weights(self, w, heads, ndims):
    #     new_w = tf.ones(w.shape[0], heads, ndims, ndims)
    #     for h in range(heads):
    #         for i in range(ndims):
    #             for j in range(ndims):
    #                 new_w[:,h,i,j] = w
    #     return new_w
 
    def call(self, queries, keys, values, mask=None):
                
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, mask=mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)
    


attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
attn_output, attn_weights = attention(ffn_output, ffn_output, return_attention_scores=True)
# Weights from attention : (Batch size, heads, Input sequence length, Input sequence length)
# Output from attnetion : (Batch size, Input sequence length, embedding_dim)

x = tf.random.uniform((100, 4, embedding_dim))
x_reshaped = tf.reshape(x, shape=(x.shape[0], x.shape[1], num_heads, -1))
x_reshaped = tf.transpose(x_reshaped, perm=(0, 2, 1, 3))

custom_attn = DotProductAttention(embedding_dim)
custom_attn_output = custom_attn(ffn_output, ffn_output, ffn_output)

layer_mha = Custom_MHA(h=num_heads, d_k=embedding_dim)
mha_output = layer_mha(ffn_output, ffn_output, ffn_output)

    
    
    
    
    
    
    
    
    
    
    
    
class weights_loss(Loss):
    
    def __init__(self, N, reduction="sum_over_batch_size", name='particles_loss'):
        super().__init__(reduction=reduction, name=name)
        self.N = N

    def LogSumExp(self, x):
        c               = tf.reduce_max(x) 
        return c + tf.math.log(tf.reduce_sum(tf.math.exp(x - c), axis=0))
    
    def call(self, y, y_pred):
        y_log = tf.math.log(y_pred + 1e-12)
        loss = - (self.LogSumExp(y_log) - tf.math.log(self.N))
        return loss


class WeightsBP(keras.Model):
       
    def __init__(self):
        super().__init__()
        self.dense_layer1 = layers.Dense(units=32, activation="relu")  
        self.dense_layer2 = layers.Dense(units=64, activation="relu")
        self.output_layer = layers.Dense(units=1, activation="sigmoid")
        
    def call(self, inputs, training=False):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        return self.output_layer(inputs)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            w_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=w_pred) 
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}
    

optimizer = tf.keras.optimizers.SGD(1e-3)
model2 = WeightsBP()
model2.compile(optimizer=optimizer, loss=weights_loss(tf.cast(Np,tf.float32)))

model2.fit(x=w1[:,tf.newaxis], y=w2, verbose=1)
model2.predict(w2[:,tf.newaxis])








import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data 
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)



learning_rate = 0.01
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

class tryBP(keras.Model):
       
    def __init__(self):
        super().__init__()
        self.dense_layer1 = layers.Dense(units=32, activation="relu")  
        self.dense_layer2 = layers.Dense(units=64, activation="relu")
        self.output_layer = layers.Dense(units=3, activation="softmax")
        
    def call(self, inputs, training=False):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        return self.output_layer(x)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred) 
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}
    
model3 = tryBP()
model3.summary() 

model3.compile(optimizer=optimizer, loss=loss_fn)
model3.fit(x=X_train, y=y_train, verbose=1, epochs=20)
preds = model3.predict(x=X_test)



layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[1., 2., 3.]])
with tf.GradientTape() as tape:
  # Forward pass
  y = layer(x)
  loss = tf.reduce_mean(y**2)
# Calculate gradients with respect to every trainable variable
grad = tape.gradient(loss, layer.trainable_variables)
for var, g in zip(layer.trainable_variables, grad):
  print(f'var: {var}, grad: {g}')
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense_3 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense_4 = tf.keras.layers.Dense(28 * 28)
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.reshape(x)
    
class GAN(tf.keras.Model):
    # define the models
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
    
    # Define the compiler
    def compile(self, disc_optimizer, gen_optimizer, loss_fn, generator_loss, discriminator_loss):
        super(GAN, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.loss_fn = loss_fn

    # Modify Train step for GAN
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # Define the loss function
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generator(noise)
            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_images)
            
            gen_loss = self.generator_loss(self.loss_fn, fake_output)
            disc_loss = self.discriminator_loss(self.loss_fn, real_output, fake_output)

        # Calculate Gradient
        grad_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        grad_gen = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Optimization Step: Update Weights & Learning Rate
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        
        return {"Gen Loss ": gen_loss,"Disc Loss" : disc_loss}