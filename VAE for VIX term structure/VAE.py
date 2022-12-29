# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:52:30 2022

@author: stran
"""

#%% import package
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import tensorflow.keras as keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model, model_from_json
from keras import backend as K
from keras import losses

#%% data
data = pd.read_csv("C:/Users/stran/OneDrive/桌面/Python練習/論文/data/data2022 1-5(整理).csv")
data_train = data.iloc[0:25047]
data_test = data.iloc[25048:41218]
data_train = data_train.rename(columns = {'Unnamed: 0' : 'time'})
data_test = data_test.rename(columns = {'Unnamed: 0' : 'time'})
data_train = data_train.set_index('time')
data_test = data_test.set_index('time')
data_train_array = data_train.to_numpy()
data_test_array = data_test.to_numpy()

#%% model hyperparameters
batch_size, epochs, n_hidden, z_dim, beta = 33, 100, 16, 2, 0.1

#%% model architecture
x = Input(shape = data_train_array.shape[1])
encoder_layer1 = Dense(n_hidden, activation = 'relu')(x)
encoder_layer2 = Dense(n_hidden//2, activation = 'relu')(encoder_layer1)
mu = Dense(z_dim)(encoder_layer2)
log_sd = Dense(z_dim)(encoder_layer2)
def sampling(args):
    mu, log_sd = args
    eps = K.random_normal(shape = (batch_size, z_dim), mean = 0, stddev = 1)
    return mu + K.exp(log_sd) * eps
latent = Lambda(sampling, output_shape=(z_dim))([mu, log_sd])

z_decoder_layer1 = Dense(n_hidden//2, activation = 'relu')
z_decoder_layer2 = Dense(n_hidden, activation = 'relu')
y_decoder = Dense(data_train.shape[1], activation='relu')

z_decoded1 = z_decoder_layer1(latent)
z_decoded2 = z_decoder_layer2(z_decoded1)
y = y_decoder(z_decoded2)

#%% losses
RE = losses.mean_squared_error(x, y)
KL_div = beta * K.sum(K.square(mu) + K.exp(log_sd) - log_sd - 1, axis = -1)
VAE_loss = RE + KL_div

#%% build model
VAE = Model(x, y)
VAE.add_loss(VAE_loss)
VAE.compile(optimizer = 'rmsprop')
VAE.summary()

#%% train
VAE.fit(data_train_array,
        data_train_array,
        batch_size = batch_size,
        epochs = epochs,
        validation_data=(data_test_array, None))

# #%% save model.weight
# VAE.save_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/VAE/vae_model.weight")

# #%% call model.weight
# VAE.load_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/VAE/vae_model.weight", by_name=False)

#%% build encoder
encoder = Model(x, latent)
encoder.summary()

#%% plot the latent space
z = encoder.predict(data_test_array, batch_size = batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z[:, 0], z[:, 1], c='blue')
plt.show()

#%% build decoder
decoder_input = Input(shape=(z_dim,))
_z_decoder1 = z_decoder_layer1(decoder_input)
_z_decoder2 = z_decoder_layer2(_z_decoder1)
_y = y_decoder(_z_decoder2)
generator = Model(decoder_input, _y)
generator.summary()


#%% reconstruct error
data_rec = generator.predict(z, batch_size=batch_size)
data_rec = pd.DataFrame(data_rec, columns = ['1','2','3','4','5','6','7','8','9','10'])
data_test_array = pd.DataFrame(data_test_array, columns = ['1','2','3','4','5','6','7','8','9','10'])
error = round((data_rec - data_test_array) / data_test_array, 4)

#%% visualize
error.plot()
plt.legend(fontsize = 'x-small')
