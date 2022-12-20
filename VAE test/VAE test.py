# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:43:41 2022

@author: stran
"""

#%% import package
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model, model_from_json
from keras import backend as K
from keras import losses



#%% import data
data_train = yf.download('^VIX', start="1990-01-01", end="2005-12-31")
data_test = yf.download('^VIX', start="2005-01-01", end="2020-12-31")
data_train = data_train.drop(columns = ['Volume'])
data_train_array = data_train.to_numpy()
data_test = data_test.drop(columns = ['Volume'])
data_test_array = data_test.to_numpy()
print(data_train_array.shape)
print(data_test_array.shape)
#%% network parameters
batch_size, epochs, n_hidden, z_dim = 4, 100, 8, 2

#%% build encoder
x = Input(shape = data_train_array.shape[1])
x_encoder_layer1 = Dense(n_hidden, activation = 'relu')(x)
x_encoder_layer2 = Dense(n_hidden//2, activation = 'relu')(x_encoder_layer1)

mu = Dense(z_dim)(x_encoder_layer2)
log_sd = Dense(z_dim)(x_encoder_layer2)

#%% sampling fcn(from encoded data) z = mu + sd * epsilon
def sampling(args):
    mu, log_sd = args
    eps = K.random_normal(shape = (batch_size, z_dim), mean = 0, stddev = 1)
    return mu + K.exp(log_sd) * eps

z = Lambda(sampling, output_shape=(z_dim))([mu, log_sd])

#%% decoder
z_decoder_layer1 = Dense(n_hidden//2, activation = 'relu')
z_decoder_layer2 = Dense(n_hidden, activation = 'relu')
y_decoder = Dense(data_train.shape[1], activation='relu')

z_decoded1 = z_decoder_layer1(z)
z_decoded2 = z_decoder_layer2(z_decoded1)
y = y_decoder(z_decoded2)

#%% losses
RE = losses.mean_squared_error(x, y) * data_train_array.shape[1]
KL_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_sd) - log_sd - 1, axis = -1)

VAE_loss = RE + KL_loss

#%% build model
VAE = Model(x, y)
VAE.add_loss(VAE_loss)
VAE.compile(optimizer = 'rmsprop')
VAE.summary()

#%% train
VAE.fit(data_train_array,
        data_train_array,
        batch_size = batch_size,
        epochs = epochs)

#%% build encoder 
encoder = Model(x, z)
encoder.summary()

#%% plot of the digit classes in the latent space
data_test_array_latent = encoder.predict(data_test_array, batch_size = batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(data_test_array_latent[:, 0], data_test_array_latent[:, 1], c='blue')
plt.colorbar()
plt.show()

#%% build decoder
decoder_input = Input(shape=(z_dim,))
_z_decoder1 = z_decoder_layer1(decoder_input)
_z_decoder2 = z_decoder_layer2(_z_decoder1)
_y = y_decoder(_z_decoder2)
generator = Model(decoder_input, _y)
generator.summary()

#%% display a predict plot
pre_data_array = generator.predict(data_test_array_latent, batch_size=batch_size)
pre = pd.DataFrame(pre_data_array, columns = ['Open','High','Low','Close','Adj Close'])
pre_data = data_test.copy()[[]] #copy df1 and erase all column
pre_data['Open'] = pre_data_array[:, 0]
pre_data['High'] = pre_data_array[:, 1]
pre_data['Low'] = pre_data_array[:, 2]
pre_data['Close'] = pre_data_array[:, 3]
pre_data['Adj Close'] = pre_data_array[:, 4]
pre_data = round(pre_data,2)

mc = mpf.make_marketcolors(up='r',down='g',inherit=True) 
s  = mpf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mc) 
kwargs = dict(type='candle', mav=(5,20,60), figratio=(10,8), figscale=0.75, title='VIX index predicted', style=s) 
mpf.plot(pre_data, **kwargs)

#%% display a real plot
mc = mpf.make_marketcolors(up='r',down='g',inherit=True) 
s  = mpf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mc) 
kwargs = dict(type='candle', mav=(5,20,60), figratio=(10,8), figscale=0.75, title='VIX index', style=s) 
mpf.plot(data_test, **kwargs)

generator.predict(np.array([[0,1]]), batch_size=batch_size)

#%% save model.architecture
with open("C:/Users/stran/OneDrive/桌面/Python練習/論文/VAE test/vae_test_model.config", 'w') as text_file:
    text_file.write(VAE.to_json())

#%% save model.weight
VAE.save_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/VAE test/vae_test_model.weight")

#%% call model.architecture
with open("C:/Users/stran/OneDrive/桌面/Python練習/論文/VAE test/vae_test_model.config", 'r') as text_file:
    VAE = model_from_json(text_file.read())

#%% call model.weight
VAE.load_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/VAE test/vae_test_model.weight", by_name=False)
