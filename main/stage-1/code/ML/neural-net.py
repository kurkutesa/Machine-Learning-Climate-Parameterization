
# coding: utf-8

# # Libraries

# Import libraries, set random seed, define rainy threshold

# In[1]:


import os

import numpy as np
import pandas as pd
from math import floor, ceil

import matplotlib.pyplot as plt
import seaborn as sns

import xarray as xr
import tensorflow as tf

print('All packages imported.')


# In[2]:


# Random seed for reproducibility

seed = 42
print(f'Random seed set as {seed}.')


# In[3]:


# Set threshold of rainy events
classification = True

prec_threshold = 0
print(f'Threshold of rainy event is {prec_threshold} mm/hr')
print('Classification' if classification else 'Regression')


# # Data Import

# In[4]:


def get_file_path(file_name):
    CURRENT_DIR = os.getcwd()
    DATA_DIR = f'{CURRENT_DIR}/../../../data/stage-1_cleaned'
    FILE_PATH = f'{DATA_DIR}/{file_name}'
    return FILE_PATH


def import_DS(FILE_PATH):
    return xr.open_dataset(FILE_PATH)

def DS_dropna(DS):
    return DS.dropna(dim='time')


# In[5]:


FILE_PATH = get_file_path(file_name='merged_dropped.cdf')
DS_raw = import_DS(FILE_PATH)
DS_raw


# In[6]:


DS = DS_dropna(DS_raw)
DS


# # Data Pre-processing

# In[7]:


str_y = 'prec_sfc_next'
str_x_scalar = ['T_sfc', 'p_sfc', 'rh_sfc', 'u_sfc', 'v_sfc', 'prec_sfc', 'down_short_diffuse_hemisp']
str_x_1d = ['T_p', 'rh_p', 'u_p', 'v_p']
plev = DS['p'].values.astype(float)  # array of pressure level

def extract(DS, str_y=str_y, str_x_scalar=str_x_scalar):
    return DS[str_y].to_dataframe().values, DS[str_x_scalar].to_dataframe().values


def merge_channels(DS, str_x_1d=str_x_1d):
    channels = [DS[str_x_1d[i]].to_dataframe().unstack(level=-1)
                for i in range(0, len(str_x_1d))]
    X_conv = np.expand_dims(channels[0].values, axis=2)

    for channel in channels[1:]:
        channel = np.expand_dims(channel.values, axis=2)
        X_conv = np.append(X_conv, channel, axis=2)

    return X_conv


# In[8]:


y, X_scalar = extract(DS)
print(y.shape)
print(X_scalar.shape)


# In[9]:


X_conv = merge_channels(DS)
X_conv.shape


# In[12]:


binary = y > prec_threshold
print('1 class ratio= {:.2%}'.format(binary.mean()))


# # Data Standardization

# In[10]:


def split(binary, y, X_scalar, X_conv, train_size=0.7, seed=seed):
    from sklearn.model_selection import train_test_split
    train_binary, test_binary, train_y, test_y, train_X_scalar, test_X_scalar, train_X_conv, test_X_conv = train_test_split(binary, y, X_scalar, X_conv,
                                                                                                                            train_size=train_size,
                                                                                                                            random_state=seed,
                                                                                                                            shuffle=True,
                                                                                                                            stratify=None)
    return train_binary, test_binary, train_y, test_y, train_X_scalar, test_X_scalar, train_X_conv, test_X_conv


def standardize(train, test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test, scaler


def standardize_3d(train, test):
    from sklearn.preprocessing import StandardScaler
    scalers = {}
    for i in range(train.shape[2]):
        scalers[i] = StandardScaler()
        train[:, :, i] = scalers[i].fit_transform(train[:, :, i])

    for i in range(test.shape[2]):
        test[:, :, i] = scalers[i].transform(test[:, :, i])

    return train, test, scalers


# In[13]:


# train-test split
train_binary, test_binary, train_y, test_y, train_X_scalar, test_X_scalar, train_X_conv, test_X_conv = split(
    binary, y, X_scalar, X_conv)


# In[14]:


# standardize
train_y, test_y, scaler_y = standardize(train_y, test_y)
train_X_scalar, test_X_scalar, scaler_X_scalar = standardize(train_X_scalar, test_X_scalar)
train_X_conv, test_X_conv, scalers_X_conv = standardize_3d(train_X_conv, test_X_conv)


# # Neural Network Training

# In[15]:


from keras.models import Model
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout, MaxPooling1D
from keras.layers import concatenate, Flatten, Input
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import regularizers


# In[19]:


final_activation, loss_metric, show_metric, labels = [
    'sigmoid', 'binary_crossentropy', ['accuracy'], train_binary] if classification else ['relu', 'mean_squared_error', None, train_y]


# Input placeholders
input_conv = Input(shape=train_X_conv.shape[1:], name='column_input')
input_scalar = Input(shape=(train_X_scalar.shape[1],), name='surface_input')

# Hidden layers
conv_1 = Conv1D(16, kernel_size=6)(input_conv)
bn_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling1D(pool_size=4)(bn_1)
act_1 = Activation('relu')(pool_1)

conv_2 = Conv1D(8, kernel_size=4)(act_1)
bn_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling1D(pool_size=5)(bn_2)
act_2 = Activation('relu')(pool_2)

flatten_2 = Flatten()(act_2)

layer = concatenate([flatten_2, input_scalar])
#layer = Dense(4, activation='relu')(layer)

output = Dense(1,
               kernel_regularizer=regularizers.l2(0.02),
               activation=final_activation,
               name='output')(layer)

# 1. Initialize
model = Model(inputs=[input_conv, input_scalar], outputs=output)
plot_model(model, show_shapes=True, to_file='model.png')
print(model.summary())


# In[17]:


# 2. Compile
model.compile(optimizer=Adam(),
              loss=loss_metric,
              metrics=show_metric)


# In[18]:


# 3. Train
model.fit([train_X_conv, train_X_scalar], labels,
          epochs=5000,
          verbose=2)


# # Testing

# ## Regression

# In[47]:


def inverse_standardize(y, y_hat, scaler):
    y, y_hat = [scaler.inverse_transform(yyy) for yyy in [y, y_hat]]
    
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(y[:,0], y_hat[:,0])
    ax.set(xlabel='Actual Precipitation', ylabel='Predicted Precipitation')
    plt.show()
    return None


# In[61]:


# 4. Test
if not classification:
    test_loss = model.evaluate([test_X_conv, test_X_scalar], test_y, verbose=0)
    print(f'Test loss= {test_loss:.4f}')

    test_y_hat = model.predict([test_X_conv, test_X_scalar])
    inverse_standardize(test_y, test_y_hat, scaler_y)


# ## Classification

# In[41]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[50]:


if classification:
    proba_binary_hat = model.predict([test_X_conv, test_X_scalar])
    test_binary_hat = proba_binary_hat >= 0.5


# In[52]:


if classification:
    print(accuracy_score(test_binary, test_binary_hat))
    print(classification_report(test_binary, test_binary_hat))
    print(confusion_matrix(test_binary, test_binary_hat))


# In[61]:


score, acc = model.evaluate([test_X_conv, test_X_scalar], test_binary)
print('Test score:', score)
print('Test accuracy:', acc)


# # Script Conversion (for ssh Burn)

# In[20]:


get_ipython().system('jupyter nbconvert --to script neural-net.ipynb')

