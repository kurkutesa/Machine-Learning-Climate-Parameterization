#! export PATH="/scratch/s1895566/miniconda/base/bin:$PATH"

import time
# MATLAB like tic toc
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: {:.1f} seconds.".format(tempTimeInterval) )
def tic():
    toc(False)
tic()

####################

# Import pkg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from math import floor, ceil
#import xarray as xr
import datetime as dt
import smtplib
import tensorflow as tf
print('All packages imported.')
toc()

# Reproducibility
random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)

####################

'''
# Mount Google Drive locally
from google.colab import drive
drive.mount('/content/gdrive')

# Check data list
get_ipython().system(u'ls "/content/gdrive/My Drive/Colab Notebooks/data/"')
get_ipython().system(u"ls '/tmp'")
'''
####################

# Read data
#DATADIR = '/content/gdrive/My Drive/Colab Notebooks/data'
DATADIR = '../data/forNN'
f = DATADIR + '/ARM_1hrlater_RFclassified_threshold_0.05.csv'
df = pd.read_csv(f,index_col=0) # the first column in .csv is index

# Double check NaN does not exist
print('There are {} NaN in the data.'.format(df.isnull().sum().sum()))
#df

####################

# Generate inputs and labels
input = df.drop(columns='prec_sfc_1hrlater').drop(columns='pred_class').copy()
label = df['prec_sfc_1hrlater']
RFclass = df['pred_class']

####################

# Split data, deep copy to prevent contaminating raw data with standardization
train_size = 0.6 # have to follow the split in RF training
train_cnt = floor(input.shape[0] * train_size)

x_train = input.iloc[0:train_cnt].copy().values
y_train = label.iloc[0:train_cnt].copy().values.reshape([-1,1])
RFclass_train = RFclass.iloc[0:train_cnt].copy().values.reshape([-1,1])
x_test = input.iloc[train_cnt:].copy().values
y_test = label.iloc[train_cnt:].copy().values.reshape([-1,1])
RFclass_test = RFclass.iloc[train_cnt:].copy().values.reshape([-1,1])

# Delete non-rainy data in training data set
x_train_rainy = x_train[RFclass_train[:,0]==1,:]
y_train_rainy = y_train[RFclass_train[:,0]==1,:]
x_test_rainy = x_test[RFclass_test[:,0]==1,:]
y_test_rainy = y_test[RFclass_test[:,0]==1,:]

x_train_dry = x_train[RFclass_train[:,0]==0,:]
y_train_dry = y_train[RFclass_train[:,0]==0,:]
x_test_dry = x_test[RFclass_test[:,0]==0,:]
y_test_dry = y_test[RFclass_test[:,0]==0,:]

intrinsic_test_dry_loss_sum = y_test_dry.sum() # coz we assumed they were dry

# Normalize data
INPUT_PRE_NORM = tf.placeholder("float", [None, None], name='pre_norm')
mean, variance = tf.nn.moments(INPUT_PRE_NORM, [0], name='moments') # batch normalization
std = tf.sqrt(variance)

NORM_MEAN = tf.placeholder("float", [None])
NORM_STD = tf.placeholder("float", [None])
normalized = (INPUT_PRE_NORM - NORM_MEAN) / NORM_STD
with tf.Session() as sess:
  # Normalize everything using mean/std of training data
  _mean, _std = sess.run([mean, std], feed_dict = {INPUT_PRE_NORM: x_train_rainy})
  x_train_rainy = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: x_train_rainy,
                                              NORM_MEAN: _mean,
                                              NORM_STD: _std})
  # Double check _mean_0, _std_1 are all zeros and ones
  #_mean_0, _std_1 = sess.run([mean, std], feed_dict = {INPUT_PRE_NORM: x_train})
  #print(_mean_0, _std_1)
  prec_mean, prec_std = _mean[2], _std[2]
  x_test_rainy = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: x_test_rainy,
                                             NORM_MEAN: _mean,
                                             NORM_STD: _std})

  # All precipitation uses the same normalization constants
  y_train_rainy = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: y_train_rainy,
                                              NORM_MEAN: np.atleast_1d(prec_mean),
                                              NORM_STD: np.atleast_1d(prec_std)})
  y_test_rainy = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: y_test_rainy,
                                             NORM_MEAN: np.atleast_1d(prec_mean),
                                             NORM_STD: np.atleast_1d(prec_std)})
toc()

# REMEMBER _rainy are normalized, but _dry are not!

####################

# Network Parameters
n_in = x_train_rainy.shape[1] # number of input
n_out = y_train_rainy.shape[1] # number of output
n_hid = [n_in, 128, 64, 32, 16, n_out]

# Create layer template
def layer(x, size_in, size_out, act_func, name='layer'):
  with tf.name_scope(name):
    with tf.name_scope('weights'):
      weight = tf.Variable(tf.truncated_normal([size_in, size_out],
                                                stddev=1.0 / tf.sqrt(tf.cast(size_in, tf.float32))), name='weight') # rule of thumb
      tf.summary.histogram('weights', weight)
    with tf.name_scope('biases'):
      bias = tf.Variable(tf.constant(0.1, shape=[size_out]), name='bias') # avoid dead neurons
      tf.summary.histogram('biases', bias)

    with tf.name_scope('pre_activations'):
      _layer = tf.add(tf.matmul(x,weight), bias)
      tf.summary.histogram('pre_activations', _layer)

    if act_func == 'relu':
      with tf.name_scope('relu'):
        _layer = tf.nn.relu(_layer)
        tf.summary.histogram('relu', _layer)
    elif act_func == 'leaky_relu':
      with tf.name_scope('leaky_relu'):
        _layer = tf.nn.leaky_relu(_layer, alpha=0.1)
        tf.summary.histogram('leaky_relu', _layer)
    return _layer

####################

# Training parameters
num_epoch = 100000
batch_size = 300
display_epoch = 5000
summ_epoch = 10

####################

# Create NN model
def neural_net(connection, act_func, loss_func, learning_rate, hparam, run_ID, text_info):
  LOGDIR = './log/' + hparam
  MODELDIR = LOGDIR + '/model.ckpt'
  tf.reset_default_graph() # clear graph stack
  sess = tf.Session() # declare a session

  # tf Graph input
  X = tf.placeholder("float", [None, n_in], name='inputs')
  Y = tf.placeholder("float", [None, n_out], name='labels')

  # Layer connection
  layer_1 = layer(X, n_in, n_hid[1], act_func,  'layer_1')
  layer_2 = layer(layer_1, n_hid[1], n_hid[2], act_func, 'layer_2')
  layer_3 = layer(layer_2, n_hid[2], n_hid[3], act_func, 'layer_3')
  layer_4 = layer(layer_3, n_hid[3], n_hid[4], act_func, 'layer_4')
  layer_out = layer(layer_4, n_hid[4], n_out, 'none', 'layer_out')

  # True data information
  with tf.name_scope('constant'):
    _prec_mean = tf.constant(prec_mean.astype(np.float32), name='prec_mean')
    _prec_std = tf.constant(prec_std.astype(np.float32), name='prec_std')

  # Loss function
  with tf.name_scope('losses'):
    if loss_func == 'quartic':
      loss = tf.reduce_mean(tf.square(tf.square(layer_out - Y)), name=loss_func+'_loss') # mean-quartic-error
    elif loss_func == 'square':
      loss = tf.reduce_mean(tf.square(layer_out - Y), name=loss_func+'_loss') # mean-square-error

  with tf.name_scope('abs_losses'):
    abs_loss = tf.reduce_mean(tf.multiply(tf.abs(layer_out - Y), _prec_std) + _prec_mean, name='abs_loss') # De-normalized mean loss

  tf.summary.scalar(loss_func+'_loss', loss)
  tf.summary.scalar('abs_loss', abs_loss)

  # Optimizer
  with tf.name_scope('train'):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  summ = tf.summary.merge_all() # merge all summaries for Tensorboard
  saver = tf.train.Saver() # declare NN config saver

  # For test only
  with tf.name_scope('abs_losses_two_classes'):
    abs_loss_two_classes = tf.reduce_sum(tf.multiply(tf.abs(layer_out - Y), _prec_std) + _prec_mean) # De-normalized mean loss
    abs_loss_two_classes = (abs_loss_two_classes + intrinsic_test_dry_loss_sum)/ y_test.size
  summ_test_abs_loss_two_classes = tf.summary.scalar('test_abs_loss_two_classes', abs_loss_two_classes)

  # For first epoch only
  run_stamp = run_ID + '/ ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' /train_size_{:.2f}'.format(train_size) + ' / {}'.format(text_info)
  summ_stamp = tf.summary.text('run_stamp', tf.convert_to_tensor(run_stamp))

  # Draw graph
  sess.run(tf.global_variables_initializer()) # initialize session

  writer = tf.summary.FileWriter(LOGDIR) # a file writer
  writer.add_graph(sess.graph) # write the graph in the session

  # Train
  for epoch in range(1, num_epoch+1):
    # Batching
    _loss_train = 0.0
    datum_cnt = len(x_train_rainy)
    total_batch = int(datum_cnt / batch_size)
    x_batches = np.array_split(x_train_rainy, total_batch)
    y_batches = np.array_split(y_train_rainy, total_batch)
    for i in range(total_batch):
      batch_x, batch_y = x_batches[i], y_batches[i]
      _opt, _loss, _summ = sess.run([optimizer, abs_loss, summ], feed_dict={X: batch_x,
                                                                            Y: batch_y})
      _loss_train += _loss / datum_cnt * len(batch_x)

    _loss_test, _summ_test_abs_loss = sess.run([abs_loss_two_classes, summ_test_abs_loss_two_classes], feed_dict={X: x_test_rainy,
                                                                                                                  Y: y_test_rainy})
    if epoch == 1:
      _summ_stamp = sess.run(summ_stamp)
      writer.add_summary(_summ_stamp)

    if epoch % summ_epoch == 0:
      writer.add_summary(_summ, epoch)
      writer.add_summary(_summ_test_abs_loss, epoch)

    if epoch % display_epoch == 0:
      print("epoch " + str(epoch) + ", train MAE=" + "{:.5f}".format(_loss_train) + ", test MAE=" + "{:.5f}".format(_loss_test))
  print("Optimization Finished!")

  # Save NN model
  save_path = saver.save(sess, MODELDIR)
  print("Model saved in path: {}".format(save_path))

  # Test
  loss_test, output_test = sess.run([abs_loss, layer_out], feed_dict={X: x_test_rainy,
                                                                      Y: y_test_rainy})
  # Test the training sets
  loss_train, output_train = sess.run([abs_loss, layer_out], feed_dict={X: x_train_rainy,
                                                                        Y: y_train_rainy})
  sess.close()
  return loss_train, output_train, loss_test, output_test
toc()

####################

# Network structures
connections = ['fc']#, 'bn', 'do']
act_funcs = ['relu', 'leaky_relu']
loss_funcs = ['square', 'quartic']
learning_rates = [1e-3, 1e-4]

# Unique run ID
run_ID = '05.1'
text_info = 'regression after RF classification (with threshold_0.05)'

####################

# Log hyperparameter with folder name
def make_hparam_str(connection, act_func, loss_func, learning_rate, n_hid):
  string = run_ID + '/' + connection + '-' + act_func + '-' + loss_func + '/'
  string = string + 'lr_{:.0e},nn'.format(learning_rate)
  for n in n_hid:
    string = string + '_{}'.format(n)
  return string

# Construct model
for connection in connections:
  for act_func in act_funcs:
    for loss_func in loss_funcs:
      for learning_rate in learning_rates:
        hparam = make_hparam_str(connection, act_func, loss_func, learning_rate, n_hid)
        print('Run model with config ' + hparam)
        loss_train, output_train, loss_test, output_test = neural_net(connection, act_func, loss_func, learning_rate, hparam, run_ID, text_info)

        # Plot
        outputs = output_test * prec_std + prec_mean
        labels = y_test_rainy * prec_std + prec_mean
        plt.figure()
        plt.scatter(labels, outputs, c='blue')
        plt.scatter(y_test_dry, y_test_dry*0, c='red')
        plt.xlabel('True precipitation')
        plt.ylabel('Predicted precipitation')
        #axes = plt.gca()
        #axes.set_xlim([0,20])
        #axes.set_ylim([0,20])
        plt.savefig('./log/' + hparam + '/test.eps')
        # Plot training data
        plt.figure()
        outputs = output_train * prec_std + prec_mean
        labels = y_train_rainy * prec_std + prec_mean
        plt.scatter(labels, outputs, c='blue')
        plt.scatter(y_train_dry, y_train_dry*0, c='red')
        plt.xlabel('True precipitation')
        plt.ylabel('Predicted precipitation')
        plt.savefig('./log/' + hparam + '/train.eps')

####################

# Send an email to me when done
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
sender = 'eden.aukalong@gmail.com'
receiver = sender
pw = 'ipnmdwjqoaxyddoi'
server.login(sender, pw)
message = 'Subject: {}\n\n{}'.format('Neural networks are trained', 'hurray?')
server.sendmail(sender, receiver, message)
server.quit()
toc()
