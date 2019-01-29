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

# Import libraries

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

# Read data

DIR = '../../../data/stage-1_cleaned/'
f = DIR + 'twparmbeatmC1_no_nan.csv'
df = pd.read_csv(f,index_col=0) # the first column in .csv is index

# Double check NaN does not exist
print('There are {} NaN in the data.'.format(df.isnull().sum().sum()))
#df

####################

# Clear un-use thing

df = df.drop(columns='time')
df = df.drop(columns=['hour', 'month'])

# Generate inputs and labels

input = df.drop(columns='prec_sfc_1hrlater')
label = df['prec_sfc_1hrlater']

####################

# Split data
train_size = 0.7
train_cnt = floor(input.shape[0] * train_size)

x_train = input.iloc[0:train_cnt].copy().values
y_train = label.iloc[0:train_cnt].copy().values.reshape([-1,1])
x_test = input.iloc[train_cnt:].copy().values
y_test = label.iloc[train_cnt:].copy().values.reshape([-1,1])

# Normalize data
INPUT_PRE_NORM = tf.placeholder("float", [None, None], name='pre_norm')
mean, variance = tf.nn.moments(INPUT_PRE_NORM, [0], name='moments') # batch normalization
std = tf.sqrt(variance)

NORM_MEAN = tf.placeholder("float", [None])
NORM_STD = tf.placeholder("float", [None])
normalized = (INPUT_PRE_NORM - NORM_MEAN) / NORM_STD
with tf.Session() as sess:
  # Normalize everything using mean/std of training data
  _mean, _std = sess.run([mean, std], feed_dict = {INPUT_PRE_NORM: x_train})

  x_train = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: x_train,
                                              NORM_MEAN: _mean,
                                              NORM_STD: _std})

  # Double check _mean_0, _std_1 are all zeros and ones
  #_mean_0, _std_1 = sess.run([mean, std], feed_dict = {INPUT_PRE_NORM: x_train})
  #print(_mean_0, _std_1)
  prec_mean, prec_std = _mean[4], _std[4]

  x_test = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: x_test,
                                             NORM_MEAN: _mean,
                                             NORM_STD: _std})

  # All precipitation uses the same normalization constants
  y_train = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: y_train,
                                              NORM_MEAN: np.atleast_1d(prec_mean),
                                              NORM_STD: np.atleast_1d(prec_std)})

  y_test = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: y_test,
                                             NORM_MEAN: np.atleast_1d(prec_mean),
                                             NORM_STD: np.atleast_1d(prec_std)})

####################

# Network Parameters
n_in = x_train.shape[1] # number of input
n_out = y_train.shape[1] # number of output
n_hid = [n_in, 8, n_out]

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
    return _layer, weight

####################

# Training parameters

num_epoch = 100000
batch_size = 300
display_epoch = 5000
summ_epoch = 10

####################

# Create NN model

def neural_net(connection, act_func, loss_func, learning_rate, hparam, run_ID, text_info):
  LOGDIR = '../../log/' + hparam
  MODELDIR = LOGDIR + '/model.ckpt'
  tf.reset_default_graph() # clear graph stack
  sess = tf.Session() # declare a session

  # tf Graph input
  X = tf.placeholder("float", [None, n_in], name='inputs')
  Y = tf.placeholder("float", [None, n_out], name='labels')

  # Layer connection
  layer_1, w_in_1 = layer(X, n_in, n_hid[1], act_func,  'layer_1')
  layer_out, w_1_out = layer(layer_1, n_hid[1], n_out, 'none', 'layer_out')

  # True data information
  with tf.name_scope('constant'):
    _prec_mean = tf.constant(prec_mean.astype(np.float32), name='prec_mean')
    _prec_std = tf.constant(prec_std.astype(np.float32), name='prec_std')

  # Loss function
  with tf.name_scope('losses'):
    if loss_func == 'quartic':
      loss = tf.reduce_mean(tf.square(tf.square(layer_out - Y)), name=loss_func+'_loss') # mean-quartic-error
    elif loss_func == 'square_l2':
      pure_loss = tf.reduce_mean(tf.square(layer_out - Y)) # mean-square-error
      regularizer = tf.nn.l2_loss(w_in_1) + tf.nn.l2_loss(w_1_out)
      beta = 0.01
      loss = tf.reduce_mean(pure_loss + beta * regularizer, name=loss_func+'_loss')

  with tf.name_scope('denorm_abs_losses'):
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
  summ_test_abs_loss = tf.summary.scalar('test_abs_loss', abs_loss)

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
    datum_cnt = len(x_train)
    total_batch = int(datum_cnt / batch_size)
    x_batches = np.array_split(x_train, total_batch)
    y_batches = np.array_split(y_train, total_batch)
    for i in range(total_batch):
      batch_x, batch_y = x_batches[i], y_batches[i]
      _opt, _loss, _summ = sess.run([optimizer, abs_loss, summ], feed_dict={X: batch_x,
                                                                            Y: batch_y})
      _loss_train += _loss / datum_cnt * len(batch_x)

    _loss_test, _summ_test_abs_loss = sess.run([abs_loss, summ_test_abs_loss], feed_dict={X: x_test,
                                                                                                                  Y: y_test})
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
  loss_test, output_test = sess.run([abs_loss, layer_out], feed_dict={X: x_test,
                                                                      Y: y_test})
  # Test the training sets
  loss_train, output_train = sess.run([abs_loss, layer_out], feed_dict={X: x_train,
                                                                        Y: y_train})
  sess.close()
  return loss_train, output_train, loss_test, output_test
toc()

####################

# Network structures
connections = ['fc']#, 'bn', 'do']
act_funcs = ['relu']#, 'leaky_relu']
loss_funcs = ['square_l2']#, 'quartic']
learning_rates = [1e-3]#, 1e-4]

# Unique run ID
run_ID = '11.4'
text_info = '1 hid layer regression with l2 reg beta=0.01'

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
        labels = y_test * prec_std + prec_mean
        plt.figure()
        plt.scatter(labels, outputs, c='blue')
        plt.xlabel('True precipitation')
        plt.ylabel('Predicted precipitation')
        #axes = plt.gca()
        #axes.set_xlim([0,20])
        #axes.set_ylim([0,20])
        plt.savefig('../../log/' + hparam + '/test.eps')
        # Plot training data
        plt.figure()
        outputs = output_train * prec_std + prec_mean
        labels = y_train * prec_std + prec_mean
        plt.scatter(labels, outputs, c='blue')
        plt.xlabel('True precipitation')
        plt.ylabel('Predicted precipitation')
        plt.savefig('../../log/' + hparam + '/train.eps')

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
