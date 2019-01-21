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
f = DATADIR + '/ARM_1hrlater.csv'
df = pd.read_csv(f,index_col=0) # the first column in .csv is index

# Double check NaN does not exist
print('There are {} NaN in the data.'.format(df.isnull().sum().sum()))
#df

####################

# Generate inputs and labels
input = df.drop(columns='prec_sfc_1hrlater')
raw_label = df['prec_sfc_1hrlater']

## >0.31 mm in 6-hour period is counted as rainy
label = (raw_label.values > 0.1) *1 # ensure it is in int type
print('Rainy period ratio= {:.4f}'.format(label.sum()/label.size))

####################

# Split data, deep copy to prevent contaminating raw data with standardization
train_size = 0.6
train_cnt = floor(input.shape[0] * train_size)

x_train = input.iloc[0:train_cnt].copy().values
y_train = label[0:train_cnt].copy().reshape([-1,1])
x_test = input.iloc[train_cnt:].copy().values
y_test = label[train_cnt:].copy().reshape([-1,1])

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

  x_test = sess.run(normalized, feed_dict = {INPUT_PRE_NORM: x_test,
                                             NORM_MEAN: _mean,
                                             NORM_STD: _std})

  # No normalization for labels
toc()

####################

# Network Parameters
n_in = x_train.shape[1] # number of input
n_out = 1 # number of output
n_hid = [n_in, 5, n_out]

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
num_epoch = 3000
batch_size = 300
display_epoch = 500
summ_epoch = 5

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
  #layer_2 = layer(layer_1, n_hid[1], n_hid[2], act_func, 'layer_2')
  #layer_3 = layer(layer_2, n_hid[2], n_hid[3], act_func, 'layer_3')
  #layer_4 = layer(layer_3, n_hid[3], n_hid[4], act_func, 'layer_4')
  logit = layer(layer_1, n_hid[1], n_out, 'none', 'layer_logit')
  with tf.name_scope('layer_out'):
    layer_out = tf.sigmoid(logit)
  with tf.name_scope('predictions'):
    pred = tf.round(layer_out)

  # True data information - no need for classification labels
  #with tf.name_scope('constant'):
    #_prec_mean = tf.constant(prec_mean.astype(np.float32), name='prec_mean')
    #_prec_std = tf.constant(prec_std.astype(np.float32), name='prec_std')

  # Loss function
  with tf.name_scope('losses'):
    if loss_func == 'xent':
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit), name=loss_func+'_loss') # sigmoid(logit) cross-entropy loss
    elif loss_func == 'square':
      loss = tf.reduce_mean(tf.square(layer_out - Y), name=loss_func+'_loss') # mean-square-error
    elif loss_func == 'hinge':
      loss = tf.reduce_mean(tf.losses.hinge_loss(labels=Y, logits=logit), name=loss_func+'_loss') # hinge loss

  with tf.name_scope('abs_losses'):
    abs_loss = tf.reduce_mean(tf.abs(layer_out - Y), name='abs_loss')

  tf.summary.scalar(loss_func+'_loss', loss)
  tf.summary.scalar('abs_loss', abs_loss)

  with tf.name_scope('acc'):
    correct_cnt = tf.equal(pred, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_cnt, tf.float32), name='accuracy')
  tf.summary.scalar('accuracy', accuracy)

  # Optimizer
  with tf.name_scope('train'):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  # Summaries and saver
  summ = tf.summary.merge_all() # merge all summaries for Tensorboard
  saver = tf.train.Saver() # declare NN config saver

  # For test only
  summ_test_abs_loss = tf.summary.scalar('test_abs_loss', abs_loss)
  summ_test_accuracy = tf.summary.scalar('test_accuracy', accuracy)

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
    _accuracy_train = 0.0
    datum_cnt = len(x_train)
    total_batch = int(datum_cnt / batch_size)
    x_batches = np.array_split(x_train, total_batch)
    y_batches = np.array_split(y_train, total_batch)
    for i in range(total_batch):
      batch_x, batch_y = x_batches[i], y_batches[i]
      _opt, _accuracy = sess.run([optimizer, accuracy], feed_dict={X: batch_x,
                                                                   Y: batch_y})
      _accuracy_train += _accuracy / datum_cnt * len(batch_x)

    _abs_loss_test, _accuracy_test, _summ_test_abs_loss, _summ_test_accuracy = sess.run([abs_loss, accuracy, summ_test_abs_loss, summ_test_accuracy], feed_dict={X: x_test,
                                                                                                                                                             Y: y_test})
    if epoch == 1:
      _summ_stamp = sess.run(summ_stamp)
      writer.add_summary(_summ_stamp)

    if epoch % summ_epoch == 0:
      _summ = sess.run(summ, feed_dict={X: x_train,
                                        Y: y_train})

      writer.add_summary(_summ, epoch)
      writer.add_summary(_summ_test_abs_loss, epoch)
      writer.add_summary(_summ_test_accuracy, epoch)

    if epoch % display_epoch == 0:
      print("epoch " + str(epoch) + ", train acc=" + "{:.5f}".format(_accuracy_train) + ", test acc=" + "{:.5f}".format(_accuracy_test))
  print("Optimization Finished!")

  # Save NN model
  save_path = saver.save(sess, MODELDIR)
  print("Model saved in path: {}".format(save_path))

  # Test
  _pred_test = sess.run(pred, feed_dict={X: x_test,
                                         Y: y_test})
  # Test the training sets
  _pred_train = sess.run(pred, feed_dict={X: x_train,
                                          Y: y_train})
  sess.close()
  return _pred_test, _pred_train
toc()

####################

# Network structures
connections = ['fc']
act_funcs = ['leaky_relu','relu'] #'lr-svmlin'
loss_funcs = ['xent','hinge']#,'square']
learning_rates = [1e-3]#, 1e-5, 1e-6]

# Unique run ID
run_ID = '03.1'
text_info = '1hdNN classification on 1_hr_later, threshold_0.1'

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
        pred_test, pred_train = neural_net(connection, act_func, loss_func, learning_rate, hparam, run_ID, text_info)

        # Plot
        test_raw_label = raw_label[train_cnt:].values
        rainy = pred_test == 1
        print('Predicted number of rainy hours= {}, total hours= {}'.format(rainy.sum(), rainy.size))
        plt.figure()
        plt.scatter(test_raw_label[rainy[:,0]], np.zeros(test_raw_label[rainy[:,0]].size) + .5, c='blue', label='rainy')
        plt.scatter(test_raw_label[~rainy[:,0]], np.zeros(test_raw_label[~rainy[:,0]].size), c='red', label='dry')
        axes = plt.gca()
        axes.set_ylim([-.5,1])
        axes.legend()
        plt.savefig('./log/' + hparam + '/test.eps')

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
