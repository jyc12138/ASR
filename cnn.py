# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:42:14 2020
@author: 贾熠辰
"""

import tensorflow as tf
import numpy as np
import scipy.io as io
import glob
import os

path= r"E:/.."
def readmfcc(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    mfccarr = []
    labels = []
    for idx, folder in enumerate(cate):
        for i in glob.glob(folder + '/*.mat'):
            data = io.loadmat(i)
            mfccarr.append(data["data"])
            labels.append(int(data["label"][0])-1)
    return np.asarray(mfccarr, np.float32), np.asarray(labels, np.int32)

data, label = readmfcc(path)


#disrupt the image order
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#placeholder
x=tf.placeholder(tf.float32,shape=[None,32,32,26],name='x')
y=tf.placeholder(tf.int32,shape=[None,],name='y')

def cnnlayer():
    #first layer
    conv1=tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #print(conv1.shape)
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #print(pool1.shape)
    #第二个卷积层(64->32)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #print(conv2.shape)
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #print(pool2.shape)
 
    #第三个卷积层(32->16)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #print(conv3.shape)
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    #print(pool3.shape)
 
    re1 = tf.reshape(pool3, [-1, 4*4*128])
    #print(re1.shape)

    #全连接层
    dense1 = tf.layers.dense(inputs=re1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    #print(dense1.shape)
    dense2= tf.layers.dense(inputs=dense1, 
                          units=256, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    #print(dense2.shape)
    logits= tf.layers.dense(inputs=dense2, 
                            units=2,
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    #print(logits.shape)
    return logits

logits=cnnlayer()

loss=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train set(0.8)and validation set(0.2)
rate=0.8
s=np.int(num_example*rate)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#minibatch
def minibatch(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#train and validation
saver= tf.train.Saver(max_to_keep=3)
max_acc= 0
n_epoch= 4
batch_size= 10
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    # train
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatch(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    print("train loss: %f" % (train_loss / n_batch))
    print("train acc: %f" % (train_acc / n_batch))
    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatch(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("validation loss: %f" % (val_loss / n_batch))
    print("validation acc: %f" % (val_acc / n_batch))
    if val_acc > max_acc:
        max_acc= val_acc
        saver.save(sess, r"/..", global_step=epoch + 1)
sess.close()
