# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:00:48 2020
@author: 贾熠辰
"""

import tensorflow as tf
import numpy as np
import scipy.io as io
import glob
import os
import scipy.io.wavfile as wav
from python_speech_features import *
import codecs
from pydub import AudioSegment
 
#cnn, same with train.py
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
predict = tf.argmax(logits, 1)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'E:/..') 

ID=("Liu Jinyi","Jia Yichen") 

print("If you want to input a single-track wav file, its length must be at least 11s; if a dimensional-sound wav, at least 6s.")
key=input("Input a .mat file: press (m); Input a .wav file: press(w). Press: ")
if key=="m":
    location=input(".mat name: ")
    data=io.loadmat(location)
    features=data["data"]

    res=sess.run(predict, feed_dict={x:[features]})
    print("detect person: ", ID[res[0]])
        
if key=="w":
    path=input(".wav name: ")
    rate,audio=wav.read(path)
    feature0=mfcc(audio,rate)
    dif1=delta(feature0,1)
    feature=np.hstack((feature0,dif1))
    feature=feature[:1024]
    feature=feature.reshape(32,32,26)

    res=sess.run(predict, feed_dict={x:[feature]})
    print("detect person: ", ID[res[0]])
    
    
    
    
