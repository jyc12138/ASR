# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:44:51 2020

@author: 贾熠辰
"""
import os
import scipy.io.wavfile as wav
from python_speech_features import *
import codecs
import numpy as np
from pydub import AudioSegment
import scipy.io as io

readpath1=r"/.."
readpath2=r"/.."

outpath1=r"/.."
outpath2=r"/.."

def deal(readpath,outpath,n):
    n=str(n)
    files=os.listdir(readpath)
    i=1
    for file in files:
         rate,audio=wav.read(readpath+"/"+file)
         #提取特征
         feature0=mfcc(audio,rate)
         #差分
         dif1=delta(feature0,1)
         #dif2=delta(feature0,2)
         feature=np.hstack((feature0,dif1))
         feature=feature[:1024]
         feature=feature.reshape(32,32,26)
         matname = n+"_{0}.mat".format(i)
         io.savemat(outpath+"/"+matname, dict([('data', feature), ('label',int(n))]))
         i=i+1
         
deal(readpath1,outpath1,1)
deal(readpath2,outpath2,2)
