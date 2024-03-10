#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:14:12 2022

@author: zhangj2
"""

import keras
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import regularizers
from keras import losses
from keras import optimizers
from keras.models import Model,load_model
from keras.layers import Activation
from keras.layers import Input, Dense, Dropout, Flatten,Embedding,LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization,Lambda
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D, concatenate,add
from keras.layers.advanced_activations import LeakyReLU

from keras.callbacks import ModelCheckpoint
from keras import backend as K
# from keras.utils import plot_model
# from keras.utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler,EarlyStopping

# In[] bulid model
num_classes=1
def res_block(x, channels, i):
    if i == 1:  # 第二个block
        strides = 1
        x_add = x
    else:  # 第一个block
        strides = 2
        # x_add 是对原输入的bottleneck操作
        x_add = Conv1D(channels,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       strides=strides)(x)
        x=BatchNormalization()(x)
 
    x = Conv1D(channels,
               kernel_size=3,
               activation='relu',
               padding='same')(x)
    x=BatchNormalization()(x)
    
    x = Conv1D(channels,
               kernel_size=3,
               padding='same',
               strides=strides)(x)
    
    x = add([x, x_add])
    Activation(K.relu)(x)
    x=BatchNormalization()(x)
    return x
    
def build_dis_model(input_shape):
    inpt = Input(shape=input_shape,name='wave_input')
     
    x = Conv1D(16,
               kernel_size=3,
               activation='relu',
               input_shape=input_shape,
               padding='same'
               )(inpt)
    
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    for i in range(2):
        x = res_block(x, 32, i)
    for i in range(2):
        x = res_block(x, 16, i)       
    for i in range(2):
        x = res_block(x, 8, i)    
    for i in range(2):
        x = res_block(x, 4, i)
                 
#    x = MaxPooling1D(pool_size=2)(x)

    r1=Bidirectional(LSTM(4,return_sequences=True,recurrent_regularizer=regularizers.l2(0.01)))(inpt)
    r1=TimeDistributed(Dense(16, activation='relu'))(r1)
    for i in range(2):
        r1 = res_block(r1, 32, i)
    for i in range(2):
        r1 = res_block(r1, 16, i)       
    for i in range(2):
        r1 = res_block(r1, 8, i)    
    for i in range(2):
        r1 = res_block(r1, 4, i)
        
    x=concatenate([x,r1],axis=1)
    for i in range(2):
        x = res_block(x, 32, i)
    for i in range(2):
        x = res_block(x, 16, i)       
    for i in range(2):
        x = res_block(x, 8, i)    
    for i in range(2):
        x = res_block(x, 4, i) 
    
    x = Flatten()(x) 
    #----
    x = Dense(512, activation='relu')(x)
    x=BatchNormalization()(x)   
    x = Dense(128, activation='relu')(x)
    x=BatchNormalization()(x)     
    x = Dense(64, activation='relu')(x)
    x=BatchNormalization()(x)   
    x = Dense(32, activation='relu')(x)
    x=BatchNormalization()(x)     
    #-------------------------------#
    x = Dense(16, activation='relu')(x)
    x=BatchNormalization()(x)   
    x = Dense(8, activation='relu')(x)
    x=BatchNormalization()(x)     
    x = Dense(num_classes, activation='relu',name='main_output')(x)
  
    # Construct the model.
    model = Model(inputs=inpt, outputs=x)
    return model

# In[]
    
# In[]   
def my_reshape(x,a,b):
    return K.reshape(x,(-1,a,b)) 
    
def res_block(x, channels, i):
    if i == 1:  # 第二个block
        strides = 1
        x_add = x
    else:  # 第一个block
        strides = 2
        # x_add 是对原输入的bottleneck操作
        x_add = Conv1D(channels,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       strides=strides)(x)
        x=BatchNormalization()(x)
 
    x = Conv1D(channels,
               kernel_size=3,
               activation='relu',
               padding='same')(x)
    x=BatchNormalization()(x)
    
    x = Conv1D(channels,
               kernel_size=3,
               padding='same',
               strides=strides)(x)
    
    x = add([x, x_add])
    x=Activation(K.relu)(x)
    x=BatchNormalization()(x)
    return x
    
def c_block(x, channels):
    x = Conv1D(channels,
               kernel_size=3,
               activation='relu',
               padding='same')(x)
    
    x=BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    return x    
    
def build_azi_model(input_shape):
    inpt = Input(shape=input_shape,name='wave_input')  
    x = Conv1D(16,  kernel_size=3, activation='relu', input_shape=input_shape,padding='same')(inpt) 
       
    for i in range(3):
        x = c_block(x, 32)     
        
    x1=Lambda(my_reshape,arguments={'a':1500*2,'b':2*3})(inpt)  
    for i in range(2):
        x1 = c_block(x1, 32)  
        
    x2=Lambda(my_reshape,arguments={'a':750*2,'b':4*3})(inpt)  
    for i in range(1):
        x2 = c_block(x2, 32)   
        
    x3=Lambda(my_reshape,arguments={'a':375*2,'b':8*3})(inpt)     
        
    x=concatenate([x,x1,x2,x3])
    
    for i in range(2):
        x = res_block(x, 32, i)
    for i in range(2):
        x = res_block(x, 16, i)       
  
    r1=Bidirectional(LSTM(4,return_sequences=True,recurrent_regularizer=regularizers.l2(0.01)))(inpt)
    r1=TimeDistributed(Dense(16, activation='relu'))(r1)
    for i in range(2):
        r1 = res_block(r1, 32, i)
    for i in range(2):
        r1 = res_block(r1, 32, i) 
    for i in range(2):
        r1 = res_block(r1, 32, i)
    for i in range(2):
        r1 = res_block(r1, 16, i)       
    for i in range(2):
        r1 = res_block(r1, 16, i)   
        
    x=concatenate([x,r1])    
    
    x = Flatten()(x)  
#    x = Dense(64)(x)
#    x = BatchNormalization()(x) 
#    x = Activation(K.relu)(x)
#    x = Dense(32)(x)
#    x = BatchNormalization()(x)    
#    x = Activation(K.relu)(x)
#    x = Dense(16)(x)
#    x = BatchNormalization()(x)    
#    x = Activation(K.relu)(x)  
#    x = Dense(8)(x)
#    x = BatchNormalization()(x)    
#    x = Activation(K.relu)(x)      
#    x = Dense(2)(x)
#    x = BatchNormalization()(x) 
#    x = Activation(K.tanh,name='main_output')(x)
    x = Dense(16, activation='relu')(x)
    x=BatchNormalization()(x)   
    x = Dense(8, activation='relu')(x)
    x=BatchNormalization()(x)     
    x = Dense(2, activation='relu')(x)
    x=BatchNormalization(name='main_output')(x)    
    # Construct the model.
    model = Model(inputs=inpt, outputs=x)
    return model 



