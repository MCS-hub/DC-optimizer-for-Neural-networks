# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:42:05 2021

@author: luu2
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers

class DC1(keras.layers.Layer):
    
    def __init__(self,units=32):
        super(DC1,self).__init__()
        self.units = units
        self.initializer = tf.keras.initializers.GlorotUniform()
        
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],self.units),
            initializer=self.initializer,
            trainable = True,
            dtype = 'float64'
            )
        self.b = self.add_weight(
            shape=(self.units,), initializer=self.initializer,trainable=True,
            dtype='float64'
            )
    
    
    def call(self,inputs):
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b
        
        G = tf.square(ReLu_w + tf.transpose(inputs))
        G = G + tf.square(ReLu_nw)
        G = 0.5*tf.reduce_sum(G,0) + ReLu_b
        
        H = tf.square(ReLu_nw + tf.transpose(inputs))
        H = H + tf.square(ReLu_w)
        H = 0.5*tf.reduce_sum(H,0) + ReLu_nb
        
        varphi = keras.activations.relu(G-H) + G+H
        vartheta = G+H
        return tf.concat([varphi,vartheta],0)
    

class DC2(keras.layers.Layer):
    def __init__(self,units=32):
        super(DC2,self).__init__()
        self.units=units
        self.initializer = tf.keras.initializers.GlorotUniform()
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]/2),self.units),
            initializer = self.initializer,
            trainable = True,
            dtype='float64'
            )
        self.b = self.add_weight(
            shape = (self.units,), initializer=self.initializer,trainable = True,
            dtype='float64'
            )
        
    def call(self,inputs):
        k = inputs.shape[-1]
        varphi = inputs[:,:int(k/2)]
        vartheta = inputs[:,int(k/2):]
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b
        
        G = tf.square(ReLu_w + tf.transpose(varphi))
        G = G + tf.square(ReLu_nw + tf.transpose(vartheta))
        G = 0.5*tf.reduce_sum(G,0) + ReLu_b
        
        H = tf.square(ReLu_nw + tf.transpose(varphi))
        H = H + tf.square(ReLu_w + tf.transpose(vartheta))
        H = 0.5*tf.reduce_sum(H,0) + ReLu_nb
        
        return tf.concat([G,H],0)
    
    
class DC(keras.layers.Layer):
    def __init__(self,units=32,activation=None):
        super(DC,self).__init__()
        self.units=units
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.activation = activation
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]/2),self.units),
            initializer = self.initializer,
            trainable = True,
            dtype='float64'
            )
        self.b = self.add_weight(
            shape = (self.units,), initializer=self.initializer,trainable = True,
            dtype='float64'
            )
        
    def call(self,inputs):
        k = inputs.shape[-1]
        varphi = inputs[:,:int(k/2)]
        vartheta = inputs[:,int(k/2):]
        
        # print("varphi",varphi)
        # print("vartheta",vartheta)
        # print("w",self.w)
        
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b
        
        # print("tf.reduce_sum(tf.square(varphi),1)",tf.reduce_sum(tf.square(varphi),1))
        # print("tf.reduce_sum(tf.square(vartheta),1)",tf.reduce_sum(tf.square(vartheta),1))
        # print("tf.reduce_sum(tf.square(self.w),0)",tf.reduce_sum(tf.square(self.w),0))
        
        temp_term = 0.5*(tf.expand_dims(tf.reduce_sum(tf.square(varphi),1)+tf.reduce_sum(tf.square(vartheta),1),1) 
                         + tf.expand_dims(tf.reduce_sum(tf.square(self.w),0),0))

        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw) + ReLu_b + temp_term 
        H = tf.matmul(varphi,ReLu_nw) + tf.matmul(vartheta,ReLu_w) + ReLu_nb + temp_term
        
        # print("G component:",G)
        # print("H component:",H)
        
        if self.activation:  #for a moment, consider ReLU only
            varphi = keras.activations.relu(G-H) + G+H
            vartheta = G+H
            return tf.concat([varphi,vartheta],1)
        else: 
           return tf.concat([G,H],1)

    
