# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 00:11:23 2021

@author: Phuc Hau
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers
from special_layers import *
from utils import *
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')

# Build the DC neural network

# initialize layers
inputs = tf.keras.Input(shape=(784,))
Dense = layers.Dense(64,activation='relu')
DC1 = DC(units=64,activation='relu')
DC2 = DC(units=10)

# stack layers together
x = Dense(inputs)
x = tf.concat([x,0*x],axis=1)
x = DC1(x)
x = DC2(x)

dc_model = keras.Model(inputs=inputs,outputs=x,name='dc_model')
dc_model.summary()


# Train a DC neural network
# Setup
batch_size = 64
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float64')/255
x_test = x_test.reshape(10000,784).astype('float64')/255


train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))

test_loss = []
residual = []
cvx_sub_prob = []
model_len = len(dc_model.trainable_weights)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

N_iter = 5 # number of iterations used to solve convex subproblem
epochs = 1

learning_rate = 0.000000001
damping_const = 80
eps = 1
k_max = 100
for epoch in range(epochs):
    for step, (x_batch,y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = dc_model(x_batch)
            delta = outputs[:,:10]
            psi = outputs[:,10:]
            current_batch_size = y_batch.shape[0]
            y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
            H = tf.reduce_sum(delta + psi,axis=1) + tf.reduce_sum(delta*y_one_hot,1)
            H = tf.reduce_sum(H)/current_batch_size
        gradH = tape.gradient(H,dc_model.trainable_weights)
        
        H0 = H
        gradH0x0 = 0
        for idx in range(model_len):
            gradH0x0 += tf.reduce_sum(gradH[idx]*dc_model.trainable_weights[idx])
        
        optimizer = keras.optimizers.Adagrad(learning_rate = learning_rate)
        
        con_grad0 = [0*elem for elem in gradH]  #xem lai
        for i in range(N_iter):
            
            with tf.GradientTape() as tape:
                outputs = dc_model(x_batch)
                delta = outputs[:,:10]
                psi = outputs[:,10:]
                G = tf.math.log(tf.reduce_sum(tf.math.exp(delta-psi),axis=1))\
                    +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                G = tf.reduce_sum(G)/current_batch_size
                
            gradG = tape.gradient(G, dc_model.trainable_weights)
            ngrad = [-gradG[idx]+gradH[idx] for idx in range(model_len)]
            
            con_grad = con_grad0
            with tf.GradientTape() as out_tape:
                with tf.GradientTape() as in_tape:
                    outputs = dc_model(x_batch)
                    delta = outputs[:,:10]
                    psi = outputs[:,10:]
                    G = tf.math.log(tf.reduce_sum(tf.math.exp(delta-psi),axis=1))\
                        +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                    G = tf.reduce_sum(G)/current_batch_size
                gradG = in_tape.gradient(G,dc_model.trainable_weights)
                elemwise_products = [
                    math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                    for grad_elem, v_elem in zip(gradG, con_grad)
                    if grad_elem is not None]
            Hess_vec = out_tape.gradient(elemwise_products,dc_model.trainable_weights)
            reg_Hess_vec = [Hess_vec[idx]+damping_const*con_grad[idx] for idx in range(model_len)]
            r = [ngrad[idx]-reg_Hess_vec[idx] for idx in range(model_len)]
            p = r
            k = 0
            if norm_square(r) >= eps:
                while True:
                    with tf.GradientTape() as out_tape:
                        with tf.GradientTape() as in_tape:
                            outputs = dc_model(x_batch)
                            delta = outputs[:,:10]
                            psi = outputs[:,10:]
                            G = tf.math.log(tf.reduce_sum(tf.math.exp(delta-psi),axis=1))\
                                +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                            G = tf.reduce_sum(G)/current_batch_size
                        gradG = in_tape.gradient(G,dc_model.trainable_weights)
                        elemwise_products = [
                            math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                            for grad_elem, v_elem in zip(gradG, p)
                            if grad_elem is not None]
                    Hess_vec = out_tape.gradient(elemwise_products,dc_model.trainable_weights)
                    reg_Hess_vec = [Hess_vec[idx]+damping_const*p[idx] for idx in range(model_len)]
                    rdotr = sum([tf.reduce_sum(r[idx]*r[idx]) for idx in range(model_len)])
                    alpha = rdotr/sum([tf.reduce_sum(p[idx]*reg_Hess_vec[idx]) for idx in range(model_len)])
                    con_grad = [con_grad[idx]+alpha*p[idx] for idx in range(model_len)]
                    r_next = [r[idx]-alpha*reg_Hess_vec[idx] for idx in range(model_len)]
                    if norm_square(r_next) < eps or k>k_max:
                        break
                    rnextdotrnext = sum([tf.reduce_sum(r_next[idx]*r_next[idx]) for idx in range(model_len)])
                    beta = rnextdotrnext/rdotr
                    r = r_next
                    p = [r[idx]+beta*p[idx] for idx in range(model_len)]
                    k += 1
               
                    
            for pst in range(model_len):
                dc_model.trainable_weights[idx].assign_add(con_grad[idx])
        
        if step%1==0:
            outputs = dc_model(x_test)
            delta = outputs[:,:10]
            psi = outputs[:,10:]
            model_prob = keras.layers.Softmax()(delta-psi)
            loss = loss_fn(y_test,model_prob).numpy()
            test_loss.append(loss)
            print("loss:",loss)
                
 