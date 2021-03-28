# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:38:03 2021

@author: luu2
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


#normal model-----------------------------------------------------------------
inputs = tf.keras.Input(shape=(784,))

x = layers.Dense(64,activation='relu')(inputs)
x =  layers.Dense(64,activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs,name='normal_model')
model.summary()

#train normal model-----------------------------------------------------------
batch_size=64
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype("float64")/255
x_test = x_test.reshape(10000, 784).astype("float64") / 255
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))

# # model--compile
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
#     )

# history = model.fit(x_train,y_train,batch_size=64,epochs=150,validation_split=0.2)
# test_score = model.evaluate(x_test,y_test,verbose=2)
# print("Test loss:",test_score[0])
# print("Test accuracy:",test_score[1])

# model: hand-design a training loop:

scce = tf.keras.losses.SparseCategoricalCrossentropy()  #cross-entropy loss
opt = tf.keras.optimizers.SGD(learning_rate=0.1)  #Adam optimizer
history = []

# compute the loss at the beginning
output_test = tf.keras.activations.softmax(model(x_test),axis=-1)
y_test = tf.convert_to_tensor(y_test,dtype=tf.float64)
test_loss = scce(y_test,output_test)
print("test_loss:",test_loss)
history.append(test_loss)

epochs = 1

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    
    i_count = 0
    for step, (x_batch,y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            output = tf.keras.activations.softmax(model(x_batch), axis=-1)
            loss = scce(y_batch,output)
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads,model.trainable_weights))
        i_count+=1
        
        if i_count%20==0:
            output_test = tf.keras.activations.softmax(model(x_test),axis=-1)
            y_test = tf.convert_to_tensor(y_test,dtype=tf.float64)
            test_loss = scce(y_test,output_test)
            print("test_loss:",test_loss.numpy())
            history.append(test_loss)
    
#-----------------------------------------------------------------------------
# #Test dc_model - verified!
# inputs = tf.random.normal(shape=(1,784))
# output = dc_model(inputs)
# G = output[:10]
# H = output[10:]
# output1 = G-H

# dense2 = layers.Dense(64,activation='relu')
# dense3 = layers.Dense(10)
# dense3(dense2(dense(inputs)))

# dense2.trainable_weights[0].assign(DC1layer.w)
# dense2.trainable_weights[1].assign(DC1layer.b)
# dense3.trainable_weights[0].assign(DC2layer.w)
# dense3.trainable_weights[1].assign(DC2layer.b)

# output2 = dense3(dense2(dense(inputs)))

# print("output1:",output1)
# print("output2:",output2)

#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
# Verify the dc model:

# create the normal model to verify
inputs = tf.keras.Input(shape=(784,))
dense1 = layers.Dense(64,activation='relu')
dense2 = layers.Dense(64,activation='relu')
dense3 = layers.Dense(10)

x = dense1(inputs)
x = dense2(x)
x = dense3(x)

normal_model = keras.Model(inputs=inputs,outputs=x,name='normal_model')

# test value
inputs = tf.random.normal(shape=(3,784))
output = dc_model(inputs)
G = output[:,:10]
H = output[:,10:]
output1 = G-H

dense1.trainable_weights[0].assign(Dense.kernel)
dense1.trainable_weights[1].assign(Dense.bias)
dense2.trainable_weights[0].assign(DC1.w)
dense2.trainable_weights[1].assign(DC1.b)
dense3.trainable_weights[0].assign(DC2.w)
dense3.trainable_weights[1].assign(DC2.b)

output2 = normal_model(inputs)

print("output1:",output1)
print("output2:",output2)

# test gradient------------------------

x_batch = inputs
y_batch = tf.constant([5,9,2])
# =============================================================================
# with tf.GradientTape() as tape:
#     outputs = dc_model(inputs)
#     delta = outputs[:,:10]
#     psi = outputs[:,10:]
#     current_batch_size = y_batch.shape[0]
#     indices = tf.dtypes.cast(y_batch,tf.int32)+10*tf.range(current_batch_size)
#     H = tf.reduce_sum(delta + psi,axis=1) + np.take(delta,indices)
#     H = tf.reduce_sum(H)/current_batch_size
# gradH = tape.gradient(H,dc_model.trainable_weights)
# 
# with tf.GradientTape() as tape:
#     outputs = dc_model(x_batch)
#     delta = outputs[:,:10]
#     psi = outputs[:,10:]
#     indices = tf.dtypes.cast(y_batch,tf.int32)+10*tf.range(current_batch_size)
#     G = tf.math.log(tf.reduce_sum(tf.math.exp(delta-psi),axis=1))+tf.reduce_sum(delta+psi,axis=1)+np.take(psi,indices)
#     G = tf.reduce_sum(G)/current_batch_size
# 
# gradG = tape.gradient(G, dc_model.trainable_weights)
# grad1 = [gradG[idx]-gradH[idx] for idx in range(model_len)]
# =============================================================================
with tf.GradientTape() as tape:
    outputs = dc_model(x_batch)
    delta = outputs[:,:10]
    psi = outputs[:,10:]
    y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
    #indices = tf.dtypes.cast(y_batch,tf.int32)+10*tf.range(current_batch_size)
    G = tf.math.log(tf.reduce_sum(tf.math.exp(delta-psi),axis=1))\
    +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
    G = tf.reduce_sum(G)/current_batch_size
    H = tf.reduce_sum(delta + psi,axis=1) + tf.reduce_sum(delta*y_one_hot,1)
    H = tf.reduce_sum(H)/current_batch_size
    F = G - H

gradF = tape.gradient(F, dc_model.trainable_weights)
print("G-H",F.numpy())
print("Grad1",gradF[1])

with tf.GradientTape() as tape:
    output2 = keras.layers.Softmax()(normal_model(inputs))
    loss = scce(y_batch,output2)
    
grad2 = tape.gradient(loss, normal_model.trainable_weights)
print("Grad2",grad2[1])
print("F",loss.numpy())

    



#-----------------------------------------------------------------------------
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
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#------------------------------------------------------------------------------
#Train using available optimizers for convex subproblems
learning_rate = 0.1
#learning_rate = 0.01
N_iter = 1 # number of iterations used to solve convex subproblem
epochs = 1

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
        
        optimizer = keras.optimizers.SGD(learning_rate = learning_rate)
        for i in range(N_iter):
            with tf.GradientTape() as tape:
                outputs = dc_model(x_batch)
                delta = outputs[:,:10]
                psi = outputs[:,10:]
                #y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
                G = tf.math.log(tf.reduce_sum(tf.math.exp(delta-psi),axis=1))\
                    +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                G = tf.reduce_sum(G)/current_batch_size
                
                #-------------------------------
                H = tf.reduce_sum(delta + psi,axis=1) + tf.reduce_sum(delta*y_one_hot,1)
                H = tf.reduce_sum(H)/current_batch_size
                #-------------------------------
            
            gradH0x = 0
            for idx in range(model_len):
                gradH0x += tf.reduce_sum(gradH[idx]*dc_model.trainable_weights[idx])
            
            res = (H - H0 - gradH0x + gradH0x0).numpy()
            residual.append(res)
            cvx_sub_prob.append((G-H0-gradH0x+gradH0x0).numpy())
            #print("residual: ",res)
            print("G(x)-H(x0)-<gradH(x0),x-x0>: ",(G-H0-gradH0x+gradH0x0).numpy())
                
                
        
            gradG = tape.gradient(G, dc_model.trainable_weights)
            grad = [gradG[idx]-gradH[idx] for idx in range(model_len)]
            optimizer.apply_gradients(zip(grad,dc_model.trainable_weights))
        print('---------------------------------------------------------')
    
        if step%20==0:
            outputs = dc_model(x_test)
            delta = outputs[:,:10]
            psi = outputs[:,10:]
            model_prob = keras.layers.Softmax()(delta-psi)
            y_test_one_hot = tf.dtypes.cast(y_test,tf.int32)
            loss = loss_fn(y_test,model_prob.numpy()).numpy()
            test_loss.append(loss)
            print("loss:",loss)
print("end")
#-----------------------------------------------------------------------------
#Train using Hessian-free for convex subproblems


