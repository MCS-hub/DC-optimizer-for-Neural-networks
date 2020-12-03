# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:21:38 2020

@author: Phuc Hau
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#functional API to build graph of layers
tf.keras.backend.set_floatx('float64')

class DC1(keras.layers.Layer):
    
    def __init__(self,units=32):
        super(DC1,self).__init__()
        self.units = units
        self.initializer = tf.keras.initializers.LecunNormal()
        
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
        self.initializer = tf.keras.initializers.LecunNormal()
    
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


inputs = tf.keras.Input(shape=(784,))

dense = layers.Dense(64,activation='relu')
DC1layer = DC1(64)
DC2layer = DC2(10)

x = dense(inputs)
x = DC1layer(x)
x = tf.expand_dims(x,axis=0)
x = DC2layer(x)

model = keras.Model(inputs=inputs,outputs=x)
optimizer = keras.optimizers.Adam(learning_rate = 0.01)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.summary()



#-------------------------------------------------
#Test model - verified!
inputs = tf.random.normal(shape=(1,784))
output = model(inputs)
G = output[:10]
H = output[10:]
output1 = G-H

dense2 = layers.Dense(64,activation='relu')
dense3 = layers.Dense(10)
dense3(dense2(dense(inputs)))

dense2.trainable_weights[0].assign(DC1layer.w)
dense2.trainable_weights[1].assign(DC1layer.b)
dense3.trainable_weights[0].assign(DC2layer.w)
dense3.trainable_weights[1].assign(DC2layer.b)

output2 = dense3(dense2(dense(inputs)))

#-------------------------------------------------

#Train a DC neural network
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float64')/255
x_test = x_test.reshape(10000,784).astype('float64')/255
x_test = x_test[:64,:]
y_test = y_test[:64]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
batch_size = 20
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
N_iter = 10

history = []

#optimizer = keras.optimizers.Adagrad(learning_rate = 0.0001)
for step, (x_batch, y_batch) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        H = 0
        for j in range(batch_size):
            output = model(tf.expand_dims(x_batch[j,:],axis=0))
            idx = tf.dtypes.cast(y_batch[j], tf.int32)
            Delta = output[:10]
            Psi = output[10:]
            H = H + tf.reduce_sum(Delta+Psi) + Delta[idx]

    gradH = tape.gradient(H,model.trainable_weights)
    H0 = H
    H0x = 0
    for idx in range(len(gradH)):
        H0x = H0x + tf.reduce_sum(gradH[idx]*model.trainable_weights[idx])
    
  
    sub_history = []
    for i in range(N_iter):
        optimizer = keras.optimizers.Adagrad(learning_rate = 0.0000001)
        with tf.GradientTape() as tape:
            G = 0
            H = 0
            for j in range(batch_size):
                output = model(tf.expand_dims(x_batch[j,:],axis=0))
                idx = tf.dtypes.cast(y_batch[j], tf.int32)
                Delta = output[:10]
                Psi = output[10:]
                G = G+ tf.math.log(tf.math.reduce_sum(tf.math.exp(Delta-Psi))) \
                    +tf.reduce_sum(Delta+Psi) + Psi[idx]
                H = H + tf.reduce_sum(Delta+Psi) + Delta[idx]
        Hx = 0        
        for idx in range(len(gradH)):
            Hx = Hx + tf.reduce_sum(gradH[idx]*model.trainable_weights[idx])
            
        sub_history.append(((G-Hx-H0+H0x)/batch_size).numpy())
        
        res = H - H0 - Hx + H0x
        print("G(x)-H(x0)-<gradH(x0),x-x0>: ", ((G-Hx-H0+H0x)/batch_size).numpy())
        #print("G(x)-H(x)                  : ",((G-H)/batch_size).numpy())
        #print("Residual                   : ", res.numpy())
        

        gradG = tape.gradient(G, model.trainable_weights)
        grad = [(gradG[pst]-gradH[pst])/batch_size for pst in range(len(gradH))]
        optimizer.apply_gradients(zip(grad,model.trainable_weights))
        
    #break
    if step%1 ==0:
        loss = 0
        for (x_online_test, y_online_test) in test_dataset:
            output = model(tf.expand_dims(x_online_test,axis=0))
            idx = tf.dtypes.cast(y_online_test,tf.int32)
            Delta = output[:10]
            Psi = output[10:]
            model_prob = keras.layers.Softmax()(Delta-Psi)
            loss = loss +loss_fn(idx.numpy(),model_prob.numpy())
            
        loss = (loss/64).numpy()    
        print("test loss:: ",loss)
        history.append(loss)
        
#-----------------------------------------------------------------------------
# # #Test deterministic DCA
# (x_train,y_train), _ = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000,784).astype('float64')/255
    
# batch_size = 64
# x_train = x_train[:batch_size,:]
# y_train = y_train[:batch_size]

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))

# num_epoch = 20

# N_iter = 10

# for epoch in range(num_epoch):
#     for (x_batch,y_batch) in train_dataset:
#         with tf.GradientTape() as tape:
#             H  = 0
#             for j in range(batch_size):
#                 output = model(tf.expand_dims(x_batch[j,:],axis=0))
#                 idx = tf.dtypes.cast(y_batch[j], tf.int32)
#                 Delta = output[:10]
#                 Psi = output[10:]
#                 H = H + tf.reduce_sum(Delta+Psi) + Delta[idx]
#         gradH = tape.gradient(H,model.trainable_weights)
        
#         for i in range(N_iter):
#             optimizer = keras.optimizers.Adagrad(learning_rate = 0.0001)
#             with tf.GradientTape() as tape:
#                 G = 0
#                 H = 0
#                 for j in range(batch_size):
#                     output = model(tf.expand_dims(x_batch[j,:],axis=0))
#                     idx = tf.dtypes.cast(y_batch[j], tf.int32)
#                     Delta = output[:10]
#                     Psi = output[10:]
#                     G = G+ tf.math.log(tf.math.reduce_sum(tf.math.exp(Delta-Psi))) \
#                         +tf.reduce_sum(Delta+Psi) + Psi[idx]
#                     H = H + tf.reduce_sum(Delta+Psi) + Delta[idx]
#             gradG = tape.gradient(G,model.trainable_weights)
#             grad = [gradG[pst] - gradH[pst] for pst in range(len(gradH))]
#             optimizer.apply_gradients(zip(grad,model.trainable_weights))
            
        
            
                
    
    


# # inputs = tf.keras.Input(shape=(784,))

# # dense = layers.Dense(64,activation='relu')
# # DC1layer = DC1(64)
# # DC2layer = DC2(10)

# # x = dense(inputs)
# # x=DC1layer(x)
# # x = tf.expand_dims(x,axis=0)
# # x=DC2layer(x)

# # model = keras.Model(inputs=inputs,outputs=x)
# # optimizer = keras.optimizers.Adam(learning_rate = 0.01)
# # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# # model.summary()

# # batch_size = 64
# # (x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
# # x_train = x_train.reshape(60000,784).astype('float64')/255
# # x_test = x_test.reshape(10000,784).astype('float64')/255
# # x_test = x_test[:64,:]
# # y_test = y_test[:64]
# # x_train = x_train[:batch_size,:]
# # x_test = x_test[:batch_size]

# # N_outer = 100
# # N_inner = 100

# # for i in range(N_outer):
# #     with tf.GradientTape() as tape:
# #         H = 0
# #         for j in range(batch_size):
# #             output = model(np.expand_dims(x_train[j,:],axis=0))
# #             idx = y_train[j]
# #             Delta = output[:10]
# #             Psi = output[10:]
# #             H = H + tf.reduce_sum(Delta+Psi) + Delta[idx]
# #     gradH = tape.gradient(H,model.trainable_weights)
    
# #     optimizer = keras.optimizers.Adam(learning_rate=0.001)
# #     print('here')
# #     for k in range(N_inner):
# #         with tf.GradientTape() as tape:
# #             G = 0
# #             for j in range(batch_size):
# #                 output = model(np.expand_dims(x[j,:],axis=0))
# #                 idx = y_train[j]
# #                 Delta = output[:10]
# #                 Psi = output[10:]
# #                 G = G+ tf.math.log(tf.math.reduce_sum(tf.math.exp(Delta-Psi))) \
# #                     +tf.reduce_sum(Delta+Psi) + Psi[idx]
# #         gradG = tape.gradient(G,model.trainable_weights)
# #         grad = [(gradG[pst]-gradH[pst])/batch_size for pst in range(len(gradH))]
# #         optimizer.apply_gradients(zip(grad,model.trainable_weights))
    
# #     G = 0
# #     H = 0
# #     for j in range(batch_size):
# #         output = model(np.expand_dims(x[j,:],axis=0))
# #         idx = y_train[j]
# #         Delta = output[:10]
# #         Psi = output[10:]
# #         G = G+ tf.math.log(tf.math.reduce_sum(tf.math.exp(Delta-Psi))) \
# #             +tf.reduce_sum(Delta+Psi) + Psi[idx]
# #         H = H + tf.reduce_sum(Delta+Psi) + Delta[idx]
# #     print((G-H).numpy())


