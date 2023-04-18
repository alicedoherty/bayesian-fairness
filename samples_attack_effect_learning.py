#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepensemble import DeepEnsemble
from run_trial import preprocess_data
from tensorflow import keras
import deepbayes.optimizers as optimizers
import numpy as np
import pickle


x_train, x_test, y_train, y_test = preprocess_data()
file = open('x_test', 'wb')
# dump information to that file
pickle.dump(x_test, file)
file.close()

epsilon = 0.1

input_shape = x_train.shape[1]
num_classes = y_train.shape[1]


neuron_num = 16

layer_num = 3

###VI

model_VI = keras.Sequential()
model_VI.add(keras.Input(shape=input_shape))

for _ in range(layer_num):
    model_VI.add(keras.layers.Dense(
        neuron_num, activation="relu"))
    
model_VI.add(keras.layers.Dense(
                num_classes, activation="softmax"))


model_VI.summary()

# Initialising some training parameters
loss = keras.losses.SparseCategoricalCrossentropy()
optimizer = optimizers.VariationalOnlineGuassNewton()
#optimizer = optimizers.HamiltonianMonteCarlo()
batch_size = 128
epochs = 15
# validation_split = 0.1

model_VI = optimizer.compile(
    model_VI, loss_fn=loss, batch_size=batch_size, epochs=epochs)

model_VI.train(x_train, np.argmax(y_train,axis=1), x_test,np.argmax(y_test,axis=1))

model_VI.save('VI_for_samples_attack')

###HMC

model_HMC = keras.Sequential()
model_HMC.add(keras.Input(shape=input_shape))

for _ in range(layer_num):
    model_HMC.add(keras.layers.Dense(
        neuron_num, activation="relu"))
    
model_HMC.add(keras.layers.Dense(
                num_classes, activation="softmax"))


model_HMC.summary()

# Initialising some training parameters
loss = keras.losses.SparseCategoricalCrossentropy()
optimizer = optimizers.HamiltonianMonteCarlo()
batch_size = 128
epochs = 75
# validation_split = 0.1

model_HMC = optimizer.compile(
    model_HMC, loss_fn=loss, batch_size=batch_size, epochs=epochs)

model_HMC.train(x_train, np.argmax(y_train,axis=1), x_test,np.argmax(y_test,axis=1))

model_HMC.save('HMC_for_samples_attack')


###Deep ensemble

model_DE = keras.Sequential()
model_DE.add(keras.Input(shape=input_shape))

for _ in range(layer_num):
    model_DE.add(keras.layers.Dense(
        neuron_num, activation="relu"))
    
model_DE.add(keras.layers.Dense(
                num_classes, activation="softmax"))


model_DE.summary()

# Initialising some training parameters
loss = keras.losses.SparseCategoricalCrossentropy()
optimizer = DeepEnsemble()
batch_size = 128
epochs = 15
# validation_split = 0.1

model_DE = optimizer.compile(
    model_DE, loss_fn=loss, batch_size=batch_size, epochs=epochs,num_models=75)

model_DE.train(x_train, np.argmax(y_train,axis=1), x_test,np.argmax(y_test,axis=1))

model_DE.save('DE_for_samples_attack')


