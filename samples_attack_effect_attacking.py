#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from deepbayes import PosteriorModel
import pickle
from run_trial import preprocess_data


import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow import keras             #importing keras as wrapper for tensorflow
# https://blog.floydhub.com/introduction-to-adversarial-machine-learning/#fgsm


def BNN_FGSM(model, inp, loss_fn, eps,n=35):
    """
    model 	-  a keras model
    inp		-  a tensorflow tensor or numpy array
    loss_fn	- a tensorflow loss function (so things are differentiable)
    eps 	- a tensorflow tensor or numpy array
    """
    inp = inp.reshape(-1, inp.shape[0])
    #  type(model) = <class 'deepbayes.optimizers.adam.Adam'>

    # set your max and min vector bounds:
    inp = np.asarray(inp)
    vector_max = inp + eps
    vector_min = inp - eps
    inp = tf.convert_to_tensor(inp)
    # Get the original prediction you want to attack
    # set this to true class label if you want
    temp = np.squeeze(model.predict(inp,n=n))
    direction = np.zeros(len(temp))  # set this to true class label if you want
    direction[np.argmax(temp)] = 1
    direction = direction.reshape(-1, direction.shape[0])
    # direction = np.argmax(direction)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inp)
        #  predictions = model(inp)
        predictions = model.predict(inp,n=n)
        loss = loss_fn(direction, predictions)

    # Computed input gradient
    inp_gradient = tape.gradient(loss, inp)

    sign = np.sign(inp_gradient)

    # add adversarial noise to the input
    adv = inp + eps*np.asarray(sign)

    # clip it between your min and your max
    adv = np.clip(adv, vector_min, vector_max)

    # clip it to be a valid input
    adv = np.clip(adv, 0, 1)

    return adv






p = PosteriorModel('VI_for_samples_attack')


epsilon = 0.2

#x_test = pickle.load('x_test')

file = open('x_test', 'rb')

# dump information to that file
x_test = pickle.load(file)
file.close()


x_test = x_test[:100]
differences = []

x_test_predictions = p.predict(x_test,n=100)

epsilons = np.full(100, epsilon)

# Index 58 is the feature for gender (0 for Female, 1 for Male)
epsilons[58] = 1.0
adversarial_examples = np.ndarray(shape=(x_test.shape))

number_of_samples = 1
how_many_times = 10
max_diff =[]
for _ in range(how_many_times):
    print('here')
    for i in range(len(x_test)):
        adversarial = BNN_FGSM(
                p, x_test[i], keras.losses.categorical_crossentropy, epsilons,n=number_of_samples)
        adversarial_examples[i] = adversarial
    
    x_test_adversarial_predictions = p.predict(adversarial_examples,n=number_of_samples)
    
    
    
    for i in range(len(x_test_predictions)):
                difference = abs(x_test_predictions[i][0] -
                                 x_test_adversarial_predictions[i][0])
                differences.append(difference)
    max_diff.append(max(differences).numpy())




