import os
import sys
import copy
import random
sys.path.append('..')
import numpy as np
import pandas as pd
import folktables
from folktables import generate_categories
from datetime import datetime

from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.model_selection import train_test_split
from indiv_utils import *

print("Main imports done")

print("Importing custom libs... this  may take a minute")
import folk_utils
import indiv_utils
import deepbayes
import tensorflow as tf
from tensorflow import keras
import deepbayes.optimizers as optimizers

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from run_trial import get_fairness_score_basic, get_fairness_score

X_train, X_test, X_val, y_train, y_test, y_val = folk_utils.get_dataset("Folk")

max_vec = np.max(np.concatenate((X_train,X_test,X_val)),axis=0)

X_train = X_train/max_vec
X_val = X_val/max_vec
X_test = X_test/max_vec

X_test = X_test[:1000]
y_test = y_test[:1000]

S_fair = np.linalg.inv(indiv_utils.sensr_metric(X_train, [-1]))



epsilon = 0.2


# Number of hidden layers in the model
layers = [1, 2, 3, 4, 5]
#layers = [1, 2]
# Number of neurons per hidden layer in the model
neurons = [64, 32, 16, 8, 4, 2]
#neurons = [64, 32]

# Measurements we're recording during the trials
measurements = [ "BNNAccuracy",  "BNNBasicScore",
                "BNNMaxDifference",  "BNNMinDifference",  "BNNMeanDifference"]

df = pd.DataFrame(columns=measurements)
for neuron_num in neurons:
    for layer_num in layers:
        model = keras.Sequential()
        model.add(keras.Input(shape=(43)))

        for _ in range(layer_num):
            model.add(keras.layers.Dense(
                neuron_num, activation="relu"))
            
        model.add(keras.layers.Dense(
                2, activation="softmax"))
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        model.summary()

        optimizer = optimizers.VariationalOnlineGuassNewton()
        batch_size = 128
        epochs = 15

        model = optimizer.compile(
            model, loss_fn=loss, batch_size=batch_size, epochs=epochs)

        model.train(X_train, y_train, X_val, y_val)


        x_test_predictions = model.predict(X_test,n=35)
        
        x_test_adversarial = np.ndarray(shape=(X_test.shape))
        #x_test_adversarial_predictions = np.ndarray(shape=(x_test_predictions.shape))
        
        for i in range(len(x_test_adversarial)):
            
            x_test_adversarial[i] = fPGD(model, np.asarray([X_test[i]]), loss, epsilon, S_fair)
        
        x_test_adversarial_predictions = model.predict(x_test_adversarial,n=35)
        test_acc = np.mean(np.argmax(x_test_predictions, axis=1)
                           == (y_test))
        accuracy = test_acc
        
        basic_score = get_fairness_score_basic(
        x_test_predictions, x_test_adversarial_predictions, "BNN")
        
        max_diff, min_diff, avrg_diff = get_fairness_score(x_test_predictions,
                                                       x_test_adversarial_predictions, "BNN")
        
        new_row = pd.DataFrame([accuracy, basic_score, max_diff,
                                     min_diff,  avrg_diff], index=measurements, columns=[f"L{layer_num}N{neuron_num}"]).T
        df = pd.concat((df, new_row))
        
 # Pandas options to display all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)
# np.set_printoptions(linewidth=100000)

f = open(f"./results/trial_{datetime.now()}_eps_{epsilon}_pgd_folks_bnn.csv", 'a')
print(df, file=f)
f.close()