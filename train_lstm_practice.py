# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:59:13 2018

@author: darren
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed,Activation

import numpy as np



def test_model1():
    NUM_FEATURES = 1
    NUM_OUTPUTS = 5
    TIME_STEPS = 50
    BATCH_SIZE = 64
    EPOCHS = 1
    MODEL_FILE = "model.h5"

    # Build the network model
    print("Building the network...")
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, \
                   input_shape=(TIME_STEPS, NUM_FEATURES)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(NUM_OUTPUTS, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    print (model.summary())
    
    random_training_data = np.random.random((100,TIME_STEPS,1))
    random_testing_data = np.random.random((100,5))
    
    
    
    # Train the network
    print("Training the network...")
    model.fit(random_training_data, random_testing_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    print("Testing the network...")
    random_test_input = np.random.random((1,TIME_STEPS,1))
    predict = model.predict(random_test_input)
    print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)
    
    random_test_input = np.random.random((2,TIME_STEPS,1))
    predict = model.predict(random_test_input)
    print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)

def test_model2():
    NUM_FEATURES = 2
    NUM_OUTPUTS = 5
    TIME_STEPS = 50
    BATCH_SIZE = 64
    EPOCHS = 1
    MODEL_FILE = "model.h5"
    
    # Build the network model
    print("Building the network...")
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, \
                   input_shape=(TIME_STEPS, NUM_FEATURES)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(NUM_OUTPUTS, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    print (model.summary())
    
    random_training_data = np.random.random((100,TIME_STEPS,NUM_FEATURES))
    random_testing_data = np.random.random((100,5))
    
    
    
    # Train the network
    print("Training the network...")
    model.fit(random_training_data, random_testing_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    print("Testing the network...")
    random_test_input = np.random.random((1,TIME_STEPS,NUM_FEATURES))
    predict = model.predict(random_test_input)
    print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)
    
    random_test_input = np.random.random((2,TIME_STEPS,NUM_FEATURES))
    predict = model.predict(random_test_input)
    print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)

def test_model3():
    NUM_FEATURES = 3
    NUM_OUTPUTS = 5
    TIME_STEPS = 50
    BATCH_SIZE = 64
    EPOCHS = 1
    MODEL_FILE = "model.h5"
    
    # Build the network model
    print("Building the network...")
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, \
                   input_shape=(TIME_STEPS, NUM_FEATURES)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(TimeDistributed(Dense(5)))
    model.add(Activation("linear"))
    
    model.compile(loss='mse', optimizer='adam')
    print (model.summary())
    
    random_training_data_x = np.random.random((100,TIME_STEPS,NUM_FEATURES))
    random_training_data_y = np.random.random((100,TIME_STEPS,5))
    
    
    
    # Train the network
    print("Training the network...")
    
    print ("Training input shape is",random_training_data_x.shape,
           " Training output shape is",random_training_data_y.shape)
    
    model.fit(random_training_data_x, random_training_data_y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    print("Testing the network...")
    random_test_input = np.random.random((1,TIME_STEPS,NUM_FEATURES))
    predict = model.predict(random_test_input)
    print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)
    
    random_test_input = np.random.random((2,TIME_STEPS,NUM_FEATURES))
    predict = model.predict(random_test_input)
    print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)

NUM_FEATURES = 3
NUM_OUTPUTS = 5
TIME_STEPS = 50
BATCH_SIZE = 64
EPOCHS = 1
MODEL_FILE = "model.h5"

# Build the network model
print("Building the network...")
model = Sequential()
model.add(LSTM(32, return_sequences=True, \
               input_shape=(TIME_STEPS, NUM_FEATURES)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add((Dense(5,activation='linear')))


model.compile(loss='mse', optimizer='adam')
print (model.summary())

random_training_data_x = np.random.random((100,TIME_STEPS,NUM_FEATURES))
random_training_data_y = np.random.random((100,TIME_STEPS,5))



# Train the network
print("Training the network...")

print ("Training input shape is",random_training_data_x.shape,
       " Training output shape is",random_training_data_y.shape)

model.fit(random_training_data_x, random_training_data_y, batch_size=BATCH_SIZE, epochs=EPOCHS)

print("Testing the network...")
random_test_input = np.random.random((1,TIME_STEPS,NUM_FEATURES))
predict = model.predict(random_test_input)
print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)

random_test_input = np.random.random((2,TIME_STEPS,NUM_FEATURES))
predict = model.predict(random_test_input)
print ("Input shape is",random_test_input.shape," Output shape is:",predict.shape)