'''
Stock price forecast competition
Model Training

Authors: Chen-Yi, Darren
Date: 2018.04.30
'''

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

from data_generator import sine_wave


NUM_FEATURES = 1
NUM_OUTPUTS = 5
TIME_STEPS = 50
BATCH_SIZE = 16
EPOCHS = 20
MODEL_FILE = "model_20.h5"


# Build the network model
print("Building the network...")
model = Sequential()
model.add(LSTM(32, return_sequences=True, \
               input_shape=(TIME_STEPS, NUM_FEATURES)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(NUM_OUTPUTS, activation='linear'))
model.compile(loss='mse', optimizer='adam')


# Generate training dataset
print("Generating dataset...")
NUM_EXAMPLES = BATCH_SIZE * 1000
EXAMPLE_LENGTH = TIME_STEPS+NUM_OUTPUTS
x = np.zeros((NUM_EXAMPLES, TIME_STEPS, NUM_FEATURES))
y = np.zeros((NUM_EXAMPLES, NUM_OUTPUTS))
for i in range(NUM_EXAMPLES):
  data = sine_wave(np.random.rand(), np.random.rand()*100+10, \
                   np.random.rand()*6.28, EXAMPLE_LENGTH)
  x[i, :, 0] = data[:TIME_STEPS]
  y[i] = data[-NUM_OUTPUTS:]


# Train the network
print("Training the network...")
model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS)


# Save the model
print("Saving model...")
model.save(MODEL_FILE)

