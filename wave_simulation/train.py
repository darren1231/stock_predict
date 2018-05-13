'''
Stock price forecast competition
Model Training

Authors: Chen-Yi, Darren
Date: 2018.04.30
'''

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import math
from matplotlib import pyplot as plt
from keras.models import load_model

def create_wave(func,amplitude, wavelength, phase, steps):
    """
    Create wave data list
    
    Args:
        fun:  the wave function you want to create such as math.sin or math.cos
        amplitude:
        wavelength:
        phase:
        steps: length of the wave data
    
    Return:
        wave data list:
    """
    wave_data_list = \
    [amplitude * func(2*math.pi*i/wavelength + phase) for i in range(steps)]
    
    return wave_data_list

def plot_function(y_list):
    """
    According to the length of y_list, create a x_list and plot
    """
    x_list = [i for i in range(len(y_list))]
    plt.plot(x_list,y_list)
    plt.show()
    
def test_function():
#    amplitude = np.random.rand()
#    wavelength = np.random.rand()*100+10
    sin_wave_data = create_wave(func=lambda x :math.sin(x),
                                amplitude=1, 
                                wavelength=wavelength,
                                phase=0, 
                                steps=100)
    
    cos_wave_data = create_wave(func=lambda x :math.cos(x),
                                amplitude=1, 
                                wavelength=wavelength,
                                phase=0, 
                                steps=100)
    
    plot_function(sin_wave_data)
    plot_function(cos_wave_data)
    plot_function(np.array(sin_wave_data)+np.array(cos_wave_data))


NUM_FEATURES = 1
NUM_OUTPUTS = 5
TIME_STEPS = 50
BATCH_SIZE = 16
EPOCHS =2
MODEL_FILE = "model.h5"


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

amplitude = np.random.rand()
wavelength = np.random.rand()*100+10

for i in range(NUM_EXAMPLES):
  data  = create_wave(func=lambda x :math.sin(x),
                                    amplitude=1, 
                                    wavelength=10,
                                    phase=0, 
                                    steps=EXAMPLE_LENGTH)
  x[i, :, 0] = data[:TIME_STEPS]
  y[i] = data[-NUM_OUTPUTS:]


# Train the network
print("Training the network...")
model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS)


# Save the model
print("Saving model...")
model.save(MODEL_FILE)

