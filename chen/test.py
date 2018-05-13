'''
Stock price forecast competition
Model Testing

Authors: Chen-Yi, Darren
Date: 2018.05.2
'''

from keras.models import load_model
import numpy as np

from data_generator import sine_wave


NUM_FEATURES = 1
NUM_OUTPUTS = 5
TIME_STEPS = 50
MODEL_FILE = "model_20.h5"
RESULT_FILE = "result.csv"

# Load the trained model
model = load_model(MODEL_FILE)

# Test the network
TEST_STEPS = TIME_STEPS
test_x = np.zeros((1, TIME_STEPS*2, NUM_FEATURES))
test_x[:, :TIME_STEPS, 0] = sine_wave(amplitude=1.0, wavelength=25, phase=0.0, steps=TIME_STEPS)
print("Testing the network...")
for i in range(0, TEST_STEPS, NUM_OUTPUTS):
  new_frame = model.predict(test_x[:, i:i+TIME_STEPS, :])
  test_x[0, i+TIME_STEPS:i+TIME_STEPS+NUM_OUTPUTS, 0] = new_frame[0]


# Save result to file
#import csv
#with open(RESULT_FILE, "w", newline='') as csvfile:
#  csvwriter = csv.writer(csvfile, delimiter=",")
#  for row in test_x[0]:
#    csvwriter.writerow(row)

from matplotlib import pyplot as plt
x_coordinate = [i for i in range(100)]
plt.plot(x_coordinate,np.squeeze(test_x))
plt.show()
