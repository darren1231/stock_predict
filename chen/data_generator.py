'''
Stock price forecast competition
Training data generator

Authors: Chen-Yi, Darren
Date: 2018.05.2
'''

def sine_wave(amplitude, wavelength, phase, steps):
  import math
  return [amplitude * math.sin(2*math.pi*i/wavelength + phase)\
          for i in range(steps)]
