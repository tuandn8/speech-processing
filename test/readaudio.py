import numpy as np 
from scipy.io.wavfile import read
import sys
import matplotlib.pyplot as plt

input_data = read('/home/vnc/Downloads/vivos/train/waves/VIVOSSPK02/VIVOSSPK02_R003.wav')
signal = input_data[1]

plt.plot(signal)
plt.ylabel('Amplitude')
plt.xlabel('time')

plt.show()
