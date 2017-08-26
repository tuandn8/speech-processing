
from scipy.io import wavfile
import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt
from glottalsource import Rapt

audio_file = "VIVOSSPK02_R016.wav"

[fs, s] = wavfile.read(audio_file)

if len(s.shape) > 1:
    s = s[:,0]/2.0  + s[:,1]/2.0
    s = s.astype(int)

algo = Rapt()
nccf, freq  = algo.pitch_tracking(s, fs)

#freq_mau = rapt_mau.rapt(audio_file)
print(len(s) / len(freq))
#plt.plot(freq, 'r-')
plt.plot(s)
# rparams = raptparams.Raptparams()
plt.plot(np.linspace(0, len(s), num=len(freq)), np.asarray(freq) * 100)
#plt.plot(freq_mau, 'bo')
plt.show()

print (freq)



