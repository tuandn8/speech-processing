import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import hilbert, chirp


duration = 1.0
fs = 400.0
samples = int(fs * duration)
t = np.arange(samples)/fs

print t[0]
print t[-1]
signal = chirp(t, 20.0, t[-1], 100)
signal *= (1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))

analytic_signal = hilbert(signal)

amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_freq = (np.diff(instantaneous_phase) / (2.*np.pi) * fs)

fig = plt.figure()
ax0 = fig.add_subplot(311)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel('time in seconds')
ax0.legend()

ax1 = fig.add_subplot(312)
ax1.plot(t[1:], instantaneous_freq)
ax1.set_xlabel('time in seconds')
ax1.set_ylim(0.0, 120.0)

ax2 = fig.add_subplot(3,1,3)
ax2.plot(t, np.real(analytic_signal), label='analytic signal')
ax2.set_xlabel('time in seconds')
ax2.legend()

plt.show()
