import numpy as np 
import scipy.signal as sig 
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Sample rates
fs = 400
# Number of sample points
N = 400
# Filter order
filter_order = 73
t = np.linspace(0, N/fs, N)
x1 = np.sin(2 * np.pi * 10 * t)
x2 = np.sin(2 * np.pi * 2 * t)
x = x1 + 0.1 * x2

X = fft(x)
f = np.linspace(0.0, 1.0 * fs/(2.0), N//2)


nyquist = fs/2

# High-pass filter to 7 Hz
desired = (0, 0, 1, 1)
bands = (0, 3, 7, nyquist)
# Remember that filter_order must large enough 
coefs = sig.firls(filter_order, bands, desired, nyq=nyquist)

y = sig.filtfilt(coefs, [1], x)
Y = fft(y)

print (np.abs(y- x1).max())

# plot orginal signal
plt.figure(1)
plt.subplot(221)
plt.plot(t, x)

# plot original signal's spectrum
plt.subplot(222)
plt.plot(f, 2.0/N *np.abs(X[0:N//2]))

# plot filtred signal spectrum
plt.subplot(223)
plt.plot(f, 2.0/N * np.abs(Y[0:N//2]))

plt.show()

