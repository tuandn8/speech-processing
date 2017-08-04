from scipy.fftpack import fft, ifft
import numpy as np 



def lpc(x, N):
    # calcualte autocorrelation vector or matrix
    X = fft(x, 2** nextpow2(2*len(x) - 1))
    R = np.real(ifft(abs(X)**2))
    R = R/(N-1)
    print R.shape
    
    a, e,  ref = levinson(R, N)

    return a

def levinson(R, n):
    a = np.zeros((n,1), dtype=float)
    ref = np.zeros((n,1), dtype=float)

    E = R[0]
    k = -R[1]/E
    a[0] = k
    ref[0] = k
    E = (1- k**2) * E

    for i in range(1, n):
        k = R[i+1]
        for j in range(0, i):
            k += a[j]*R[i-j]
        k = -k/E

        a[i] = k
        E = (1 - k**2) * E

        for j in range (0, i):
            a[j] = a[j] + k * a[i-j-1]
    
    return a, E, ref



def nextpow2(x):
    res = np.ceil(np.log2(x))
    return res.astype('int')


from scipy.signal import lfilter
import scipy.io as sio

t = np.linspace(0, 4 * np.pi, 5000)
x = np.sin(t)
x = x[0:5000]

sio.savemat('np_vector.mat', {'vect':x})
print x.shape
a = lpc(x, 3)
print a.shape
a = np.fliplr(a)
a = np.append(a, 0)
print a.shape


est_x = lfilter(-a, [1], x)
e = x - est_x

import matplotlib.pyplot as plt

plt.plot(t, x, '-r')
plt.plot(t, est_x, '-b')

plt.grid()
plt.show()

# noise = np.random.randn(50000, 1)
# x = lfilter([1], [1, 1/2, 1/3, 1/4], noise)
# x = x[45904:50000]
# x.reshape( 1, len(x))
# print x.shape
# #sio.savemat('np_vector.mat', {'vect':x})

# a = lpc(x, 3)
# #a = np.fliplr(a)
# a = np.append(a, 0)
# print a

# est_x = lfilter(a, [1], x)
# e = x - est_x

# import matplotlib.pyplot as plt

# range_x = np.arange(95)
# plt.plot(range_x, x[4001:4097], '-r')
# plt.plot(range_x, est_x[4001:4097], '-b')

# plt.grid()
# plt.show()

