from scipy.fftpack import fft, ifft
import numpy as np 

def lpc(x, N):
    # calcualte autocorrelation vector or matrix
    X = fft(x, 2** nextpow2(2*len(x) - 1))
    R = np.real(ifft(abs(X)**2))
    R = R/(N-1)
    print(R.shape)
    
    a, e, ref = levinson(R, N)
    return a, e, ref

def levinson(R, order):
    # coefficients array
    a = np.empty(order+1, dtype=float)
    # reflection array
    ref = np.empty(order + 1, dtype=float)
    # temporal array
    tmp = np.empty(order + 1, dtype=float)
    
    a[0] = 1.0
    e = R[0]

    for i in range(1, order + 1):
        k = R[i]
        for j in range(1,i):
            k += a[j] * R[i-j]
        ref[i-1] = -k/e
        a[i] = ref[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += ref[i-1] * np.conj(t[i-j])

        e *= 1 - ref[i-1] * np.conj(ref[i-1])

    return a, e, ref 
        

def nextpow2(x):
    res = np.ceil(np.log2(x))
    return res.astype('int')


from scipy.signal import lfilter
import scipy.io as sio

t = np.linspace(0, 4 * np.pi, 500)
x = np.sin(t)  - np.cos(t)


a, e, ref = lpc(x, 10)
#a = np.fliplr(a)
#a = np.append(a, 0)
print(a)
b = a[1:]
b = np.insert(b, 0, 0)
print(b)
est_x = lfilter(-b, [1], x)

e = x - est_x

import matplotlib.pyplot as plt

plt.plot(t, x, '-r',)
plt.plot(t, est_x, '-b')
plt.legend(('real','estimate'))
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

