from scipy.signal import lfilter
from scipy.fftpack import fft, ifft
import scipy.io as sio
import numpy as np

from util import lpc

t = np.linspace(0, 4 * np.pi, 5000)
x = np.sin(t) + 0.1 * np.random.randn(5000,)

sio.savemat('x', {'x':x})


print(x.shape)


a, e, ref = lpc.lpc(x, 13)
print(a.reshape(len(a), ))
# a = np.append(0, a)
# #print(a)
#
# est_x = lfilter(-a, [1.0], x)
# e = x - est_x
#
# import matplotlib.pyplot as plt
#
# plt.plot(t, x, '-r')
# plt.plot(t, est_x, '-b')
# plt.legend(('real', 'estimate'))
#
# plt.grid()
# plt.show()

# noise = np.random.randn(50000, 1)
# x = lfilter([1], [1, 1/2, 1/3, 1/4], noise)
# x = x[45904:50000]
# x = np.reshape(x, (len(x), ))
#
# sio.savemat('x', {'x':x})
#
# a,e ,ref = lpc.lpc(x, 3)
#
# print(a)
# a = np.append(a, 0)
#
# est_x = lfilter(-a, [1], x)
# e = x - est_x
#
# import matplotlib.pyplot as plt
#
# range_x = np.arange(95)
#
# plt.plot(range_x, x[4001:4097], '-r')
# plt.plot(range_x, est_x[4001:4097], '-b')
#
# plt.grid()
# plt.show()

