from scipy.signal import lfilter
import scipy.io as sio
import numpy as np

from util import lpc

# t = np.linspace(0.1, 4 * np.pi, 500)
# x = np.sin(t) + np.log(t)


# a, e, ref = lpc.lpc(x, 10)
# #a = np.fliplr(a)
# #a = np.append(a, 0)
# print(a)
# b = a[1:]
# b = np.insert(b, 0, 0)
# print(b)
# est_x = lfilter(-b, [1], x)

# e = x - est_x

# import matplotlib.pyplot as plt

# plt.plot(t, x, '-r',)
# plt.plot(t, est_x, '-b')
# plt.legend(('real','estimate'))
# plt.grid()
# plt.show()

noise = np.random.randn(50000,1)
x = lfilter([1], [1, 1/2, 1/3, 1/4], noise)
x = x[45904:50000]
x = np.reshape(x,(len(x), ))

#sio.savemat('x',{'x':x})

a,e,ref = lpc(x, 6)
b = a[1:]
b = np.insert(b, 0, 0)
print(a)

est_x = lfilter(-np.insert(a[1:], 0, 0), [1], x)
e = x - est_x

import matplotlib.pyplot as plt

range_x = np.arange(95)
plt.plot(range_x, x[4001:4097], '-r')
plt.plot(range_x, est_x[4001:4097], '-b')

plt.grid()
plt.show()

