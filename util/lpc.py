from scipy.fftpack import fft, ifft
import numpy as np

def lpc(x, N):
    # calcualte autocorrelation vector or matrix
    X = fft(x, 2** nextpow2(2*len(x) - 1))
    R = np.real(ifft(abs(X)**2))
    R = R/(N-1)

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
            tmp[j] = a[j]

        for j in range(1, i):
            a[j] += ref[i-1] * np.conj(tmp[i-j])

        e *= 1 - ref[i-1] * np.conj(ref[i-1])

    return a, e, ref


def nextpow2(x):
    res = np.ceil(np.log2(x))
    return res.astype('int')
