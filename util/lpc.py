from scipy.fftpack import fft, ifft
import numpy as np 

def lpc(x, N=None):
    # calculate auto-correlation vector or matrix
    m = len(x)
    if N is None:
        N = m -1
    elif N < 0:
        raise ValueError('Lpc order must be larger than 0')
    elif N > m:
        raise ValueError('Lpc order must less than len of signal input')

    X = fft(x, 2** nextpow2(2*len(x) - 1))
    R = np.real(ifft(abs(X)**2))
    R = R/m

    print(R)

    a, e, ref = levinson(R, N)
    a = np.fliplr(a)
    return a, e, ref

def levinson(R, n):
    a = np.zeros((n,1), dtype=float)
    ref = np.zeros((n,1), dtype=float)

    E = R[0]
    k = -R[1]/E
    a[0] = k
    ref[0] = k
    E = (1 - k**2) * E

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


