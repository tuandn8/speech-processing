"""IAIF Glottal inverse filtering

    This class estimates vocal tract linear prediction coefficients and
    the glottal volume velocity waveform (glottal flow) from a speech signal
     frame using Iterative Adaptive Inverse Filtering (IAIF) method.

     References
        P.Alku "Glottal wave analysis with picth synchronous iterative adaptive
        inverse filtering"
"""

import scipy.signal as sig
import numpy as np
from util import lpc


class IAIF:
    """
        fs:     sample rate
        p_vt:   order of LPC analysis for vocal tract
        p_gl:   order of LPC analysis for glottal source
        d:      leaky integration coefficient
        hpfilt: high-pass filter flag (0: do not apply, 1..N: apply N times)
    """
    def __init__(self, fs):
        self.d = 0.99
        self.p_gl = 2 * round(fs/4000)
        self.p_vt = 2 * round(fs/2000) + 4
        self.hpfilt = 1
        self.fs = fs
        self.preflt = self.p_vt + 1

    def _hpfilter_fir(self, Fstop, Fpass, fs, N):
        return sig.firls(N, [0, Fstop, Fpass, fs/2] / (fs/2),  [0, 0, 1, 1], [1, 1])

    def iaif(self, x, hpfilt, hpfilter_in = []):
        """ Estimate vocal tract glottal flow and derivative
        Inputs:
            x:  speech signal frame
            hpfilt: high-pass filter flag
            hpfilter_in:

        Outputs:
            g:  glottal volume velocity waveform
            dg: glottal volume velocity derivative waveform
            a:  LPC coefficients of vocal tract
            da: LPC coefficients of source spectrum
        """
        if self.hpfilt > 0:
            fstop = 40         # Stopband frequency
            fpass = 70         # Passband frequency
            nfir  = round(300/16000 * self.fs)

            if divmod(nfir, 2) == 1:
                nfir +=1

            # It is very expensive to calculate the firls filter! However, as long as the
            # fs does not change, the firls filter does not change. Therefore, the computed
            # filter is returned and can be passed to this function latter on to avoid the
            # calculated of the (same) filter
            B = []
            if len(hpfilter_in) == 0:
                B = self._hpfilter_fir(fstop, fpass, self.fs, nfir)
            else:
                B = hpfilter_in

            hpfilter_out = B

            for i in range(hpfilt):
                x = sig.lfilter(B, 1, [x, [0] * (round(len(B) / 2) - 1), 1])
                x = x[round(len(B) / 2)::len(x)]

        if __name__ == '__main__':
            if len(x) > self.p_vt:
                win = sig.hann(len(x))
                signal = [np.linspace(-x(0), x(0), self.preflt), x]
                idx = [i for i in range(self.preflt+1, len(signal))]

                hg1 = lpc(np.multiply(x, win), 1)
                y = sig.lfilter(hg1, 1, signal)
                y = y(idx)

                # Estimate the effect of the vocal tract (Hvt1) and cancel it out through
                # inverse filtering. The effect of the lip radiation is canceled through
                # intergration. Signal g1 is the first estimate of the glottal flow
                Hvt1 = lpc(np.multiply(y, win), self.p_vt)
                g1 = sig.lfilter(Hvt1, 1, signal)
                g1 = sig.lfilter(1, [1, -self.d], y)
                y = y(idx)

                # Re-estimate the effect of the glottal flow (Hg2). Cancel the contribution
                # of the glottis and the lip radiation through inverse filtering and
                # integration, respectively
                Hg2 = lpc(np.multiply(g1, win), self.p_gl)
                y = sig.lfilter(Hg2, 1, signal)
                y = sig.lfilter(1, [1, -self.d], y)
                y = y(idx)

                # Estimate the model of the vocal tract (Hvt2) and cancel it out through
                # inverse filtering. The final estimate of the glottal flow is obtained through
                # canceling the effect of the lip radiation
                Hvt2 = lpc(np.multiply(y, win), self.p_vt)
                dg = filter(Hvt2, 1, signal)
                g = sig.lfilter(1, [1, -self.d], dg)
                g = g[self.preflt+1: len(g)]
                dg = dg(idx)

                # Set vocal tract model to 'a' and glottal source spectral model to 'ag'
                a = Hvt2
                ag = Hg2
            else:
                g = []
                dg = []
                a = []
                ag = []
                print('IAIF - frame not analyzed')

        return g, dg, a, ag





