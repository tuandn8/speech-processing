import numpy as np
import scipy.signal as sig

from scipy.fftpack import fft, ifft 

class Rapt:
    """rapt estimates the location of glottal closure instants (GCI),
    also known as epochs from digitized acoustic speech signals. It
    simultaneously estimates the local frequency (F0) and voicing 
    state of the speech on per-epoch basis.

    The processing state are:
    * Optionally high-pass filter the signal at 80Hz to remove rumble, etc
    * Compute the LPC residual, optaining an approximation of the differential
      glottal flow.
    * Normalize the amplitude of the residual by a local RMS measure.
    * Pick the prominent peaks in the glottal flow, and grade them by
      peakiness, skew and relative amplitude.
    * Compute correlates of voicing to serve as pseudo-probabilities
      of voicing, voicing onset and voicing offset.
    * For every peak selected from the residual, compute a normalized
      cross-correlation function (NCCF) of the LPC residual with a
      relatively short reference window centered on the peak.
    * For each peak in the residual, hypothesize all following peaks
      within a specified F0 seaqrch range, that might be the end of a
      period starting on that peak.
    * Grade each of these hypothesized periods on local measures of
      "voicedness" using the NCCF and the pseudo-probability of voicing
      feature.
    * Generate an unvoiced hypothesis for each period and grade it
      for "voicelessness".
    * Do a dynamic programming iteration to grade the goodness of
      continuity between all hypotheses that start on a peak and
      those that end on the same peak.  For voiced-voiced
      connections add a cost for F0 transitions.  For
      unvoiced-voiced and voiced-unvoiced transitions add a cost
      that is modulated by the voicing onset or offset inverse
      pseudo-probability.  Unvoiced-unvoiced transitions incur no cost.
    * Backtrack through the lowest-cost path developed during the
      dynamic-programming stage to determine the best peak collection
      in the residual.  At each voiced peak, find the peak in the NCCF
      (computed above) that corresponds to the duration closest to the
      inter-peak interval, and use that as the inverse F0 for the
      peak.

    References:
        Talkin, D. et al(1995). A Robust Algorithm for Pitch Tracking(RAPT). 
        Speech Coding and Synthesis, The Eds. Ams-Terdam, Netherlands:Elsevier.
    """

    def __init__(self):
        # Period for the returned F0 signal
        self.ExternalFrameInterval = 0.005
        # Internal feature-computation params for internal feature computation
        self.InternalFrameInterval = 0.002
        # Max and min for F0 search
        self.MinF0Search = 40.0
        self.MaxF0Search = 500.0
        #Pulse spacing to use in unvoiced regions of the returned epoch signal
        self.UnvoicedPulseInterval = 0.01
        self.DoHighpass = True
        self.DoHilbertTransform = False


    def __init__(self, fs, minF0, maxF0, highpass, hilbert):
        self._fs = fs
        self.MinF0Search = minF0
        self.MaxF0Search = maxF0
        self.DoHighpass = highpass
        self.DoHilbertTransform = hilbert
        
        # for high-pass filter
        self._corner_freq = 80.0
        self._filter_duration = 0.05

        # for LPC inverse filter
        self._frame_duration = 0.02 # second
        self._lpc_frame_interval = 0.01
        self._pre_emphasis = 0.98
        self._noise_floor = 70.0 # SNR in dB simulated during LPC analysis

    def _hilbertTrasform(self, s, N):
        """Perform Hilbert transform to phase distortion"""
        return sig.hilbert(s, N)


    def _hpFilter(self, s, fs):
        """Perform high-pass filter to remove effect of DC 
        and low frequency components"""
        nyquist_rate = fs/2.0
        filter_order = 5
        b, a = sig.butter(filter_order, self._corner_freq/nyquist_rate, btype='high')
        return sig.filtfilt(b, a, s)

    def _calculate_features(self):
        if self._fs <=0:
            print('Sample rate is not valid')
            return False

    def _getLPCResidual(self, s, fs):
        """Compute the LPC residual of the speech signal input s.
        Sample rate fs is the rate of the input and the residual output.
        The order of the LPC analysis is automatically set to be appropriate for the
        sample rate, and the output is integrated so it approximates the derivative 
        of the glottal flow.
        """

    def _getBandpassRmsSignal():
        


