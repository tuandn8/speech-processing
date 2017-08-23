import numpy as np
import scipy.signal as sig
import pywt

import matplotlib.pyplot as plt

from util import lpc
import rapt

class WaveletExtractor:
    """
    Wavelet octave coefficients of residual (WOCOR) extractor
    The process of extracting the proposed WOCOR features is formulated in
    the following steps:
    1.  Voicing decision and pitch extraction: voicing decision and pitch
        extraction are done by the robust algorithm for pitch tracking
        [RAPT Talkin, D. et al(1995)]. Only voiced speech is kept for subsequent
        processing. In the source-filter model, the excitation signal for
        unvoiced speech is approximated as a random noise. We believe that
        such a noise-like signal carries little speaker-specific information
        in the time-frequency domain.
    2.  LP inverse filtering: for each voiced speech portion, a sequence of LP
        residual signals of 30ms long is obtained by inverse filtering the speech
        signal. Where the LP coefficients are computed on Hamming windowed speech
        frames using auto-correlation method. The residual signal of neighboring
        frames are concatenated to get the residual signal, and their amplitude
        is normalized within [-1,1] to reduce intra-speaker variation.
    3.  Pitch-synchronous windowing: with the pitch periods estimated on step
        1, pitch pulses in the residual signal are located. For each pitch pulse,
        pitch-synchronous wavelet analysis is applied with a Hamming window of two
        pitch periods long. Let t_i-1, t_i, t_i+1 denote the locations of three
        successive pitch pulses. The analysis window for the pitch pulse span from
        t_i-1 to t_i+1.
    4.  Wavelet transform of the residual signal: applied the wavelet transform
        to windowed residual signal.
    5.  Generating the features parameters: divide each octave group to
        sub-group and apply 2-norm to get the complete feature vectors WOCOR.

    References:
        Zheng, N. et al. Integration of complementary acoustic features for
        speaker recognition. IEEE Signal Processing Letters 14, 181–184 (2007).
    """

    def __init__(self):
        self.__pitch_extractor = rapt.Rapt()


    def _get_pitch(self, original_audio, fs):
        nccf, freq = self.__pitch_extractor.pitch_tracking(original_audio, fs)
        return freq


    def _get_residual_signal(self, original_audio, fs, pitch, lpc_order=12,
                             window_length=0.03):
        """
        Get residual signal by LP inverse filtering.
        Based on estimated pitch, only process on voiced signal. First, extract
        the voiced segment to voiced_signal. Second, split each voiced
        segment into lpc frames has length equal window_length. Then use LP
        to calculate residual signal in each lpc frame.
        :param original_audio: original audio signal
        :param fs: sampling rate
        :param pitch: pitch estimated from self._get_pitch
        :param lpc_order: LPC order
        :param window_length: window length
        :return: list tuple residual signal only for each voiced region. Each
                tuple has 3 element: start sample, end sample, residual on
                each frame. The length from start to end frame is equal for all
                residual.
        """
        # First find voiced speech portion, only extract portion has length
        # larger than window_length
        frame_count = len(pitch)
        self._samples_per_lpc_window = int(round(window_length * fs))  # 480
        self._samples_per_pitch_frame = int(round(self.__pitch_extractor.get_frame_step_size() *fs)) # 160
        self._minimum_pitchs_need = int(round(float(window_length * fs) / float(self._samples_per_pitch_frame))) # 3
        voiced_signal = []
        residual_voiced_signal = []

        n_consecutive_frame = 0
        for i in range(frame_count):
            if pitch[i] > 0.0:
                n_consecutive_frame += 1
            else:
                if n_consecutive_frame >= self._minimum_pitchs_need:
                    voiced_start = (i+1-n_consecutive_frame) * self._samples_per_pitch_frame
                    voiced_end   = (i + 1) * self._samples_per_pitch_frame
                    voiced_signal.append((voiced_start, voiced_end, original_audio[voiced_start:voiced_end]))

                n_consecutive_frame = 0

        # LP analysis on voiced portion
        n_voice_portion = len(voiced_signal)
        for i in range(n_voice_portion):
            voiced_portion = voiced_signal[i]
            n_lpc_frame = len(voiced_portion[2]) // self._samples_per_lpc_window
            voiced_portion_offset = voiced_portion[0]

            for j in range(n_lpc_frame):
                lpc_frame_start = j * self._samples_per_lpc_window
                lpc_frame_end = (j+1) * self._samples_per_lpc_window
                lpc_frame = voiced_portion[2][lpc_frame_start:lpc_frame_end]
                han_win = sig.hanning(len(lpc_frame))
                lpc_frame = han_win*lpc_frame
                a, e, ref = lpc(lpc_frame, lpc_order)
                lpc_frame_est = sig.lfilter(-np.insert(a[1:], 0, 0), 1.0, lpc_frame)
                residual = lpc_frame - lpc_frame_est
                residual = residual / np.max(abs(residual))
                residual_voiced_signal.append(( lpc_frame_start + voiced_portion_offset, 
                                                lpc_frame_end + voiced_portion_offset, 
                                                residual))

        return residual_voiced_signal


    def _get_pitch_pulses(self, fs, residual_signal, pitch):
        """
        Use the pitch estimated to find the pitch pulses in each LPC frame. Find
        the largest pitch in LPC frame, then find the periodic pulses before/after
        by move some samples.
        :param fs: sampling rate
        :param residual_signal: list tuple residual signal which is estimated by
                using self._get_residual_signal()
        :param pitch: estimated pitch (list of freq)
        :return: residual pitch pulse index samples
        """
        #MIN_PERIOD = int(0.002 * fs) # 20ms  <=> 500Hz  32 samples
        #MAX_PERIOD = int(0.016 * fs) # 160ms <=> 62.6Hz 256 samples
        MAG_THRESHOLD = 0.5          # magnitude relative threshold to highest = 50%

        residual_max_idx = []
        
        n_residual = len(residual_signal)
        for i in range(n_residual):
            peak_idx = []
            residual = residual_signal[i][2]
            pitch_freq = pitch[int(round(residual_signal[i][0] / self._samples_per_pitch_frame))]
            period_samples = int(fs / pitch_freq)
            if period_samples < 30:
                MIN_PERIOD = period_samples // 2
            else:
                MIN_PERIOD = period_samples - 30
            MAX_PERIOD = period_samples + 30
            residual_len = len(residual)
            maximum_peak = max_peak = np.max(residual)
            maximum_idx  = max_idx = np.argmax(residual)
            peak_idx.append(max_idx)

            # find backward
            j = maximum_idx - MAX_PERIOD
            if j < 0:
                j = 0      
            while j >= 0 and max_idx - MIN_PERIOD >= 0:
                prev_peak_frame = residual[j: max_idx - MIN_PERIOD]
                max_peak = np.max(prev_peak_frame)
                if float(max_peak)/float(maximum_peak) < MAG_THRESHOLD:
                    break
                max_idx = j + np.argmax(prev_peak_frame)
                peak_idx.append(max_idx)
                j = max_idx - MAX_PERIOD
                if j < 0:
                    j = 0
                
            # find forward
            j = maximum_idx + MAX_PERIOD
            if j > residual_len:
                j = residual_len

            max_idx = maximum_idx
            while j <= residual_len and max_idx + MIN_PERIOD < residual_len:
                next_peak_frame = residual[max_idx + MIN_PERIOD : j]
                max_peak = np.max(next_peak_frame)
                if float(max_peak)/float(maximum_peak) < MAG_THRESHOLD:
                    break
                max_idx += MIN_PERIOD + np.argmax(next_peak_frame)
                peak_idx.append(max_idx)
                j = max_idx + MAX_PERIOD
                if j > residual_len:
                    j = residual_len

            # soft peak idx to samples index
            peak_idx.sort()
            residual_max_idx.append((peak_idx))
        return residual_max_idx     


    def _feature_extract(self, pitch_pulse_index, residual_signal):
        """
        Extract features from residual signal
        :param pitch_pulse_index: pitch pulse index in each residual frame
        :param residual_signal: list of residual frame
        :return:
        """
        wocors = []
        daubechies = 'gaus1'
        M = 4
        K = 4
        for pulse_idx, residual in zip(pitch_pulse_index, residual_signal):
            if len(pulse_idx) < 3:
                continue
            
            # apply Hamming window to residual signal
            for i in range(len(pulse_idx) - 2):
                w_residual = residual[2][pulse_idx[i]: pulse_idx[i+2]]
                N = len(w_residual)
                w_ham = sig.hamming(N)
                e_h = w_ham * w_residual

                # calculate WOCOR - wavelet coefficients
                wocor = []
                coefs, freqs = pywt.cwt(e_h, [1,2,3,4], daubechies)
                for k in range(K):
                    for m in range(1, M+1):
                        wkm = coefs[k][int((m-1)*N/M):int(m*N/M)]
                        wocor.append(np.linalg.norm(wkm, ord=2))
                wocors.append(wocor)    
    
        return wocors

    
    def get_wavelet_features(self, original_audio, fs):
        """
        Calculate WOCOR features
        :param original_audio: original audio signal
        :param fs: sampling rate
        :return: list of list of WOCOR features.
        """
        pitch = self._get_pitch(original_audio, fs)
        residual_signal = self._get_residual_signal(original_audio, fs, pitch)
        pitch_pulse_idx = self._get_pitch_pulses(fs, residual_signal, pitch)
        wavelet_coeffs = self._feature_extract(pitch_pulse_idx, residual_signal)
        return wavelet_coeffs


        