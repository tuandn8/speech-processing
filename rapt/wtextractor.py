import numpy as np
import scipy.signal as sig
import pywt

import matplotlib.pyplot as plt


from util import lpc
from rapt import rapt

class WaveletExtractor:

    def __init__(self):
        self.__pitch_extractor = rapt.Rapt()


    def _get_pitch(self, original_audio, fs):
        nccf, freq = self.__pitch_extractor.pitch_tracking(original_audio, fs)
        return freq


    def _get_residual_signal(self, original_audio, fs, pitch, lpc_order=12,
                             window_length=0.03):
        """
        Get residual signal by LP inverse filtering.
        :param original_audio: original audio signal
        :param fs: sampling rate
        :param pitch: pitch estimated from self._get_pitch
        :param lpc_order: LPC order
        :param window_length: window length
        :return: residual signal only for voiced region
        """

        # First find voiced speech portion, only extract portion has length
        # larger than window_length
        frame_count = len(pitch)
        self._samples_per_lpc_window = int(round(window_length * fs))
        self._samples_per_frame = int(round(self.__pitch_extractor.get_frame_step_size() *fs))
        self._minimum_pitchs_need = int(round(float(window_length * fs) / float(self._samples_per_frame)))
        voiced_signal = []
        residual_voiced_signal = []

        n_consecutive_frame = 0
        for i in range(frame_count):
            if pitch[i] > 0.0:
                n_consecutive_frame += 1
            else:
                if n_consecutive_frame >= self._minimum_pitchs_need:
                    voiced_start = (i+1-n_consecutive_frame) * self._samples_per_frame
                    voiced_end   = (i + 1) * self._samples_per_frame
                    voiced_signal.append([original_audio[voiced_start:voiced_end]])

                n_consecutive_frame = 0

        # LP analysis on voiced portion
        n_voice_portion = len(voiced_signal)
        for i in range(n_voice_portion):
            voiced_portion = voiced_signal[i][0]
            n_lpc_frame = len(voiced_portion) // self._samples_per_lpc_window

            for j in range(n_lpc_frame):
                lpc_frame_start = j * self._samples_per_lpc_window
                lpc_frame_end = (j+1) * self._samples_per_lpc_window
                lpc_frame = voiced_portion[lpc_frame_start:lpc_frame_end]
                a, e, ref = lpc.lpc(lpc_frame, lpc_order)
                a = np.append(0, -a)

                lpc_frame_est = sig.lfilter(-a, 1.0, lpc_frame)
                residual = lpc_frame - lpc_frame_est
                residual_voiced_signal.append(lpc_frame - lpc_frame_est)

                # if you need can use lpc for remain of voiced portion
                # remain_voiced_portion = voiced_portion[n_lpc_frame * self._samples_per_window:]

        return residual_voiced_signal


    def _wavelet_residual_signal(self, residual_signal, pitch):
        pass


    def _generate_feature(self, wavelet):
        pass


    def get_wavelet_features(self, original_audio, fs):
        pitch = self._get_pitch(original_audio, fs)
        residual_signal = self._get_residual_signal(original_audio, fs, pitch)

        plt.plot(residual_signal)