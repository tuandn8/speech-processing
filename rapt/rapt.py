import math
import numpy as np 
from scipy import signal

import raptparams
import nccfparams


class Rapt:
    """rapt estimates the location of glottal closure instants (GCI), also known
        as epochs from digitized acoustic speech signals. It simultaneously estimates
        the local frequency (F0) and voicing state of the speech on per-epoch basis.

        The processing state are:
        * Optionally high-pass filter the signal at 80Hz to remove rumble, etc
        * Compute the LPC residual, optaining an approximation of the differential
          glottal flow.
        * Normalize the amplitude of the residual by a local RMS measure.
        * Pick the prominent peaks in the glottal flow, and grade them by peakiness,
          skew and relative amplitude.
        * Compute correlates of voicing to serve as pseudo-probabilities of voicing,
          voicing onset and voicing offset.
        * For every peak selected from the residual, compute a normalized
          cross-correlation function (NCCF) of the LPC residual with a relatively
          short reference window centered on the peak.
        * For each peak in the residual, hypothesize all following peaks within a
          specified F0 seaqrch range, that might be the end of a period starting on
          that peak.
        * Grade each of these hypothesized periods on local measures of "voicedness"
          using the NCCF and the pseudo-probability of voicing
          feature.
        * Generate an unvoiced hypothesis for each period and grade it for
          "voicelessness".
        * Do a dynamic programming iteration to grade the goodness of continuity
          between all hypotheses that start on a peak and those that end on the same
          peak. For voiced-voice connections add a cost for F0 transitions. For
          unvoiced-voiced and voiced-unvoiced transitions add a cost that is
          modulated by the voicing onset or offset inverse pseudo-probability.
          Unvoiced-unvoiced transitions incur no cost.
        * Backtrack through the lowest-cost path developed during the
          dynamic-programming stage to determine the best peak collection in the
          residual.  At each voiced peak, find the peak in the NCCF (computed above)
          that corresponds to the duration closest to the inter-peak interval, and
          use that as the inverse F0 for the peak.

        References:
            Talkin, D. et al(1995). A Robust Algorithm for Pitch Tracking(RAPT).
            Speech Coding and Synthesis, The Eds. Ams-Terdam, Netherlands:Elsevier.
    """
    def __init__(self):
        self.params = raptparams.Raptparams()


    def pitch_tracking(self, original_audio, fs):
        self._original_audio = original_audio
        self._fs = fs
        if self.params.is_two_pass_nccf:
            downsample_rate, downsampled_audio = self._get_downsampled_audio(original_audio, fs,
                                                                self.params.maximum_allowed_freq,
                                                                self.params.is_run_filter)
            # calculate parameters for RAPT with input audio
            self._calculate_params(original_audio, fs, downsampled_audio, downsample_rate)
            
            # get F0 candidates using NCCF
            nccf_results = self._run_nccf(original_audio, fs, downsampled_audio, downsample_rate)
        else:
            self._calculate_params(original_audio, fs)
            nccf_results = self._run_nccf(original_audio, fs)

        # dynamic programming - determine voicing state at each period candidate
        freq_estimate = self._get_freq_estimate(nccf_results[0], fs)

        # filter out high freq points
        for i, item in enumerate(freq_estimate):
            if item > 500.0:
                freq_estimate[i] = 0.0
            
        return freq_estimate


    def rapt_with_nccf(self, original_audio, fs):
        """
        The main method that perform pitch tracking RAPT algorithm
        :param original_audio: audio signal
        :param fs:  sampling rate
        :return: the pitch, frame time for each frame in 20ms
        """
        self._original_audio = original_audio
        self._fs = fs
        if self.params.is_two_pass_nccf:
            (downsample_rate, downsampled_audio) = self._get_downsampled_audio(original_audio, fs,
                                                            self.params.maximum_allowed_freq,
                                                            self.params.is_run_filter)

            # calculate parameters for RAPT with input audio
            self._calculate_params(original_audio, fs, downsampled_audio, downsample_rate)

            # get F0 candidates using nccf
            nccf_results = self._run_nccf(original_audio, fs, downsampled_audio, downsample_rate)
        else:
            self._calculate_params(original_audio, fs)
            nccf_results = self._run_nccf(original_audio, fs)
        
        # dynamic programming - determine voicing state at each period candidate
        freq_estimate = self._get_freq_estimate(nccf_results[0], fs)

        # filter out high freq points
        for i, item in enumerate(freq_estimate):
            if item > 500.0:
                freq_estimate[i] = 0.0

        return (nccf_results, freq_estimate)
       
    
    def _get_downsampled_audio(self, original_audio, fs, maximum_allowed_freq, is_filter):
        """
        Calculate downsampling rate, downsample audio, return as tuple include
        downsample_rate and downsampled audio
        :param original_audio: original signal
        :param fs: sampling rate
        :param maximum_allowed_freq: maximum F0 freq
        :param is_filter: perform lowpass and high pass filter
        :return: downsample rate and downsampled audio
        """
        downsample_rate = self._calcualte_downsampling_rate(fs, maximum_allowed_freq)
        
        # low pass filter
        # TODO(tuandn4) need high pass filter to remove silent intervals or low-amplitude unvoiced intervals
        if is_filter:
            # low-pass filter
            freq_cutoff = 0.05 / (0.5 * float(downsample_rate))
            taps = 100   # filter length = filter order + 1 = 100

            filter_coefs = signal.firwin(taps, cutoff=freq_cutoff, width=0.005, window='hanning')
            filtered_audio = signal.lfilter(filter_coefs, 1, original_audio)

            # high-pass filter
            # freq_cutoff = 80
            # nyquist_rate = fs/2
            # filter_coefs = signal.firwin(taps-1, cutoff=freq_cutoff, pass_zero=False, width=0.005, window='hann', nyq=nyquist_rate)
            # filtered_audio = signal.lfilter(filter_coefs, 1.0, filtered_audio)

            # downsample signal
            downsampled_audio = self._downsample_audio(filtered_audio, fs, downsample_rate)
        else:
            downsampled_audio = self._downsample_audio(original_audio, fs, downsample_rate)
        
        return (downsample_rate, downsampled_audio)


    def _calcualte_downsampling_rate(self, fs, maximum_f0):
        """Determines downsampling rate to appy to the audio input passed for 
        RAPT processing"""
        return int(fs/ round(fs/(4 * maximum_f0)))


    def _downsample_audio(self, original_audio, fs, downsample_rate):
        """Given the original audio sample/rate and desired downsampling rate
        returns a downsampled version of the audio input"""
        sample_rate_ratio = float(downsample_rate) / float(fs)
        number_of_samples = int(len(original_audio) * sample_rate_ratio)
        downsampled_audio = signal.resample(original_audio, int(number_of_samples))

        return downsampled_audio


    def _calculate_params(self, original_audio, fs, downsampled_audio = None, downsample_rate = None):
        """
        Calculate some parameter for NCCF
        :param original_audio:
        :param fs:
        :param downsampled_audio:
        :param downsample_rate:
        :return:
        """
        self.params.original_audio = original_audio

        if downsampled_audio is not None:
            self.params.sample_rate_ratio = float(fs) / float(downsample_rate)
        
        self.params.samples_per_frame = int(round(self.params.frame_step_size * fs))
        self.params.hanning_window_length = int(round(0.03 * fs))
        self.params.hanning_window_vals = np.hanning(self.params.hanning_window_length)

        # offset adjusts window centers to be 20ms apart regardless of frame step
        # size - so the goal here is to find diff between frame size and 20ms apart
        self.params.rms_offset = int(round(((float(fs)/1000.0) * 20.0) - 
                                    self.params.samples_per_frame))
        

    def _run_nccf(self, original_audio, fs, downsampled_audio = None, downsample_rate = None):
        """
        Run two or one pass NCCF
        :param original_audio: original signal
        :param fs: signal sampling rate
        :param downsampled_audio: downsampled signal
        :param downsample_rate: downsample sampling rate of downsampled signal
        :return: tuble includes nccf_results and/or first pass result
        """
        if self.params.is_two_pass_nccf:
            first_pass = self._first_pass_nccf(downsampled_audio, downsample_rate)
            nccf_results = self._second_pass_nccf(original_audio, fs, first_pass)
            return (nccf_results, first_pass)
        else:
            nccf_results = self._one_pass_nccf(original_audio,fs, False)
            return (nccf_results, None)


    def _one_pass_nccf(self, audio, fs):
        """
        Run NCCF on full audio sample and returns top correlations per frames
        :param audio: signal
        :param fs: sampling rate
        :return:
        """
        self._get_nccf_params(audio, fs, True)

        # difference between K-1 and starting value of k
        lag_range = (self.nccfparams.longest_lag_per_frame - 1) - self.nccfparams.shortest_lag_per_frame
        candidates = [None] * self.nccfparams.max_frame_count

        for i in range(0, self.nccfparams.max_frame_count):
            all_lag_results = self._get_correlations_for_all_lags(audio, i, lag_range)

            candidates[i] = self._get_marked_results(all_lag_results, False)

        return candidates


    def _first_pass_nccf(self, audio, fs):
        """
        Runs normalized cross correlation function (NCCF) on downsampled audio,
        outputting a set of potential F0 candidates that could be used to determine
        the pitch at each given frame of the audio sample
        :param audio:
        :param fs:
        :return: candidates: all NCCF larger than 0.3 * local maximum NCCF in each frame
        """
        self._get_nccf_params(audio, fs, True)

        # difference between K-1 and starting value of k
        lag_range = (self.nccfparams.longest_lag_per_frame - 1) - self.nccfparams.shortest_lag_per_frame
        candidates = [None] * self.nccfparams.max_frame_count

        for i in range(0, self.nccfparams.max_frame_count):
            candidates[i] = self._get_firstpass_frame_results(audio, fs, i, lag_range)

        return candidates


    def _second_pass_nccf(self, original_audio, fs, first_pass):
        """
        Return NCCF on downsampled audio, outputting a set of F0 that could be used
        to determine the pitch by Dynamic programming
        :param original_audio:
        :param fs:
        :param first_pass:
        :return:
        """
        self._get_nccf_params(original_audio, fs, False)

        lag_range = (self.nccfparams.longest_lag_per_frame - 1) - self.nccfparams.shortest_lag_per_frame
        candidates = [None] * self.nccfparams.max_frame_count

        for i in range(self.nccfparams.max_frame_count):
            candidates[i] = self._get_secondpass_frame_results(original_audio, i, lag_range, first_pass)
        
        return candidates


    def _get_nccf_params(self, audio, fs, is_first_pass):
        """Creates and returns nccf params object w/ nccf-specific values"""
        self.nccfparams = nccfparams.Nccfparams()

        # value 'n' in NCCF equation
        self.nccfparams.samples_correlated_per_lag = int(round(self.params.correlation_window_size * fs))
        
        # start value of k in NCCF equation
        if is_first_pass:
            self.nccfparams.shortest_lag_per_frame = int(round(fs/self.params.maximum_allowed_freq))
        else:
            self.nccfparams.shortest_lag_per_frame = 0

        # value 'K' in NCCF equation
        self.nccfparams.longest_lag_per_frame = int(round(fs/self.params.minimum_allowed_freq))

        # value z in NCCF equation
        self.nccfparams.samples_per_frame = int(round(self.params.frame_step_size * fs))

        # value of M-1 in NCCF equation
        self.nccfparams.max_frame_count = int(round(float(len(audio)) / float(self.nccfparams.samples_per_frame)) - 1)


    def _get_firstpass_frame_results(self, audio, fs, current_frame, lag_range):
        """
        Calculate correlation (theta) for all lags, and get the highest
        correlation val (theta_max) from the calculated lags for each frame
        """
        all_lag_results = self._get_correlations_for_all_lags(audio, fs, current_frame, lag_range)
        marked_values = self._get_marked_results(all_lag_results, True)
        return marked_values

    
    def _get_secondpass_frame_results(self, audio, current_frame, lag_range, first_pass):
        """
        Calculate correlation for all lags and get the highest correctional value
        from the calculated lags and first pass
        """
        lag_results = self._get_correlations_for_input_lags(audio, current_frame, first_pass, lag_range)
        marked_values = self._get_marked_results(lag_results, False)
        return marked_values

    
    def _get_correlations_for_all_lags(self, audio, fs, current_frame, lag_range):
        """
        Value of theta_max in NCCF equation, max for current frame
        """
        candidates = [0.0] * lag_range
        max_correlation_val = 0.0

        for k in range(lag_range):
            current_lag = k + self.nccfparams.shortest_lag_per_frame

            # determine if the current lag value causes us to go past the end of 
            # the audio sample - if so - skip and set val to zero
            if ((current_lag + self.nccfparams.samples_correlated_per_lag - 1) +
                (current_frame * self.nccfparams.samples_per_frame)) >= len(audio):
                continue

            candidates[k] = self._get_correlation(audio, current_frame, current_lag)

            if candidates[k] > max_correlation_val:
                max_correlation_val = candidates[k]

        return (candidates, max_correlation_val)

        
    def _get_correlations_for_input_lags(self, audio, current_frame, first_pass, lag_range):
        candidates = [0.0] * lag_range
        max_correlation_val = 0.0
        sorted_firstpass_results = first_pass[current_frame]
        sorted_firstpass_results.sort(key=lambda tup: tup[0])

        for lag_val in sorted_firstpass_results:
            # 1st pass lag value has been interpolated for original audio sample
            lag_peak = lag_val[0]

            # for each peak check the closest 7 lags 
            if lag_peak > 10 and lag_peak < lag_range - 11:
                for k in range(lag_peak - 10, lag_peak + 11):
                    # determine if the current lag value causes us to go past the
                    # end of the audio sample - if so - skip and set val to zero
                    sample_range = (k + (self.nccfparams.samples_correlated_per_lag - 1) +
                                    (current_frame * self.nccfparams.samples_per_frame))
                    if sample_range >= len(audio):
                        continue
                
                    candidates[k] = self._get_correlation(audio, current_frame, k, False)
                    if candidates[k] > max_correlation_val:
                        max_correlation_val = candidates[k]

        return (candidates, max_correlation_val)


    def _get_marked_results(self, lag_results, is_first_pass = True):
        # values that meet certain threshold shall be marked for consideration
        min_valid_correlation = (lag_results[1] * self.params.min_acceptable_peak_val)
        max_allowed_candidates = self.params.max_hypotheses_per_frame - 1
        candidates = []
        
        if is_first_pass:
            candidates = self._extrapolate_lag_val(lag_results, min_valid_correlation)
        else:
            for k, kval in enumerate(lag_results[0]):
                if kval > min_valid_correlation:
                    current_lag = k + self.nccfparams.shortest_lag_per_frame
                    candidates.append((current_lag, kval))
        
        # now check to see if selected candidates exceed max allowed:
        if len(candidates) > max_allowed_candidates:
            candidates.sort(key=lambda tup: tup[1], reverse=True)
            returned_candidates = candidates[0:max_allowed_candidates]

            # re-sort before returning so that it is in order of low to highest k
            returned_candidates.sort(key=lambda tup: tup[0])
        else:
            returned_candidates = candidates

        return returned_candidates

    
    def _get_correlation(self, audio_sample, frame, lag, is_first_pass = True):
        samples = 0
        samples_correlated_per_lag = self.nccfparams.samples_correlated_per_lag
        frame_start = frame * self.nccfparams.samples_per_frame
        final_correlated_sample = frame_start + samples_correlated_per_lag

        frame_sum = np.sum(audio_sample[frame_start:final_correlated_sample])
        mean_for_window = ((1.0 / float(samples_correlated_per_lag)) * frame_sum)

        audio_slice = audio_sample[frame_start:final_correlated_sample]
        lag_audio_slice = audio_sample[frame_start + lag:final_correlated_sample + lag]

        samples = np.sum((audio_slice - mean_for_window) * (lag_audio_slice - mean_for_window))

        denominator_base = np.sum((audio_slice - float(mean_for_window))**2)
        denominator_lag = np.sum((lag_audio_slice - float(mean_for_window))**2)

        if is_first_pass and self.params.is_two_pass_nccf:
            denominator = math.sqrt(denominator_base * denominator_lag)
        else:
            denominator = ((denominator_base * denominator_lag) + self.params.additive_constant)
            denominator = math.sqrt(denominator)

        return float(samples) / float(denominator)


    def _extrapolate_lag_val(self, lag_results, min_valid_correlation):
        extrapolated_cands = []
        
        if len(lag_results[0]) == 0:
            return extrapolated_cands
        elif len(lag_results[0]) == 1:
            current_lag = 0 + self.nccfparams.shortest_lag_per_frame
            new_lag = int(round(current_lag * self.params.sample_rate_ratio))
            extrapolated_cands.append((new_lag, lag_results[0][0]))
            return extrapolated_cands

        least_lag = self.params.sample_rate_ratio * self.nccfparams.shortest_lag_per_frame
        most_lag = self.params.sample_rate_ratio * self.nccfparams.longest_lag_per_frame

        for k, k_val in enumerate(lag_results[0]):
            if k_val > min_valid_correlation:
                current_lag = k + self.nccfparams.shortest_lag_per_frame
                new_lag = int(round(current_lag * self.params.sample_rate_ratio))

                if k == 0:
                    # if at 1st lag value, interpolate using 0,0 input on left
                    prev_lag = k - 1 + self.nccfparams.shortest_lag_per_frame
                    new_prev = int(round(prev_lag * self.params.sample_rate_ratio))
                    next_lag = k + 1 + self.nccfparams.shortest_lag_per_frame
                    new_next = int(round(next_lag * self.params.sample_rate_ratio))
                    lags = np.array([new_prev, new_lag, new_next])
                    vals = np.array([0.0, k_val, lag_results[0][k+1]])
                    para = np.polyfit(lags, vals, 2)
                    final_lag = int(round(-para[1]/(2 * para[0])))
                    final_corr = float(para[0] * final_lag**2 + para[1] * final_lag + para[2])

                    if final_lag < least_lag or final_lag > most_lag or final_corr < -1.0 or final_corr > 1.0:
                        current_lag = k + self.nccfparams.shortest_lag_per_frame
                        new_lag = int(round(current_lag*self.params.sample_rate_ratio))
                        extrapolated_cands.append((new_lag, k_val))
                    else:
                        extrapolated_cands.append((final_lag, final_corr))
                elif k == len(lag_results[0]) - 1:
                    # if at last lag value, interpolate using 0,0 input on right
                    next_lag = k + 1 + self.nccfparams.shortest_lag_per_frame
                    new_next = int(round(next_lag * self.params.sample_rate_ratio))
                    prev_lag = (k - 1 + self.nccfparams.shortest_lag_per_frame)
                    new_prev = int(round(prev_lag * self.params.sample_rate_ratio))
                    lags = np.array([new_prev, new_lag, new_next])
                    vals = np.array([lag_results[0][k-1], k_val, 0.0])
                    para = np.polyfit(lags, vals, 2)
                    final_lag = int(round(-para[1] / (2 * para[0])))
                    final_corr = float(para[0] * final_lag**2 + para[1] * final_lag + para[2])
                    if (final_lag < least_lag or final_lag > most_lag or
                            final_corr < -1.0 or final_corr > 1.0):
                        current_lag = k + self.nccfparams.shortest_lag_per_frame
                        new_lag = int(round(current_lag * self.params.sample_rate_ratio))
                        extrapolated_cands.append((new_lag, k_val))
                    else:
                        extrapolated_cands.append((final_lag, final_corr))
                else:
                    # we are in middle of lag results - use left and right
                    next_lag = (k + 1 + self.nccfparams.shortest_lag_per_frame)
                    new_next = int(round(next_lag * self.params.sample_rate_ratio))
                    prev_lag = k - 1 + self.nccfparams.shortest_lag_per_frame
                    new_prev = int(round(prev_lag * self.params.sample_rate_ratio))
                    lags = np.array([new_prev, new_lag, new_next])
                    vals = np.array([lag_results[0][k-1], k_val, lag_results[0][k+1]])
                    para = np.polyfit(lags, vals, 2)
                    final_lag = int(round(-para[1] / (2 * para[0])))
                    final_corr = float(para[0] * final_lag**2 + para[1] * final_lag + para[2])
                    if (final_lag < least_lag or final_lag > most_lag or
                            final_corr < -1.0 or final_corr > 1.0):
                        current_lag = k + self.nccfparams.shortest_lag_per_frame
                        new_lag = int(round(current_lag * self.params.sample_rate_ratio))
                        extrapolated_cands.append((new_lag, k_val))
                    else:
                        extrapolated_cands.append((final_lag, final_corr))

        return extrapolated_cands


    def _get_peak_lag_val(self, lag_results, lag_index):
        current_lag = lag_index + self.nccfparams.shortest_lag_per_frame
        extrapolated_lag = int(round(current_lag * self.params.sample_rate_ratio))
        return (extrapolated_lag, lag_results[lag_index])


    def _get_correlations_for_all_lags(self, audio, fs, current_frame, lag_range):
        # Value of theta_max in NCCF equation, max for the current frame
        candidates = [0.0] * lag_range
        max_correlation_val = 0.0
        for k in range(0, lag_range):
            current_lag = k + self.nccfparams.shortest_lag_per_frame

            # determine if the current lag value causes us to go pass the end
            # of the audio sample - if so - skip and set val to 0
            if ((current_lag + (self.nccfparams.samples_correlated_per_lag - 1)
                    + (current_frame * self.nccfparams.samples_per_frame)) >= len(audio)):
                continue
            
            candidates[k] = self._get_correlation(audio, current_frame, current_lag)

            if candidates[k] > max_correlation_val:
                max_correlation_val = candidates[k]
        
        return (candidates, max_correlation_val)


    # Dynamic programming to estimate F0
    def _get_freq_estimate(self, nccf_results, sample_rate):
        """
        This method will obtain best candidate per frame and calc freq estimate per frame
        """
        results = []
        candidates = self._determine_state_per_frame(nccf_results, sample_rate)

        for candidate in candidates:
            if candidate > 0:
                results.append(sample_rate/candidate)
            else:
                results.append(0.0)

        return results

    
    def _determine_state_per_frame(self, nccf_results, sample_rate):
        """
        Prepare to call a recursive function that will determine the optimal
        voicing state candidate per frame
        """
        candidates = []
        # add unvoiced candidate entry per frame (tuple w/0 lag, 0 correlation)
        for result in nccf_results:
            result.append((0, 0.0))

        # now call recursive function that will calcualte cost per candidate
        final_candiates = self._select_candidates(nccf_results, sample_rate)

        # sort results - take the lag of the lowest cost candidate for its last item
        final_candiates.sort(key=lambda y: y[-1][0])

        # with the result, take the lag of the lowest cost candidate per frame
        for result in final_candiates[0]:
            candidates.append(result[1][0])

        return candidates


    def _select_candidates(self, nccf_results, sample_rate):
        """

        :param nccf_results:
        :param sample_rate:
        :return:
        """
        max_for_frame = self._select_max_correlation_for_frame(nccf_results[0])
        frame_candidates = []

        for candidate in nccf_results[0]:
            best_cost = None
            local_cost = self._calcualte_local_cost(candidate, max_for_frame, sample_rate)

            for initial_candidate in [(0.0, (1, 0.1)), (0.0, (0, 0.0))]:
                delta_cost = self._get_delta_cost(candidate, initial_candidate, 0)
                total_cost = local_cost + delta_cost
                if best_cost is None or total_cost <= best_cost:
                    best_cost = total_cost
            frame_candidates.append([(best_cost, candidate)])

        # now we have initial costs for frame 0. lets loop through later frames
        final_candiates = self._get_next_cands(1, frame_candidates, nccf_results, sample_rate)

        return final_candiates


    def _get_next_cands(self, frame_idx, prev_candidates, nccf_results, sample_rate):
        frame_max = self._select_max_correlation_for_frame(nccf_results[frame_idx])
        final_candiates = []

        for candidate in nccf_results[frame_idx]:
            best_cost = None
            returned_path = None
            local_cost = self._calcualte_local_cost(candidate, frame_max, sample_rate)

            for prev_candidate in prev_candidates:
                delta_cost = self._get_delta_cost(candidate, prev_candidate[-1], frame_idx)
                total_cost = local_cost + delta_cost

                if best_cost is None or total_cost <= best_cost:
                    best_cost = total_cost
                    returned_path = list(prev_candidate)
            returned_path.append((best_cost, candidate))
            final_candiates.append(returned_path)

        next_idx = frame_idx + 1
        if next_idx < len(nccf_results):
            return self._get_next_cands(next_idx, final_candiates, nccf_results, sample_rate)

        return final_candiates


    def _select_max_correlation_for_frame(self, nccf_results_frame):
        """
        Find maximum NCCF
        """
        max_val = 0.0
        for hypothesis in nccf_results_frame:
            if hypothesis[1] > max_val:
                max_val = hypothesis[1]

        return max_val


    def _calcualte_local_cost(self, candidate, max_corr_for_frame, sample_rate):
        """
        Calcualte local cost of hypothesis d_i,j
        d_i_j = 1 - Cij * (1 - beta * Lij) for voiced
              = VO_BIAS + max(Cij)  for unvoiced
        :param candidate: the value of the j_th local maximum at frame i
        :param max_corr_for_frame:
        :param sample_rate:
        :return:
        """
        lag_val = candidate[0]
        correlation_val = candidate[1]
        if lag_val == 0 and correlation_val == 0.0:
            # unvoiced hypothesis: add VO_BIAS to largest correlation val in frame
            cost = self.params.voicing_bias + max_corr_for_frame
        else:
            # voiced hypothesis
            lag_weight = (float(self.params.lag_weight) / float(sample_rate / float(self.params.minimum_allowed_freq)))
            cost = (1.0 - correlation_val * (1.0 - float(lag_weight) * float(lag_val)))

        return cost


    def _get_delta_cost(self, candidate, prev_candidate, frame_idx):
        # determine what type of transition
        if self._is_unvoiced(candidate) and self._is_unvoiced(prev_candidate[1]):
            return self._get_unvoiced_to_unvoiced_cost(prev_candidate)
        elif self._is_unvoiced(candidate):
            return self._get_voiced_to_unvoiced_cost(candidate, prev_candidate, frame_idx)
        elif self._is_unvoiced(prev_candidate[1]):
            return self._get_unvoiced_to_voiced_cost(candidate, prev_candidate, frame_idx)
        else:
            return self._get_voiced_to_voiced_cost(candidate, prev_candidate)


    def _is_unvoiced(self, candidate):
        return candidate[0] == 0 and candidate[1] == 0.0


    def _get_voiced_to_voiced_cost(self, candidate, prev_entry):
        prev_cost = prev_entry[0]
        prev_candidate = prev_entry[1]

        # value of epsilon in voiced-to-voiced delta formula
        freq_jump_cost = np.log(float(candidate[0])/ float(prev_candidate[0]))
        transition_cost = (self.params.freq_weight * (self.params.doubling_cost + abs(freq_jump_cost - np.log(2))))
        final_cost = prev_cost + transition_cost

        return final_cost


    def _get_unvoiced_to_unvoiced_cost(self, prev_entry):
        return prev_entry[0] + 0.0


    def _get_voiced_to_unvoiced_cost(self, candidate, prev_entry, frame_idx):
        prev_cost = prev_entry[0]
        # NOTE(tuandn4) not using spec_mode itakura distortion because hard to
        # calculate this 
        # delta = self.params.transition_cost + (self.params.spec_mod_transition_cost *
        #    self._get_spec_stationarity()) + (self.params.amp_mod_transition_cost *
        #    self._get_rms_ratio(frame_idx))
        delta = (self.params.transition_cost + self.params.amp_mod_transition_cost * self._get_rms_ratio(frame_idx))
        return prev_cost + delta


    def _get_unvoiced_to_voiced_cost(self, candidate, prev_entry, frame_idx):
        prev_cost = prev_entry[0]
        # prev_candidate = prev_entry[1]
        # NOTE(tuandn4) not using spec_mode itakura distortion because hard to
        # calculate this
        # delta = self.params.transition_cost + (self.params.spec_mod_transition_cost *
        #        self._get_spec_stationarity()) + (self.params.amp_mod_transition_cost / 
        #        self._get_rms_ratio(frame_idx))
        current_rms_ratio = self._get_rms_ratio(frame_idx)

        # TODO(tuandn4) figure out how to better handle rms ratio on final frame
        if current_rms_ratio <= 0:
            return prev_cost + self.params.transition_cost

        delta = self.params.transition_cost + (self.params.amp_mod_transition_cost / current_rms_ratio)
        return prev_cost + delta


    def _get_spec_stationarity(self):
        itakura_distortion = 1
        return 0.2 / (itakura_distortion - 0.8)


    def _get_rms_ratio(self, frame_idx):
        samples_per_frame = self.params.samples_per_frame
        rms_offset = self.params.rms_offset
        hanning_win_len = self.params.hanning_window_length
        hanning_win_vals = self.params.hanning_window_vals
        audio_sample = self._original_audio
        curr_frame_start = frame_idx * samples_per_frame
        prev_frame_start = (frame_idx - 1) * samples_per_frame

        if prev_frame_start < 0:
            prev_frame_start = 0

        max_window_diff = len(audio_sample) - (curr_frame_start + rms_offset + hanning_win_len)

        if max_window_diff < 0:
            hanning_win_len += max_window_diff

        if hanning_win_len < 0:
            hanning_win_len = 0

        curr_sum = 0
        prev_sum = 0
        curr_frame_index = curr_frame_start + rms_offset
        prev_frame_index = prev_frame_start - rms_offset

        if prev_frame_index < 0:
            prev_frame_index = 0

        audio_slice = audio_sample[curr_frame_index:curr_frame_index + hanning_win_len]
        prev_audio_slice = audio_sample[prev_frame_index:prev_frame_index + hanning_win_len]

        # sine window len may be reduced (sine we are end of sample), make sure the
        # hanning window vals match up with our slice of the audio sample
        hanning_win_val = hanning_win_vals[:hanning_win_len]

        curr_sum = np.sum((audio_slice * hanning_win_val) ** 2)
        prev_sum = np.sum((prev_audio_slice * hanning_win_val) ** 2)

        if curr_sum == 0.0 and prev_sum == 0.0 and hanning_win_len == 0:
            return 0.0

        rms_curr = math.sqrt(float(curr_sum) / float(hanning_win_len))
        rms_prev = math.sqrt(float(prev_sum) / float(hanning_win_len))

        return (rms_curr / rms_prev)
    
            

        






