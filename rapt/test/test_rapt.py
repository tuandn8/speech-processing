
from scipy.io import wavfile
import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt
import rapt



audio_file = "test\\VIVOSSPK02_R002.wav"

[fs, s] = wavfile.read(audio_file)

if len(s.shape) > 1:
    s = s[:,0]/2.0  + s[:,1]/2.0
    s = s.astype(int)

algo = rapt.Rapt()
nccf, freq  = algo.pitch_tracking(s, fs)

#freq_mau = rapt_mau.rapt(audio_file)
print(len(s) / len(freq))
plt.plot(freq, 'r-')
# plt.plot(s)
# rparams = raptparams.Raptparams()
# plt.plot(np.linspace(0, len(s), num=len(freq)), freq)
#plt.plot(freq_mau, 'bo')
plt.show()

print (freq)

# params = raptparams.Raptparams()
# nccf = nccfparams.Nccfparams()
# is_first_pass = False
# fs = 2000

# # value 'n' in NCCF equation
# nccf.samples_correlated_per_lag = int(round(params.correlation_window_size * fs))

# # start value of k in NCCF equation
# if is_first_pass:
#     nccf.shortest_lag_per_frame = int(round(fs / params.maximum_allowed_freq))
# else:
#     nccf.shortest_lag_per_frame = 0

# # value 'K' in NCCF equation
# nccf.longest_lag_per_frame = int(round(fs / params.minimum_allowed_freq))

# # value z in NCCF equation
# nccf.samples_per_frame = int(round(params.frame_step_size * fs))

# # value of M-1 in NCCF equation
# nccf.max_frame_count = int(round(float(5875) / float(nccf.samples_per_frame)) - 1)


# print('shortest_lag_per_frame = ', nccf.shortest_lag_per_frame)
# print('longest_lag_per_frame = ', nccf.longest_lag_per_frame)
# print('samples_per_frame = ', nccf.samples_per_frame)
# print('max_frame_count = ', nccf.max_frame_count)

# params.sample_rate_ratio = 8.0
# param= (params, nccf)

# first_pass = sio.loadmat('first_pass.mat')
# print (first_pass['first_pass'].shape)
# first_pass = np.asarray(first_pass['first_pass']).tolist()

# print(rapt_mau._get_correlations_for_input_lags(s, 0, first_pass['first_pass'], 39, param))

