
from scipy.io import wavfile
import matplotlib.pyplot as plt
import rapt
import rapt_mau

audio_file = "/home/vnc/workspace/speech-processing/VIVOSSPK02_R002.wav"

[fs, s] = wavfile.read(audio_file)

algo = rapt.Rapt()
(nccf, freq) = algo.pitch_tracking(s, fs)

(nccf_mau, freq_mau) = rapt_mau.rapt_with_nccf(audio_file)

plt.plot(freq, 'r--')
plt.plot(freq_mau, 'b--')
plt.show()

