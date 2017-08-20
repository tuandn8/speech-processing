import wtextractor
import matplotlib.pyplot as plt 


from scipy.io import wavfile

audio_file = "test\\VIVOSSPK02_R002.wav"

fs, original_audio = wavfile.read(audio_file)

print(original_audio.shape)

plt.plot(original_audio)
plt.show()

extrator = wtextractor.WaveletExtractor()

extrator.get_wavelet_features(original_audio, fs)