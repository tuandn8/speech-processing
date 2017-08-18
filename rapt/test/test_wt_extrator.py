from rapt import wtextractor

from scipy.io import wavfile

audio_file = "VIVOSSPK02_R002.wav"

fs, original_audio = wavfile.read(audio_file)

extrator = wtextractor.WaveletExtractor()

extrator.get_wavelet_features(original_audio, fs)