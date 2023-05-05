import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import os

# 音声ファイルを読み込む
DIR = os.path.dirname(__file__)
PATH = os.path.join(DIR,"sample.wav")
data, samplerate = librosa.load(PATH)

# スペクトログラムを計算する
window_size = 1024
hop_size = 512
window = signal.windows.hann(window_size) # 窓関数
freq, t, yf = signal.stft(data, fs = samplerate, window = window, nperseg = window_size, noverlap = window_size-hop_size) # フーリエ
_, y_inv = signal.istft(yf, fs = samplerate, window = window, nperseg = window_size, noverlap = window_size-hop_size) # 逆フーリエ

# グラフを描画する
fig, axs = plt.subplots(3, 1, figsize = (8, 8))

# 音声波形を描画する
time = np.arange(len(data)) / samplerate
axs[0].plot(time, data)
axs[0].set_title("Original signal")
axs[0].set_ylabel("Magnitude")

# スペクトログラムを描画する
pcm = axs[1].pcolormesh(t, freq/1000, 10 * np.log10(np.abs(yf)), cmap = "viridis")
axs[1].set_title("Spectrogram")
axs[1].set_ylabel("Frequency[kHz]")
cax = axs[1].inset_axes([1.05, 0, 0.05, 1], transform=axs[1].transAxes, )
fig.colorbar(pcm, cax = cax)

# フーリエ逆変換した波形を描画する
time_inv = np.arange(len(y_inv)) / samplerate
axs[2].plot(time_inv, y_inv)
axs[2].set_title("Re-synthesized signal")
axs[2].set_xlabel("Time[s]")
axs[2].set_ylabel("Magnitude")

# グラフを表示する
plt.tight_layout() # 余白を調整
save_PATH = os.path.join(DIR, "ex1")
plt.savefig(save_PATH, bbox_inches='tight')
plt.show()