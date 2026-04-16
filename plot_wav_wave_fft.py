#!/usr/bin/env python3

"""Plot single WAV: time waveform + FFT spectrum (subplots)."""
import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def plot_wav_wave_fft(wav_path, max_time=None, max_freq=None, channel=None):
    # Load WAV
    fs, data = wavfile.read(wav_path)
    print(f"Loaded {wav_path}: fs={fs}Hz, duration={len(data)/fs:.1f}s")
    
    # Channel
    if data.ndim > 1:
        if channel is None:
            data = data.mean(axis=1)
        else:
            data = data[:, channel]
    
    # Normalize
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    
    time = np.arange(len(data)) / fs
    
    """# Limit time
    if max_time is not None:
        plot_max_time = max_time
    max_time = time[-1]
    mask = time <= max_time
    time = time[mask]
    data = data[mask]"""
    
    # FFT RMS
    N = len(data)
    window = np.hanning(N)
    windowed = data * window
    spec = rfft(windowed)
    freqs = rfftfreq(N, 1/fs)
    mag_linear = np.abs(spec) / (np.sum(window)/2)
    rms = np.sqrt(mag_linear ** 2)  # RMS spectrum
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform
    ax1.plot(time, data)
    ax1.set_title(f'Waveform - {os.path.basename(wav_path)}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    if max_time is not None:
        ax1.set_xlim(0, max_time)
    ax1.grid(True, alpha=0.3)
    
    # FFT RMS (semilogy)
    if max_freq is None:
        max_freq = fs / 2
    ax2.semilogy(freqs, rms)
    ax2.set_xlim(0, max_freq)
    ax2.set_title('FFT RMS Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('RMS Magnitude')

    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot WAV waveform + FFT')
    parser.add_argument('wav_file', help='Path to WAV file')
    parser.add_argument('--max-time', type=float, help='Max time (s) to plot')
    parser.add_argument('--max-freq', type=float, default=5000, help='Max freq (Hz)')
    parser.add_argument('--channel', type=int, help='Channel (0=left, 1=right)')
    args = parser.parse_args()
    
    plot_wav_wave_fft(args.wav_file, args.max_time, args.max_freq, args.channel)
