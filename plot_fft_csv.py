import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import butter, filtfilt

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a

def lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order)
    return filtfilt(b, a, data)

def rms_spectrum(spec_mag_linear):
    # RMS: sqrt(mean power per bin)
    power = spec_mag_linear ** 2
    rms = np.sqrt(power)
    return rms

def main():
    parser = argparse.ArgumentParser(description="Plot FFT CSV - RMS ylog default")
    parser.add_argument("csv_path", nargs='?', default=None, help="CSV path")
    parser.add_argument("--full", action="store_true", help="Raw dB")
    parser.add_argument("--lowcut", type=float, default=1000, help="Lowpass Hz")
    parser.add_argument("--scale", choices=['rms', 'db', 'linear'], default='rms', help="Scale: rms (ylog), db, linear")
    args = parser.parse_args()

    # Load
    if args.csv_path:
        df = pd.read_csv(args.csv_path)
    else:
        dfs = [pd.read_csv(p) for p in sorted(glob.glob('fft/all_fft_results_batch_*.csv'))]
        df = pd.concat(dfs, axis=1).loc[:, ~pd.concat(dfs, axis=1).columns.duplicated()]

    print("DF Info:\n", df.info())
    print("\nHead:\n", df.head())

    freqs = df['frequency'].values
    fs = freqs[-1] * 2
    amp_cols = df.select_dtypes(include=[np.number]).columns.drop('frequency')[:15]

    fig, ax = plt.subplots(figsize=(15, 9))
    plotted = 0
    for col in amp_cols:
        short = str(col)[:20]
        data = df[col].values

        if args.scale == 'db':
            plot_data = data
            ylabel = 'dB'
        elif args.scale == 'linear':
            # Inverse dB to linear
            plot_data = 10 ** (data / 20)
            ylabel = 'Linear Magnitude'
        else:  # rms default
            lin = 10 ** (data / 20)
            plot_data = rms_spectrum(lin)
            ylabel = 'RMS Spectrum'

        if not args.full:
            plot_data = lowpass_filter(plot_data, args.lowcut, fs)

        ax.semilogy(freqs, plot_data, linewidth=1.0, label=short, alpha=0.7)
        plotted += 1

    title = f"FFT {args.scale.upper()} {'Filtered' if not args.full else 'Full'} ({args.lowcut}Hz cutoff)"
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    #ax.set_xlim(0, 5000)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print(f"Plotted {plotted} files in y-log.")

if __name__ == '__main__':
    main()

