
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fft_utils import lowpass_filter, extract_top_peaks, save_peaks_histogram

def process_csv(csv_path: str, make_plots: bool = False) -> list[float]:
    """
    Process CSV for plots + peaks (plots only).
    """
    try:
        df = pd.read_csv(csv_path, dtype=np.float32)
        freqs = df['frequency'].to_numpy(dtype=np.float64)
        fs = float(freqs[-1] * 2)

        amp_cols = [c for c in df.columns if c != 'frequency' and pd.api.types.is_numeric_dtype(df[c])]

        if not make_plots:
            return extract_top_peaks(df)

        peaks_this_csv = extract_top_peaks(df)

        fig, ax = plt.subplots(figsize=(12, 8))
        # Re-compute filtered for plotting
        for col in amp_cols:
            data = df[col].to_numpy(dtype=np.float64)
            filtered = lowpass_filter(10 ** (data / 20.0), lowcut=1000.0, fs=fs)
            ax.semilogy(freqs, filtered, linewidth=0.8, label=str(col)[:25], alpha=0.7)

        ax.set_title(f"RMS ylog Filtered — {os.path.basename(csv_path)}")
        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('RMS')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        jpg_path = csv_path.replace('.csv', '_rms_filtered.jpg')
        fig.savefig(jpg_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        del df

        return peaks_this_csv

    except Exception as e:
        print(f"✗ {csv_path}: {e}")
        return []


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch plot FFT CSVs and extract top 3 peaks.')
    parser.add_argument('--plots', action='store_true', help='Generate spectrum plots (default: analysis only)')
    args = parser.parse_args()

    csv_files = sorted(glob.glob('fft/all_fft_results_batch_*.csv'))
    print(f"Processing {len(csv_files)} CSVs…")

    all_peaks: list[float] = []
    for csv_path in csv_files:
        all_peaks.extend(process_csv(csv_path, make_plots=args.plots))

    if all_peaks:
        save_peaks_histogram(all_peaks)

    print("Done.")
