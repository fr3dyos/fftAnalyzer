#!/usr/bin/env python3
\"\"\"Exclusive peaks analysis from FFT CSVs.\"\"\"

import glob
import pandas as pd
from fft_utils import extract_top_peaks, save_peaks_histogram


if __name__ == '__main__':
    csv_files = sorted(glob.glob('fft/all_fft_results_batch_*.csv'))
    print(f"Processing {len(csv_files)} CSVs for top peaks...")

    all_peaks = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_peaks.extend(extract_top_peaks(df))

    if all_peaks:
        save_peaks_histogram(all_peaks)
    print("Peaks analysis complete.")
