from functools import lru_cache
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@lru_cache(maxsize=None)
def butter_lowpass(lowcut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    normal_lowcut = min(lowcut / nyq, 0.99)
    return butter(order, normal_lowcut, btype='low')


def lowpass_filter(data: np.ndarray, lowcut: float = 1000.0, fs: float = 44100.0, order: int = 5) -> np.ndarray:
    b, a = butter_lowpass(lowcut, fs, order)
    return filtfilt(b, a, data)


def extract_top_peaks(df: pd.DataFrame, n_peaks: int = 3) -> list[float]:
    """
    Extract top N peaks from one spectrum column.
    """
    freqs = df['frequency'].to_numpy(dtype=np.float64)
    fs = float(freqs[-1] * 2)
    peaks_list = []
    
    for col in df.select_dtypes(include=[np.number]).columns.drop('frequency'):
        data = df[col].to_numpy(dtype=np.float64)
        filtered = lowpass_filter(10 ** (data / 20.0), lowcut=1000.0, fs=fs)
        
        std_f, mean_f = filtered.std(), filtered.mean()
        peaks, _ = find_peaks(filtered, prominence=std_f * 0.1, height=mean_f * 1.5)
        if peaks.size:
            peak_heights = filtered[peaks]
            top_idx = np.argsort(peak_heights)[-n_peaks:][::-1]
            for rank, rel_idx in enumerate(top_idx, 1):
                p_idx = peaks[rel_idx]
                peak_freq = float(freqs[p_idx])
                peaks_list.append(peak_freq)
                print(f"  {col} peak{rank}: {peak_freq:.1f} Hz")
    
    return peaks_list


def save_peaks_histogram(all_peaks: list[float], out_dir: str = 'fft') -> None:
    peak_arr = np.asarray(all_peaks)
    bin_size = 50
    bins = np.arange(0, peak_arr.max() + bin_size, bin_size)
    counts, edges = np.histogram(peak_arr, bins=bins)
    centers = edges[:-1] + bin_size / 2

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(centers, counts, width=bin_size * 0.8)
    ax.set_title('Top 3 Peaks Frequencies Histogram (50 Hz bins)')
    ax.set_xlabel('Freq Bin Center (Hz)')
    ax.set_ylabel('Peak Count')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(centers)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    hist_path = os.path.join(out_dir, 'peaks_histogram.jpg')
    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame({'peak_freq': all_peaks}).to_csv(
        os.path.join(out_dir, 'peaks_summary.csv'), index=False
    )
    print(f"✓ Top 3 peaks histogram → {hist_path}")
