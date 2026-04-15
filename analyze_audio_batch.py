import csv
import multiprocessing as mp
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

TARGET_DIR = r"C:\Users\Fredy\Bioren\Dept P&D - Audios remora 07-04-26 - norsul rio"
CSV_PATH = 'fft/all_fft_results.csv'

def read_wav_normalized(path):
    fs, data = wavfile.read(path)
    print(f'{os.path.basename(path)}: fs={fs}, shape={data.shape}')
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return fs, data

def process_file(filename):
    try:
        path = os.path.join(TARGET_DIR, filename)
        fs, data = read_wav_normalized(path)
        N = len(data)
        window = np.hanning(N)
        windowed = data * window
        spec = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(N, 1/fs)
        mag = np.abs(spec) / (np.sum(window)/2)
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
        print(f"Processed {filename}: {len(freqs)} freq bins")
        # Peaks for console
        peaks, _ = find_peaks(mag_db, prominence=1.0)
        if peaks.size > 0:
            top_peak_freq = freqs[peaks[np.argmax(mag_db[peaks])]]
            print(f"  Top peak: {top_peak_freq:.2f} Hz")
        return filename, freqs.tolist(), mag_db.tolist()
    except Exception as e:
        print(f"Error {filename}: {e}")
        return filename, [], []

def main():
    if not os.path.exists(TARGET_DIR):
        raise ValueError(f"Target dir not found: {TARGET_DIR}")
    
    wav_files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files.")
    
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    
    with mp.Pool() as pool:
        results = pool.map(process_file, wav_files)
    
    # Filter successful
    results = [r for r in results if r[1]]
    
    filenames = [r[0] for r in results]
    
    # Assume all same freq len (take first)
    if not results:
        print("No files processed.")
        return
    freqs = np.array(results[0][1])
    num_bins = len(freqs)
    num_files = len(results)
    
    # Verify lengths, skip mismatched
    valid_results = []
    for fn, f, m in results:
        if len(f) == num_bins:
            valid_results.append((fn, f, m))
        else:
            print(f"Warning: {fn} freqs len {len(f)} != {num_bins}, skipping.")
    results = valid_results
    
    # Write transposed CSV
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['frequency'] + [r[0] for r in results]
        writer.writerow(header)
        for i in range(num_bins):
            row = [freqs[i]] + [results[j][2][i] for j in range(len(results))]
            writer.writerow(row)
    
    print(f"Saved {CSV_PATH} ({len(results)} files, {num_bins} freq bins)")

if __name__ == '__main__':
    main()

