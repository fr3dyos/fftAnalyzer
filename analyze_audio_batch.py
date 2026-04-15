import csv
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

TARGET_DIR = r"C:\Users\Fredy\Bioren\Dept P&D - Audios remora 07-04-26 - norsul rio"
PROCESSED_FILE = 'fft/processed_files.txt'
BATCH_SIZE = 50

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

def compute_fft(filename):
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
        print(f"  Processed: {len(freqs)} freq bins")
        peaks, _ = find_peaks(mag_db, prominence=1.0)
        if peaks.size > 0:
            top_peak_idx = np.argmax(mag_db[peaks])
            top_peak_freq = freqs[peaks[top_peak_idx]]
            print(f"  Top peak: {top_peak_freq:.2f} Hz")
        return filename, freqs.tolist(), mag_db.tolist()
    except Exception as e:
        print(f"  Error: {e}")
        return filename, [], []

def main():
    if not os.path.exists(TARGET_DIR):
        raise ValueError(f"Target dir not found: {TARGET_DIR}")

    all_wav_files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith('.wav')]
    total_files = len(all_wav_files)
    print(f"Found {total_files} WAV files.")

    os.makedirs('fft', exist_ok=True)

    # Load processed
    processed = set()
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r') as pf:
            processed = set(line.strip() for line in pf if line.strip())
        skipped = len(processed)
        print(f"Skipped {skipped} already processed files.")
    else:
        open(PROCESSED_FILE, 'w').close()

    wav_files = [f for f in all_wav_files if f not in processed]

    if not wav_files:
        print("All files processed.")
        return

    print(f"Processing {len(wav_files)} remaining files.")

    batch_num = 1 + max(0, len([f for f in os.listdir('fft') if f.startswith('all_fft_results_batch_')]))
    
    for start in range(0, len(wav_files), BATCH_SIZE):
        batch = wav_files[start:start + BATCH_SIZE]
        batch_results = []
        for j, filename in enumerate(batch, start + 1):
            print(f"\nProcessing file {start + j} of {total_files}: {filename}")
            res = compute_fft(filename)
            if res[1]:  # valid
                batch_results.append(res)
            # Always mark processed
            with open(PROCESSED_FILE, 'a') as pf:
                pf.write(filename + '\n')
        if batch_results:
            # Assume same freqs
            freqs = np.array(batch_results[0][1])
            num_bins = len(freqs)
            # Verify
            valid_batch = []
            for fn, f, m in batch_results:
                if len(f) == num_bins:
                    valid_batch.append((fn, f, m))
                else:
                    print(f"Warning: {fn} len mismatch, skipping.")
            if valid_batch:
                csv_path = f'fft/all_fft_results_batch_{batch_num}.csv'
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    header = ['frequency'] + [r[0] for r in valid_batch]
                    writer.writerow(header)
                    for i in range(num_bins):
                        row = [freqs[i]] + [valid_batch[k][2][i] for k in range(len(valid_batch))]
                        writer.writerow(row)
                print(f"Saved batch {batch_num}: {len(valid_batch)} files ({num_bins} bins)")
                batch_num += 1

    print("Processing complete. Check fft/all_fft_results_batch_*.csv")

if __name__ == '__main__':
    main()

