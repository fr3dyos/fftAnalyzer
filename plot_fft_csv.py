import csv
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_PATH = 'fft/all_fft_results.csv'

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Run analyze_audio_batch.py first.")
    
    freqs = []
    amps = {}
    
    with open(CSV_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        filenames = header[1:]
        for row in reader:
            freqs.append(float(row[0]))
            for i, amp_str in enumerate(row[1:]):
                amps.setdefault(filenames[i], []).append(float(amp_str))
    
    freqs = np.array(freqs)
    
    plt.figure(figsize=(12, 8))
    for fn, amp_list in amps.items():
        plt.plot(freqs, np.array(amp_list), linewidth=0.8, label=os.path.splitext(fn)[0][:20] + '...', alpha=0.7)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('FFT Results - All Files Overlaid')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5000)  # Adjust as needed
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

