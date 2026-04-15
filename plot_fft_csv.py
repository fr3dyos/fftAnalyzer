import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_all_batches():
    amps = {}
    freqs = None
    for csv_path in sorted(glob.glob('fft/all_fft_results_batch_*.csv')):
        print(f"Loading {csv_path}")
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            filenames = header[1:]
            csv_freqs = []
            for row in reader:
                csv_freqs.append(float(row[0]))
            this_freqs = np.array(csv_freqs)
            if freqs is None:
                freqs = this_freqs
            for i, fn in enumerate(filenames):
                row_amps = []
                with open(csv_path, 'r') as csvfile2:  # reread for amps
                    r = csv.reader(csvfile2)
                    next(r)
                    for row in r:
                        row_amps.append(float(row[i+1]))
                amps[fn] = np.array(row_amps)
    return freqs, amps

def main():
    freqs, amps = load_all_batches()
    if freqs is None:
        print("No batch CSVs found.")
        return
    
    plt.figure(figsize=(15, 10))
    plotted = 0
    for fn, amp_list in amps.items():
        if plotted >= 100:  # limit for clarity
            break
        short_fn = os.path.splitext(fn)[0][:25] + '...' if len(os.path.splitext(fn)[0]) > 25 else os.path.splitext(fn)[0]
        plt.plot(freqs, amp_list, linewidth=0.6, label=short_fn, alpha=0.6)
        plotted += 1
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('FFT Results - All Files/Batches Overlaid (up to 100 for clarity)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5000)
    if plotted < 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print(f"Plotted {plotted} files from batches.")

if __name__ == '__main__':
    main()

