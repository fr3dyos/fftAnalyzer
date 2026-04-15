import argparse
import csv
import multiprocessing as mp
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks  # Added for peak detection
import matplotlib.pyplot as plt


def read_wav_normalized(path, channel=None):
    fs, data = wavfile.read(path)
    print(f'fs: {fs}')
    # select channel if stereo
    if data.ndim > 1:
        if channel is None:
            data = data.mean(axis=1)  # mix to mono
        else:
            data = data[:, channel]
    # convert integers to float in [-1,1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return fs, data


def process_file(file, directory, channel, show_db, fft_dir):
    """Process a single WAV file for FFT analysis."""
    try:
        path = os.path.join(directory, file)
        fs, data = read_wav_normalized(path, channel)
        segment_size = 20 #in s
        segment_length = int(segment_size * fs)
        num_segments = (len(data) + segment_length - 1) // segment_length

        segments_data = []
        csv_file = os.path.join(fft_dir, f"{os.path.splitext(file)[0]}_fft.csv")
        freqs_list = None
        mags_list = []

        for i in range(num_segments):
            start = i * segment_length
            end = min(start + segment_length, len(data))
            segment = data[start:end]
            N = len(segment)
            if N == 0:
                continue
            window = np.hanning(N)
            windowed = segment * window
            spec = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(N, d=1.0 / fs)
            mag = np.abs(spec) / (np.sum(window) / 2.0)
            if show_db:
                mag_plot = 20 * np.log10(np.maximum(mag, 1e-12))
            else:
                mag_plot = mag

            if freqs_list is None:
                freqs_list = freqs
            mags_list.append(mag_plot)
            title = f"{file} Segment {i+1}"
            segments_data.append((freqs, mag_plot, title, fs))

            # Print Analysis Info to Shell
            print(f"\n[{file}] Segment {i+1} Analysis:")
            max_analysis_freq = freqs[-1]
            print(f"  > Max Frequency Reached: {max_analysis_freq:.2f} Hz")

            prominence = 1.0 if show_db else 0.001
            peaks, _ = find_peaks(mag_plot, prominence=prominence)

            if len(peaks) > 0:
                peak_freqs = freqs[peaks]
                peak_mags = mag_plot[peaks]
                sorted_indices = np.argsort(peak_mags)[::-1]
                top_n = 10
                print(f"  > Detected Peaks (Top {min(len(peaks), top_n)} sorted by magnitude):")
                for idx in sorted_indices[:top_n]:
                    f_val = peak_freqs[idx]
                    m_val = peak_mags[idx]
                    unit = "dB" if show_db else "Linear"
                    print(f"    - {f_val:.2f} Hz (Mag: {m_val:.2f} {unit})")
            else:
                print("  > No significant peaks detected.")

        # Write CSV if there are segments
        if mags_list:
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['Frequency'] + [f'Mag_Segment_{i+1}' for i in range(len(mags_list))]
                writer.writerow(header)
                for j, freq in enumerate(freqs_list):
                    row = [freq] + [mags[j] if j < len(mags) else '' for mags in mags_list]
                    writer.writerow(row)

        return segments_data
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return []


def plot_fft_segments(directory, channel=None, max_freq=None, show_db=True):
    files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
    if not files:
        raise ValueError("No .wav files found in directory")

    # Ensure fft folder exists
    fft_dir = 'fft'
    os.makedirs(fft_dir, exist_ok=True)

    # Use multiprocessing to process files in parallel
    with mp.Pool(processes=min(mp.cpu_count(), len(files))) as pool:
        results = pool.starmap(process_file, [(file, directory, channel, show_db, fft_dir) for file in files])

    all_segments_data = [segment for file_segments in results for segment in file_segments]

    if not all_segments_data:
        raise ValueError("No segments to plot")

    # Plot all in one figure
    fig, ax = plt.subplots(figsize=(10, 7))
    for freqs, mag_plot, title, fs in all_segments_data:
        ax.plot(freqs, mag_plot, linewidth=0.7, label=title)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]" if show_db else "Magnitude")
    ax.set_title("FFT Analysis of All Segments")
    ax.grid(True)
    xlim_max = max_freq if max_freq is not None else fs / 2
    ax.set_xlim(0, xlim_max)
    #ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot FFT analysis of .wav files in 20s segments")
    parser.add_argument("wavdir", nargs='?', default='wav', help="Path to directory containing .wav files (default: 'wav')")
    parser.add_argument("--channel", type=int, default=None,
                        help="Channel index (0-based). If omitted, channels are mixed to mono")
    parser.add_argument("--max-freq", type=float, default=None,
                        help="Maximum frequency to show on the x-axis (Hz)")
    parser.add_argument("--linear", action="store_true",
                        help="Show linear magnitude instead of dB")
    args = parser.parse_args()

    if not os.path.isdir(args.wavdir):
        raise SystemExit(f"Directory not found: {args.wavdir}")

    plot_fft_segments(args.wavdir, channel=args.channel, max_freq=args.max_freq, show_db=not args.linear)


if __name__ == "__main__":
    main()
