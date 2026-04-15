# Rêmora Audio FFT Analyzer

Author: Fredy Osorio (ing.fredyosorio.com)

## Overview
Python project for batch FFT analysis of WAV audio files from MarineTraffic/Remora recordings.

## Tools

### 1. fft_plot.py (Original)
Segmented FFT (20s chunks) for WAV dirs, parallel processing, per-file CSV, peaks, overlay plot.

Usage: `python fft_plot.py [wavdir] [--channel N] [--max-freq F] [--linear]`

### 2. Batch Analysis for Bioren Dir
- **analyze_audio_batch.py**: Full-file FFT (~1min files) for all WAVs in `C:\\Users\\Fredy\\Bioren\\Dept P&D - Audios remora 07-04-26 - norsul rio`, saves `fft/all_fft_results.csv` (frequency + amplitude_dB per file).
- **plot_fft_csv.py**: Loads CSV, overlays all FFT curves.

## Setup
```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage - Batch
```
python analyze_audio_batch.py  # Processes ~280 files (parallel), prints peaks
python plot_fft_csv.py  # Interactive plot (zoom/pan)
```

## Output
- CSV: Rows=freq bins, cols=files' dB amps (skips mismatched fs).
- Plots: Matplotlib overlay, xlim 0-5kHz, legend.

## Dependencies
requirements.txt: numpy, scipy, matplotlib.

## Notes
- Multiprocessing scales to CPUs.
- Test data in wav/fft/.
- testFileCreator.py generates test WAVs.
- Target dir hardcoded, ~44.1kHz assumed.

## .gitignore
Ignores .venv, *.csv, fft/, TODO.md.

## Author
Fredy Osorio
ing.fredyosorio@gmail.com

