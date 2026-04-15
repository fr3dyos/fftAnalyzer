# Rêmora Audio FFT Analyzer

Author: Fredy Osorio (ing.fredyosorio@gmail.com)

## Overview
Python toolkit for FFT analysis of MarineTraffic/Rêmora audio recordings.

## Tools

### fft_plot.py (Segmented WAV Analyzer)
Processes WAV dir into 20s segments, FFT + peaks + per-file CSV + overlay plot.

Usage: `python fft_plot.py [wavdir] [--channel N] [--max-freq F] [--linear]`

### Batch Full-File FFT (Bioren Audio)
- **analyze_audio_batch.py**: Low-mem sequential batches (50 files), full FFT, batch CSVs (`fft/all_fft_results_batch_N.csv`: freq + amps_dB), resume (processed_files.txt), progress "file X of TOTAL".

  Usage: `python analyze_audio_batch.py`

- **plot_fft_csv.py**: Plot specific batch CSV or all batches overlaid (limit 100 curves).

  Usage: `python plot_fft_csv.py [csv_path]` (e.g. `fft/all_fft_results_batch_1.csv`)

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Output Examples
- CSVs: freq bin rows, file amp cols.
- Plots: Interactive matplotlib, 0-5kHz.

## Features
- Resume after interrupt.
- Peak detection console.
- Git ready (.gitignore, initial commit).

## Test Data
wav/, testFileCreator.py.

## Dependencies
numpy scipy matplotlib

## License
MIT © Fredy Osorio

