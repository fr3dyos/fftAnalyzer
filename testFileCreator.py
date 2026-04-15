import numpy as np
from scipy.io import wavfile


def generate_test_wav(filename="test_signal.wav", duration=30, sample_rate=44100*2):
    """
    Generates a .wav file with 5 specific frequencies and white noise.
    """
    print(f"Generating {filename}...")
    
    # 1. Setup time axis
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 2. Define the 5 arbitrary frequencies (in Hz)
    # Example: A4 (440), E5 (659), C6 (1046), and two non-musical freqs
    frequencies = [440, 659, 1046, 2500, 5000]
    
    # 3. Generate the signal
    signal = np.array([np.sin(2 * np.pi * f * t) for f in frequencies]).sum(axis=0)
    
    # 4. Generate White Noise
    # Mean of 0, standard deviation of 0.3
    noise = np.random.randn(len(t)) * 0.3
    
    # 5. Mix signal and noise
    final_signal = signal + noise
    
    # 6. Normalize to prevent clipping (keep values between -1.0 and 1.0)
    max_val = np.max(np.abs(final_signal))
    if max_val > 0:
        final_signal = final_signal / max_val
        
    # 7. Convert to 16-bit PCM format for standard compatibility
    # Scale to range [-32767, 32767] and cast to int16
    audio_data = (final_signal * 32767).astype(np.int16)
    
    # 8. Save the file
    # Ensure directory exists if path is provided
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    wavfile.write(filename, sample_rate, audio_data)
    
    print(f"Success! Saved '{filename}'")
    print(f" - Duration: {duration}s")
    print(f" - Sample Rate: {sample_rate}Hz")
    print(f" - Frequencies: {frequencies}")
    print(" - Noise: White Gaussian")