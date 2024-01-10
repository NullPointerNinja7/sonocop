import threading
import soundfile as sf
#from augment import augment_audio_for_training
import numpy as np
#from scipy.signal.windows import blackmanharris

#roll our own so we dont include the huge scipy library
def blackmanharris(N):
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    n = np.arange(N)
    w = a0 - a1 * np.cos(2 * np.pi * n / N) + a2 * np.cos(4 * np.pi * n / N) - a3 * np.cos(6 * np.pi * n / N)
    return w

WINDOW_SIZE=1024
SAMPLE_RATE=44100 # typical for FLAC
MIN_FREQ=16000 # can also try 14000
SECONDS_TO_USE=5 # approx number of seconds of audio to use from middle

# Shared flag to signal threads to shutdown
shutdown_flag = threading.Event()

# Function to calculate the energy of an audio signal
def calculate_energy(samples):
    return np.sum(samples ** 2) / len(samples)

def make_spectrogram(file_path, augment_audio=False, window_size=WINDOW_SIZE, overlap_percent=12, duration=SECONDS_TO_USE*1000, zero_padding_factor=1):
    
    # Read the audio file using soundfile
    samples, sample_rate = sf.read(file_path)

    #if (sample_rate != SAMPLE_RATE):
    #    print(f"Warning: treating sample rate {sample_rate} as {SAMPLE_RATE}")
    sample_rate = SAMPLE_RATE

    # Calculate start and end frames for a 5-second slice from the middle
    total_frames = len(samples)
    start_frame = (total_frames - duration * sample_rate // 1000) // 2
    end_frame = start_frame + duration * sample_rate // 1000
    if (end_frame >= total_frames):
        return np.array([])
    # Slice the audio data
    samples = samples[start_frame:end_frame]

    # Use only the first track if stereo
    if samples.ndim > 1:
        samples = samples[:, 0]

    # Check if the audio is silent
    if calculate_energy(samples) < 1e-6:
        print("The audio file is silent.")
        # Handle silent file (e.g., skip processing, return empty array, etc.)
        return np.array([])  # Example: returning an empty array

    #if (augment_audio):
    #    samples = augment_audio_for_training(samples, sample_rate)
    hop_length = int(window_size * (1 - overlap_percent / 100))  # 12% overlap trying to emulate sox
    n_fft = window_size * zero_padding_factor

    # Apply Blackman-Harris window
    window = blackmanharris(window_size)

    # Compute the spectrogram
    S = np.array([np.fft.rfft(window * samples[i:i + window_size], n=n_fft)
                  for i in range(0, len(samples) - window_size, hop_length)])

    # Convert to dBFS (Decibels relative to full scale)
    # Calculate the maximum absolute value of S to use as a reference
    max_S = np.max(np.abs(S))

    # Define a floor value that is very small, to avoid log(0)
    floor = 1e-10

    # Check if max_S is too small
    if max_S < floor:
        # Handle the case where max_S is too small (e.g., set S_dBFS to a default value or skip processing)
        print(f"************* SILENT FFT *** FIle has very small max S {file_path}. Skipping ****************")
        return np.array([])
    else:
        # Convert to dBFS (Decibels relative to full scale), ensuring no zero values
        S_dBFS = 20 * np.log10(np.maximum(np.abs(S) / max_S, floor))
    
    # Adjust dynamic range
    S_dBFS = S_dBFS - np.max(S_dBFS)  # Shift max to 0 dBFS
    # Set the range from -120 to 0 dBFS
    S_dBFS = np.clip(S_dBFS, -120, 0)
    return S_dBFS
