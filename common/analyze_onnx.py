import onnxruntime as ort
import numpy as np
import common.create_fft as create_fft
import os
import numpy as np

#returns percentage chance this is a bad file

def print_score(file_path, skip_warning, skip_info, score_percent):
    # Check if the score is above 40%
    if score_percent >= 50:
        if not skip_warning:
            # Red color code
            print(f"\033[91m{file_path.encode('utf-8', errors='replace')} is transcoded. Chance of Transcode: {score_percent:.2f}%\033[0m")
    elif not skip_info:
        print(f"{file_path.encode('utf-8', errors='replace')} is not transcoded. Chance of Transcode: {score_percent:.2f}%")

def analyze_onnx(providers, input_dir, model_filename, skip_info=False, skip_warning=False):
    print(f"Analyzing .flac files")

    bad_file_count = 0  # Initialize counter for bad files

    for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.flac'):
                    file_path = os.path.join(root, filename)
                    S_dBFS = create_fft.make_spectrogram(file_path, False)
                    # Check if the spectrogram is empty
                    if S_dBFS.size == 0:
                        print(f"Skipping empty (silent?) spectrogram for file: {file_path}")
                        continue
                    # Debug: Check the shape of S_dBFS
                    #print(f"File: {filename.encode('utf-8', errors='replace')}, Shape before expansion: {S_dBFS.shape}")

                    # Load the ONNX model
                    session = ort.InferenceSession(model_filename, providers=providers)

                    # Convert and expand dimensions
                    S_dBFS = S_dBFS.astype(np.float32)
                    S_dBFS = np.expand_dims(S_dBFS, axis=0)  # Add the batch dimension
                    S_dBFS = np.expand_dims(S_dBFS, axis=0)  # Add the channel dimension

                    # Debug: Check the shape after expansion
                    #print(f"File: {filename.encode('utf-8', errors='replace')}, Shape after expansion: {S_dBFS.shape}")

                    # Run the model
                    outputs = session.run(None, {'input': S_dBFS})
                    score_percentage = (1 - outputs[0].item()) * 100  # Convert to percentage
                    if (score_percentage > 50):
                        bad_file_count += 1  # Increment the bad file count
                    print_score(file_path, skip_warning, skip_info, score_percentage)

    return bad_file_count
