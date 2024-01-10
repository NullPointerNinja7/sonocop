import os
import numpy as np
import onnxruntime as ort
import common.create_fft as create_fft
import common.globals as globals
import sys
import argparse
import signal
import concurrent.futures
import threading
import traceback
import time
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Shared flag to signal threads to shutdown
shutdown_flag = threading.Event()

def signal_handler(sig, frame):
    print('\nCaught Ctrl-C, application will exit as soon as current threads finish. One minute...')
    shutdown_flag.set()  # Signal all threads to shutdown
    sys.exit(0)
    

def process_file(file_path, model_filename, providers, expected_label):
    if shutdown_flag.is_set():
        return  # Exit the function if shutdown is signaled  
    S_dBFS = create_fft.make_spectrogram(file_path, False)
    if S_dBFS.size == 0:
        return None

    # Load the ONNX model
    session = ort.InferenceSession(model_filename, providers=providers)
    S_dBFS = S_dBFS.astype(np.float32)
    S_dBFS = np.expand_dims(np.expand_dims(S_dBFS, axis=0), axis=0)

    # Run the model
    outputs = session.run(None, {'input': S_dBFS})
    score_percentage = (1 - outputs[0].item()) * 100
    predicted_probability = outputs[0].item()

    # Check prediction
    prediction = 0 if score_percentage >= 50 else 1
    prediction_type = 'TP' if prediction == 1 and expected_label == 1 else \
                      'FP' if prediction == 1 and expected_label == 0 else \
                      'TN' if prediction == 0 and expected_label == 0 else 'FN'

    # Return the prediction type, predicted probability, and actual label
    return prediction_type, predicted_probability, expected_label


def evaluate_model_accuracy(good_dir, bad_dir, model_filename, providers):
    signal.signal(signal.SIGINT, signal_handler)
    results = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    probabilities = []
    labels = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = set()
        for label, directory in [(1, good_dir), (0, bad_dir)]:
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    if shutdown_flag.is_set():
                        return  # Exit the function if shutdown is signaled      

                    if filename.endswith('.flac'):
                        file_path = os.path.join(root, filename)
                        future = executor.submit(process_file, file_path, model_filename, providers, label)
                        futures.add(future)

        while futures:
            time.sleep(0.5)
            if shutdown_flag.is_set():
                return  # Exit the function if shutdown is signaled      

            done, futures = concurrent.futures.wait(futures, timeout=0, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for future in done:
                try:
                    prediction_type, probability, label = future.result()
                    if prediction_type:
                        results[prediction_type] += 1
                    probabilities.append(probability)
                    labels.append(label)
                except Exception as e:
                    print(f"Exception occurred during processing: {e}")
                    traceback.print_exc()
                    exit(0)

    total_files = sum(results.values())
    accuracy = ((results['TP'] + results['TN']) / total_files) * 100 if total_files > 0 else 0
    precision = results['TP'] / (results['TP'] + results['FP']) if (results['TP'] + results['FP']) > 0 else 0
    recall = results['TP'] / (results['TP'] + results['FN']) if (results['TP'] + results['FN']) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate the accuracy for TP and TN separately
    accuracy_tp = (results['TP'] / (results['TP'] + results['FN'])) if (results['TP'] + results['FN']) > 0 else 0
    accuracy_tn = (results['TN'] / (results['TN'] + results['FP'])) if (results['TN'] + results['FP']) > 0 else 0

    # Create and save the extended report
    extended_report = (f"Model Extended Accuracy Report:\n"
                       f"Total Files: {total_files}\n"
                       f"Correct Predictions (TP + TN): {results['TP'] + results['TN']}\n"
                       f"Accuracy: {accuracy:.2f}%\n"
                       f"Precision: {precision:.2f}\n"
                       f"Recall: {recall:.2f}\n"
                       f"F1-Score: {f1_score:.2f}\n"
                       f"Confusion Matrix:\n"
                       f"\tTP: {results['TP']}  FP: {results['FP']}\n"
                       f"\tFN: {results['FN']}  TN: {results['TN']}")                       
    print(extended_report)
    with open("extended_accuracy_report.txt", "w") as file:
        file.write(extended_report)


    # Setup the figure and axes for a grid with 1 row and 2 columns
    # Increase the figure height to create space for text
    fig, ax = plt.subplots(1, 2, figsize=(20, 16))

    # Plot the calibration curve on the first subplot
    prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10)
    ax[0].plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    ax[0].plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    ax[0].set_xlabel('Mean Predicted Probability')
    ax[0].set_ylabel('Fraction of Positives')
    ax[0].set_title('Calibration Curve')
    ax[0].legend()
    ax[0].grid(True)

    # Plot the confusion matrix on the second subplot
    confusion_matrix = np.array([[results['TP'], results['FP']],
                                [results['FN'], results['TN']]])
    cax = ax[1].matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax[1])
    ax[1].set_title('Confusion Matrix')

    # Correcting the warning by setting the ticks before the labels
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(['Bad', 'Good'])
    ax[1].set_yticklabels(['Bad', 'Good'])

    # Calculate individual accuracies for TP and TN quadrants and convert to percentages
    accuracy_tp = (results['TP'] / (results['TP'] + results['FN'])) * 100 if (results['TP'] + results['FN']) > 0 else 0
    accuracy_tn = (results['TN'] / (results['TN'] + results['FP'])) * 100 if (results['TN'] + results['FP']) > 0 else 0

    # Annotate the confusion matrix with text
    for (i, j), value in np.ndenumerate(confusion_matrix):
        if i == j:  # For TP and TN
            quadrant_accuracy = accuracy_tp if i == 0 else accuracy_tn
            text = f"{value}\n{quadrant_accuracy:.2f}% Acc"
            color = 'yellow'
        else:  # For FP and FN
            text = f"{value}"
            color = 'black'
        ax[1].text(j, i, text, ha='center', va='center', color=color)

    # Add overall metrics as separate text entries aligned to the left
    metrics = {
        'Total Files:': total_files,
        'Correct Predictions (TP + TN):': results['TP'] + results['TN'],
        'Accuracy:': f"{accuracy:.2f}%",
        'Precision:': f"{precision:.2f}",
        'Recall:': f"{recall:.2f}",
        'F1 Score:': f"{f1_score:.2f}"
    }

    # Plot the metrics text below the subplots
    text_x = 0.05
    text_y_start = 0.15  # Starting y position for the metrics text
    for i, (label, value) in enumerate(metrics.items()):
        text_y = text_y_start - i * 0.015  # Increment the y position for each line
        fig.text(text_x, text_y, f"{label} {value}", ha='left', fontsize=12)

    # Manually adjust subplot params to avoid overlap and make room for the additional text
    fig.subplots_adjust(bottom=0.2)

    # Save the plot as a PNG image
    plt.savefig('combined_plot.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="report: Report on sonocop model accuracy")

    # Common arguments
    parser.add_argument('good_dir', type=str, nargs='?', help="Directory containing 'good' FLAC files.") 
    parser.add_argument('bad_dir', type=str, nargs='?', help="Directory containing 'bad' transcoded FLAC files.") 

    args = parser.parse_args()

    # Check if GPU is available and ONNX Runtime GPU package is installed
    #if ort.get_device() == 'GPU' and not args.use_cpu:
    #    providers = ['CUDAExecutionProvider']
    #    print("Using GPU")
    #else:
    providers = ['CPUExecutionProvider']
    #print("Using CPU")
        
    if not args.good_dir:
        parser.error('good_dir required.')   
    if not args.bad_dir:
        parser.error('bad_dir required.')   
    #print(f"Processing files from {args.input_dir}")    

    signal.signal(signal.SIGINT, signal_handler)

    model_path =globals.getmodelpath()
    print (f"Opening model from {model_path}")
    evaluate_model_accuracy(args.good_dir, args.bad_dir, model_path, providers)

if __name__ == "__main__":
    main()
