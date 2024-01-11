# sonocop.py
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
import signal
import common.analyze_onnx as analyze_onnx
import common.globals as globals
import sys

def signal_handler(sig, frame):
    print('\nCaught Ctrl-C, cleaning up...')
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="sonocop: Detect Transcodes.")

    # Common arguments
    parser.add_argument('input_dir', type=str, nargs='?', help="Directory to search for FLAC files.") 

    #parser.add_argument('--use_cpu', action='store_true', help="For use CPU instead of GPU")
    parser.add_argument('--skip_info', action='store_true', help="Skip informational messages including listing good file")
    parser.add_argument('--skip_warning', action='store_true', help="Skip warning messages including listing bad files")

    args = parser.parse_args()


    # Check if GPU is available and ONNX Runtime GPU package is installed
    #if ort.get_device() == 'GPU' and not args.use_cpu:
    #    providers = ['CUDAExecutionProvider']
    #    print("Using GPU")
    #else:
    providers = ['CPUExecutionProvider']
    #print("Using CPU")
        
    if not args.input_dir:
        parser.error('input_dir required.')   
    #print(f"Processing files from {args.input_dir}")    

    signal.signal(signal.SIGINT, signal_handler)

    model_path =globals.getmodelpath()

    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"File does not exist: {model_path}")
        sys.exit(-1)  # Exit the program with a non-zero exit code to indicate an error

    # If the file exists, continue with your program
    print(f"Opening model from {model_path}")

    bad_file_count = analyze_onnx.analyze_onnx(providers, args.input_dir, model_path, args.skip_info, args.skip_warning)
    sys.exit(bad_file_count)

if __name__ == "__main__":
    main()
