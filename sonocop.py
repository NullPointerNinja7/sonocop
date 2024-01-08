# sonocop.py
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
import signal
import analyze_onnx
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

    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        application_path = sys._MEIPASS  
    else:
        application_path = os.path.dirname(os.path.abspath(__file__)) 
        
    model_path = os.path.join(application_path, "model.ort")

    analyze_onnx.analyze_onnx(providers, args.input_dir, model_path, args.skip_info, args.skip_warning)

if __name__ == "__main__":
    main()