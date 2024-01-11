import sys
import os

STUDY_NAME="study_sonocop_fixed1"
STORAGE_URL = "sqlite:///best_training_params.db"
MODEL_FILENAME="model.ort"

def getmodelpath():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        application_path = sys._MEIPASS  
        model_path = os.path.join(application_path, "common", MODEL_FILENAME)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(application_path, MODEL_FILENAME)
    return model_path
