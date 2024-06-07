from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent

PLOTS_PATH = os.path.join(BASE_DIR, 'assets', 'plots')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
SIGNALS_PATH = os.path.join(BASE_DIR, 'assets', 'signals')
SOFA_PATH = os.path.join(BASE_DIR, 'assets', 'sofa_files')
IMAGES_PATH = os.path.join(BASE_DIR, 'assets', 'images')

FS = 44100

RMS_THRESHOLD = 0.004

N_SPEAKERS = 9

LOW_FREQ_CUTOFF = 1500
HIGH_FREQ_CUTOFF = 1500
ORDER = 6
