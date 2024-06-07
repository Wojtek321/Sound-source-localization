from utils import measurement_number, ITD, ILD, MODELS_PATH, SIGNALS_PATH, SOFA_PATH
from scipy.signal import convolve
from scipy.io import wavfile
from netCDF4 import Dataset
from librosa import resample
from joblib import load
import numpy as np
import gradio as gr
import os


MODELS = {
    'Decision Tree': 'decision_tree_regressor.joblib',
    'K Neighbors': 'k_neighbors_regressor.joblib',
    'Random Forest': 'random_forest_regressor.joblib',
}

SIGNALS = {
    'Eye of the tiger': 'eyeofthetiger.wav',
    'Nirvana': 'nirvana.wav',
    'Scooby Doo': 'scooby.wav',
    'Shrek': 'shrek.wav',
    'Guitar': 'guitar.wav',
    'Podcast': 'podcast.wav',
    'Trumpet': 'trumpet.wav',
}

sofa_file = Dataset(os.path.join(SOFA_PATH, 'sadie.sofa'))
SOFA_FS = sofa_file['Data.SamplingRate'][:][0]
ARR = sofa_file['SourcePosition'][:]

TIME = 1


def pred(angle, sound_id, model_id):
    model = load(os.path.join(MODELS_PATH, MODELS[model_id]))
    scaler_itd = load(os.path.join(MODELS_PATH, 'scaler_itd.joblib'))
    scaler_ild = load(os.path.join(MODELS_PATH, 'scaler_ild.joblib'))
    scaler_y = load(os.path.join(MODELS_PATH, 'scaler_y.joblib'))

    fs, data = wavfile.read(os.path.join(SIGNALS_PATH, SIGNALS[sound_id]))
    data = data / (np.max(np.abs(data)))
    data = resample(data, orig_sr=fs, target_sr=SOFA_FS)

    angle = (angle + 360) % 360 if angle < 0 else angle

    idx = measurement_number(angle, ARR)
    H_l = sofa_file["Data.IR"][idx, 1, :]
    H_r = sofa_file["Data.IR"][idx, 0, :]

    Y_l = convolve(data, H_l)
    Y_r = convolve(data, H_r)

    sound = np.column_stack((Y_l, Y_r))

    Y_l = Y_l[:int(TIME * fs)]
    Y_r = Y_r[:int(TIME * fs)]

    itd = ITD(Y_l, Y_r, SOFA_FS)
    ild = ILD(Y_l, Y_r)
    itd = scaler_itd.transform(np.reshape(itd, (-1, 1)))
    ild = scaler_ild.transform(np.reshape(ild, (-1, 1)))

    X = np.column_stack((itd, ild))
    Y = model.predict(X)
    Y = scaler_y.inverse_transform(np.reshape(Y, (-1, 1)))
    pred_angle = np.ravel(Y)[0]
    pred_angle = (pred_angle - 360) if pred_angle > 180 else pred_angle

    return (SOFA_FS, sound), pred_angle


demo = gr.Interface(
    title='Simulating spatial sound and determining its source direction.',
    fn=pred,
    inputs=[
        gr.Slider(minimum=-90, maximum=90, value=0, step=5, label='Azimuth angle', info='Angle equal to 0 means a sound from the opposite side, closer to -90 is a sound from the left, and closer to 90 from the right.'),
        gr.Dropdown(choices=[key for key, _ in SIGNALS.items()], value='Eye of the tiger', multiselect=False, label='Sound to be simulated'),
        gr.Dropdown(choices=[key for key, _ in MODELS.items()], value='Decision Tree', multiselect=False, label='Model to be used'),
    ],
    outputs=[
        gr.Audio(label='Simulated sound'),
        gr.Textbox(label='Predicted azimuth angle'),
    ]
)

demo.launch(inbrowser=False)
