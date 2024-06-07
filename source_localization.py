from utils import ITD, ILD, RMS, FS, RMS_THRESHOLD, MODELS_PATH
from joblib import load
from gui.app import Window
import sounddevice as sd
import numpy as np
import os


MODEL = 'k_neighbors_regressor.joblib'

root = Window(round_to_speaker=True)

devices = sd.query_devices()
print(devices)
mic_name = devices[1]['name']


model = load(os.path.join(MODELS_PATH, MODEL))

scaler_itd = load(os.path.join(MODELS_PATH, 'scaler_itd.joblib'))
scaler_ild = load(os.path.join(MODELS_PATH, 'scaler_ild.joblib'))
scaler_y = load(os.path.join(MODELS_PATH, 'scaler_y.joblib'))


def audio_callback(indata, frames, time, status):
    # indata /= np.max(np.abs(indata))

    if RMS(np.sum(indata, axis=1)/2) > RMS_THRESHOLD:
        Y_l = indata[:,0]
        Y_r = indata[:,1]

        itd = ITD(Y_l, Y_r, fs=FS)
        ild = ILD(Y_l, Y_r)

        itd = scaler_itd.transform(np.reshape(itd, (-1, 1)))
        ild = scaler_ild.transform(np.reshape(ild, (-1, 1)))

        X = np.column_stack((itd, ild))

        Y = model.predict(X)
        Y = scaler_y.inverse_transform(np.reshape(Y, (-1, 1)))
        angle = np.ravel(Y)[0]

        root.update_arrow(angle)



input = sd.InputStream(samplerate=FS, channels=2, device=mic_name, callback=audio_callback, blocksize=10000)


with input:
    root.mainloop()
