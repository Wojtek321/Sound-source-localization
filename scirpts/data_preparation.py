from utils import ITD, ILD, measurement_number, PLOTS_PATH, SIGNALS_PATH, SOFA_PATH, MODELS_PATH
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from scipy.signal import convolve
from librosa import resample
from netCDF4 import Dataset
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import os


TIME = 0.2


SIGNALS = ['eyeofthetiger.wav', 'guitar.wav', 'nirvana.wav', 'noise.wav', 'podcast.wav', 'voice.wav']
SOFA_FILES = ['sadie.sofa', 'club_fritz.sofa']

angles = np.concatenate((range(0, 91, 5), range(270, 360, 5)))
Y = []
ITDS = []
ILDS = []


# data visualization
with Dataset(os.path.join(SOFA_PATH, SOFA_FILES[0])) as sofa_file:
    sofa_fs = sofa_file['Data.SamplingRate'][:][0]
    ARR = sofa_file['SourcePosition'][:]

    fs, data = wavfile.read(os.path.join(SIGNALS_PATH, 'scooby.wav'))
    data = data / (np.max(np.abs(data)))
    data = data[:int(TIME * fs)]
    data = resample(data, orig_sr=fs, target_sr=sofa_fs)

    for angle in range(0, 360, 5):
        idx = measurement_number(angle, ARR)

        H_l = sofa_file["Data.IR"][idx, 1, :]
        H_r = sofa_file["Data.IR"][idx, 0, :]

        Y_l = convolve(data, H_l)
        Y_r = convolve(data, H_r)

        itd = ITD(Y_l, Y_r, sofa_fs)
        ild = ILD(Y_l, Y_r)

        ITDS.append(itd)
        ILDS.append(ild)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(13, 7)
    ax = axs[0]; ax.stem(range(0, 360, 5), np.dot(ITDS, 1000)); ax.set_title("ITD vs azimuth"); ax.set_xlabel("Azimuth [°]"); ax.set_ylabel("Time [ms]")
    ax = axs[1]; ax.stem(range(0, 360, 5), ILDS); ax.set_title("ILD vs azimuth"); ax.set_xlabel("Azimuth [°]"); ax.set_ylabel("Level difference [dB]")
    fig.set_tight_layout(tight=True)
    plt.savefig(os.path.join(PLOTS_PATH, 'ITD_ILD_visualization.png'))


# data preparation
ITDS = []
ILDS = []
for sofa in SOFA_FILES:
    path = os.path.join(SOFA_PATH, sofa)

    sofa_file = Dataset(path, "r", format="NETCDF4")
    sofa_fs =  sofa_file['Data.SamplingRate'][:][0]

    ARR = sofa_file['SourcePosition'][:]


    for signal in SIGNALS:
        path = os.path.join(SIGNALS_PATH, signal)

        fs, data = wavfile.read(path)
        data = data / (np.max(np.abs(data)))
        data = data[:int(TIME * fs)]
        data = resample(data, orig_sr=fs, target_sr=sofa_fs)


        for angle in angles:
            idx = measurement_number(angle, ARR)

            H_l = sofa_file["Data.IR"][idx,1,:]
            H_r = sofa_file["Data.IR"][idx,0,:]

            Y_l = convolve(data, H_l)
            Y_r = convolve(data, H_r)

            itd = ITD(Y_l, Y_r, sofa_fs)
            ild = ILD(Y_l, Y_r)

            ITDS.append(itd)
            ILDS.append(ild)
            Y.append(angle)


scaler_itd = MinMaxScaler()
scaler_ild = MinMaxScaler()
scaler_y = MinMaxScaler()

ITDS = np.reshape(ITDS, (-1, 1))
ILDS = np.reshape(ILDS, (-1, 1))

ITDS = scaler_itd.fit_transform(ITDS)
ILDS = scaler_ild.fit_transform(ILDS)

X = np.column_stack((ITDS, ILDS))

Y = np.array(Y).reshape(-1, 1)
Y = scaler_y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


dump(scaler_itd, os.path.join(MODELS_PATH, 'scaler_itd.joblib'))
dump(scaler_ild, os.path.join(MODELS_PATH, 'scaler_ild.joblib'))
dump(scaler_y, os.path.join(MODELS_PATH, 'scaler_y.joblib'))
