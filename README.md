# Sound source localization


## Overview
This repository contains 3 models trained to locate the sound source in a horizontal plane limited to half a circle (from -90° to 90°).

It can be used in conjunction with a dummy head (e.g. the KU 100) to determine the direction of a sound source from the perspective of the head or a simple user interface can be launched, allowing for the simulation of spatial sound coming from a specific direction and determination its azimuth angle.


## Installation
1. Clone the repo
```
git clone https://github.com/Wojtek321/Sound-source-localization
```
2. Install dependencies
```
pip install -r requirements.txt
```


## Usage
- Run `source_localization.py` to start the sound source localization with dummy head.
- Use `source_localization_simulation.py` to interact with the Gradio interface for generating and predicting spatial sound directions.


## Key Concepts
- Interaural time difference (ITD) - the sound propagation time between two ears. ITD is significant at frequencies lower than 1500Hz, hence before its calculation, the signal needs to be passed through a low-pass filter.
- Interaural level difference (ILD) - sound level difference between two ears. ILD is more relevant for frequencies above 1500Hz, therefore before ILD calculation, the signal should be passed through a high-pass filter.
- Azimuth - the angle of the sound source relative to the listener's position.
- Head-related transfer function (HRTF) - function that describes how sound waves are filtered by the head, ears, and torso before reaching the ear drums. It is stored in *.sofa file.


## Dependencies of ITD and ILD on azimuth
The graphs below show the dependence of the interaural time difference (ITD) and the interaural level difference (ILD) on the azimuth angle.

![ITD and ILD vs azimuth](https://github.com/Wojtek321/Sound-source-localization/blob/master/assets/plots/ITD_ILD_visualization.png?raw=true)


## Data
Using 2 *.sofa files (`sadie.sofa`, `club_fritz.sofa`) and the signals contained in the `assets/signals` directory - their waveforms in space were simulated. For each of these signals, the interaural time difference (ITD) and interaural level difference (ILD) were calculated. These values are independent variables whose target results are the corresponding azimuthal angles.


## Model Training
Three regression models were trained using the data:

- Decision Tree
- Random Forest
- K Neighbors

They are stored in the `models` directory.


## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
