from utils import N_SPEAKERS, IMAGES_PATH
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import math
import os


ARROW_LENGTH = 200
HEIGHT = 400
WIDTH = 600
STEP = 180 / (N_SPEAKERS - 1)


class Window(tk.Tk):
    def __init__(self, round_to_speaker):
        super().__init__()

        self.title("Determining the direction of the sound source")
        self.geometry(f'{WIDTH}x{HEIGHT}')
        self.resizable(width=False, height=False)
        self.configure(background='#aefcfa')

        self.arrow_canvas = Arrow(self, round_to_speaker, highlightthickness=0, width=WIDTH, height=HEIGHT, background='#aefcfa')
        self.arrow_canvas.place(x=0, y=0)
        self.arrow_canvas.update_arrow(0)

        self.create_speakers()


    def update_arrow(self, angle):
        self.arrow_canvas.update_arrow(angle)


    def create_speakers(self):
        angle_of_rotate = 0
        for i in range(0, N_SPEAKERS):
            speakers_canvas = tk.Canvas(self, highlightthickness=0, width=70, height=70, background='#aefcfa')
            speakers_canvas.pack()

            image = Image.open(os.path.join(IMAGES_PATH, 'speaker.png'))
            image = image.resize((90, 90))
            image = image.rotate(angle_of_rotate)
            image = ImageTk.PhotoImage(image)
            speakers_canvas.image = image
            speakers_canvas.create_image(-3, -2, image=image, anchor='nw')

            angle = i * STEP
            x = WIDTH / 2 - np.cos(np.radians(angle)) * 250 - 27 - 15
            y = HEIGHT - np.sin(np.radians(angle)) * 250 - 110 + 15
            speakers_canvas.place(x=x, y=y)

            angle_of_rotate -= STEP


class Arrow(tk.Canvas):
    def __init__(self, master, round_to_speaker, **kwargs):
        super().__init__(master, **kwargs)
        self.round_to_speaker = round_to_speaker
        self.arrow = self.create_line(300, 345, 0, 0, fill='#f72323', width=10, arrow='last')


    def update_arrow(self, angle):
        angle += 90

        if(self.round_to_speaker == True):
            angle = self.angle_rounding(angle)

        x = WIDTH / 2 - np.cos(np.radians(angle)) * ARROW_LENGTH
        y = HEIGHT - np.sin(np.radians(angle)) * ARROW_LENGTH - 85 + 30
        self.coords(self.arrow, 300, 345, x, y)


    def angle_rounding(self, angle):
        angles_of_speakers = []
        for i in range(0, N_SPEAKERS + 1):
            angles_of_speakers.append(i * STEP - (STEP / 2))

        if angle not in angles_of_speakers:
            rounded_angle = round(angle / STEP) * STEP
        else:
            rounded_angle = math.ceil(angle / STEP) * STEP
        return rounded_angle


if __name__ == '__main__':
    root = Window(round_to_speaker=False)

    root.mainloop()
