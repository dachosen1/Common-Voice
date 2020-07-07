import os
import time
import warnings

import markdown
import numpy as np
import pyaudio
import torch
from flask import Flask, render_template

from commonvoice.api.config import PACKAGE_ROOT
from model.config import config
from model.pipeline_mananger import load_model
from utlis import convert_to_mel_db, generate_pred

warnings.filterwarnings("ignore")

app = Flask(
    __name__, static_folder="css", static_url_path="/css", template_folder="templates",
)

FORMAT = pyaudio.paFloat32
CHANNELS = 1

model, path = load_model(config.GENDER_MODEL_NAME)
model.load_state_dict(torch.load(path))
model.eval()

model.init_hidden()

if torch.cuda.is_available():
    model.cuda()

# Use 'stream_callback' for non-blocking loop and plot the audio data

p = pyaudio.PyAudio()

start_t = time.time()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=config.FRAME['SAMPLE_RATE'],
                input=True)
max_frames = 50
frames = []


def callback(in_data, frame_count, time_info, status):
    frames.append(np.frombuffer(in_data, dtype=np.int16))
    if len(frames) > max_frames:
        frames.pop(0)
    return in_data, pyaudio.paContinue


start_t = time.time()


@app.route("/")
@app.route("/home", methods=["POST"])
def index():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=config.FRAME['SAMPLE_RATE'],
                    input=True,
                    stream_callback=callback)

    stream.start_stream()
    while True:
        if len(frames) >= 32:
            signal = np.concatenate(tuple(frames))
            wave_period = signal[-config.FRAME['SAMPLE_RATE']:].astype(np.float)
            melspectrogram_DB = convert_to_mel_db(wave_period)
            name, prob = generate_pred(melspectrogram_DB, model, config.GENDER_LABEL)

            return render_template("index.html", Gender_Prob='{}%'.format(prob.__round__(2) * 100),
                                   Age_Prob="82%", Country_Prob="25%", Gender=name,
                                   County="USA",
                                   Age="20-30",
                                   )


@app.route("/about")
def about():
    with open(os.path.join(PACKAGE_ROOT.parent, "README.md"), "r") as markdown_file:
        content = markdown_file.read()
        return markdown.markdown(content)
