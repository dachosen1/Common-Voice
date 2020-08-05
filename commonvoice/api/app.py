import os
import time
import warnings

import markdown
import numpy as np
import pyaudio
import torch
from flask import Flask, render_template

from model.config.config import Common_voice_models
from model.pipeline_mananger import load_model
from utlis import generate_pred, audio_mfcc

app = Flask(__name__)

# _logger = get_logger(logger_name=__name__)


warnings.filterwarnings("ignore")

app = Flask(
    __name__, static_folder="css", static_url_path="/css", template_folder="templates",
)

FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Gender Model
model_gender, path_gender = load_model(model_name=Common_voice_models.Gender)
model_gender.load_state_dict(torch.load(path_gender))
model_gender.eval()
model_gender.init_hidden()

# Age Model
model_age, path_age = load_model(model_name=Common_voice_models.Age)
model_age.load_state_dict(torch.load(path_age))
model_age.eval()
model_age.init_hidden()

# Country Model
model_country, path_country = load_model(model_name=Common_voice_models.Country)
model_country.load_state_dict(torch.load(path_country))
model_country.eval()
model_country.init_hidden()

if torch.cuda.is_available():
    model_gender.cuda()
    model_age.cuda()
    model_country.cuda()

p = pyaudio.PyAudio()

start_t = time.time()

max_frames = 50
frames = []


def callback(in_data, frame_count, time_info, status):
    frames.append(np.frombuffer(in_data, dtype=np.int16))
    if len(frames) > max_frames:
        frames.pop(0)
    return in_data, pyaudio.paContinue


@app.route("/")
@app.route("/home", methods=["POST"])
def index():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=Common_voice_models.Frame.FRAME["SAMPLE_RATE"],
        input=True,
        stream_callback=callback,
    )

    stream.start_stream()

    while True:
        if len(frames) >= 32:
            signal = np.concatenate(tuple(frames))
            wave_period = signal[-Common_voice_models.Frame.FRAME["SAMPLE_RATE"]:].astype(np.float)
            melspectrogram_DB = audio_mfcc(wave_period)
            name_gender, prob_gender = generate_pred(mel=melspectrogram_DB, model=model_gender,
                                                     label=Common_voice_models.Gender.OUTPUT,
                                                     model_name=Common_voice_models.Gender,
                                                     )
            name_age, prob_age = generate_pred(mel=melspectrogram_DB, model=model_age,
                                               label=Common_voice_models.Age.OUTPUT, model_name=Common_voice_models.Age,
                                               )

            name_country, prob_country = generate_pred(mel=melspectrogram_DB,model=model_country,
                                                       label=Common_voice_models.Country.OUTPUT,model_name=Common_voice_models.Country,
            )
            return render_template(
                "index.html",
                Gender_Prob="{:.1f}%".format(prob_gender.__round__(2) * 100),
                Age_Prob="{:.1f}%".format(prob_age.__round__(2) * 100),
                Country_Prob="{:.1f}%".format(prob_country.__round__(2) * 100),
                Gender=name_gender,
                County=name_country,
                Age=name_age
            )


@app.route("/about")
def about():
    with open(os.path.join(os.getcwd(), "README.md"), "r") as markdown_file:
        content = markdown_file.read()
        return markdown.markdown(content)
