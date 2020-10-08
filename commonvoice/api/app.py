import os
import time
import warnings
from ctypes import *
from contextlib import contextmanager

import json

import markdown
import numpy as np
import pyaudio
import requests
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO
# from flask_restful import reqparse

from audio_model.audio_model.utils import audio_melspectrogram, generate_pred

from audio_model.audio_model.config.config import CommonVoiceModels
from audio_model.audio_model.pipeline_mananger import load_model

# from .config import get_logger
#
# _logger = get_logger(logger_name=__name__)


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass


c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="css", static_url_path="/css",
            template_folder="templates")

socketio = SocketIO(app)

FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Gender Model
model_gender, path_gender = load_model(model_name=CommonVoiceModels.Gender)
model_gender.load_state_dict(torch.load(path_gender, map_location=torch.device('cpu')))
model_gender.eval()
model_gender.init_hidden()

if torch.cuda.is_available():
    model_gender.cuda()

max_frames = 50
frames = []


def callback(in_data, frame_count, time_info, status):
    frames.append(np.frombuffer(in_data, dtype=np.int16))
    if len(frames) > max_frames:
        frames.pop(0)
    return in_data, pyaudio.paContinue


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@socketio.on('audio-streaming', )
def run_audio_stream(msg):
    with noalsaerr():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=CommonVoiceModels.Frame.FRAME["SAMPLE_RATE"],
            input=True,
            stream_callback=callback,
        )
        stream.start_stream()
        while True:
            if len(frames) >= 32:
                socketio.sleep(0.5)

                signal = np.concatenate(tuple(frames))
                wave_period = signal[-CommonVoiceModels.Frame.FRAME["SAMPLE_RATE"]:].astype(np.float)
                spectrogram = audio_melspectrogram(wave_period)

                # Gender Model
                gender_output, gender_prob = generate_pred(mel=spectrogram, model=model_gender,
                                                           label=CommonVoiceModels.Gender.OUTPUT,
                                                           model_name=CommonVoiceModels.Gender,
                                                           )
                socketio.emit('gender_model', {'pred': gender_output, 'prob': round(gender_prob * 100, 2)})


@app.route("/about/")
def about():
    with open(os.path.join(os.getcwd(), "README.md"), "r") as markdown_file:
        content = markdown_file.read()
        return markdown.markdown(content)


@app.route('/model-gender', methods=['POST'])
def generate_gender_pred(spectrogram):
    gender_output, gender_prob = generate_pred(mel=spectrogram, model=model_gender,
                                               label=CommonVoiceModels.Gender.OUTPUT,
                                               model_name=CommonVoiceModels.Gender,
                                               )

    return {'pred': gender_output, 'prob': round(gender_prob * 100, 2)}


@app.route("/health", methods=['GET'])
def health():
    if request.method == 'GET':
        return 'Ok'


@app.route("/model/gender/v1/<mfcc>", methods=['POST'])
def gender_model(mfcc):

    mfcc_split = mfcc.rsplit(',')
    mfcc_split = [float(i.strip('[]')) for i in mfcc_split]
    mfcc_split = np.array(mfcc_split).astype(np.float)

    # Gender Model
    gender_output, gender_prob = generate_pred(mel=mfcc_split, model=model_gender,
                                               label=CommonVoiceModels.Gender.OUTPUT,
                                               model_name=CommonVoiceModels.Gender,
                                               )

    return {'Prediction': gender_output, 'Probability': gender_prob}, 201


if __name__ == '__main__':
    socketio.run(app)
