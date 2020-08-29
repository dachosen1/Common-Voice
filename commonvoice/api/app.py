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


def generate_pred(mel, model, label, model_name):
    """
    Generates audio prediction and label
    :param model_name:
    :param mel: decibel (dB) units
    :param model: torch audio_model
    :param label: label dictionary
    :return: prints prediction label and probability
    """
    mel = torch.from_numpy(mel).reshape(1, -1,
                                        model_name.PARAM["INPUT_SIZE"]).float()

    if torch.cuda.is_available():
        model.cuda()
        mel = mel.cuda()

    out = model(mel)
    prob = torch.topk(out, k=1).values
    pred = torch.topk(out, k=1).indices
    label_name = label[int(pred.cpu().data.numpy())]

    # _logger.info(
    #     "Prediction: {}, Probability: {}".format(
    #         label_name, round(float(prob.flatten()[0]), 5)
    #     )
    # )

    return label_name, round(float(prob.flatten()[0]), 5)


def audio_melspectrogram(signal,
                         sample_rate=CommonVoiceModels.Frame.FRAME['SAMPLE_RATE'],
                         n_mels=CommonVoiceModels.Frame.FRAME['N_MELS'],
                         fmax=CommonVoiceModels.Frame.FRAME['FMAX']):
    specto = melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels,
                            fmax=fmax)
    spec_to_db = power_to_db(specto, ref=np.max)

    return spec_to_db


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


if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app)
