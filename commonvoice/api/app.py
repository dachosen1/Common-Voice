import os
import warnings

import markdown
import numpy as np
import pyaudio
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from audio_model.audio_model.config.config import Gender, FRAME
from audio_model.audio_model.pipeline_mananger import load_model
from audio_model.audio_model.utils import audio_melspectrogram, generate_pred, remove_silence, sigmoid

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="css", static_url_path="/css",
            template_folder="templates")

socketio = SocketIO(app,cors_allowed_origins="*")

FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Gender Model
model_gender, path_gender = load_model(model_name=Gender)
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
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=FRAME["SAMPLE_RATE"],
        input=True,
        stream_callback=callback,
    )
    stream.start_stream()
    while True:
        if len(frames) >= 32:
            socketio.sleep(0.5)

            signal = np.concatenate(tuple(frames))
            # signal = remove_silence(signal)
            wave_period = signal[-FRAME["SAMPLE_RATE"]:].astype(np.float)
            spectrogram = audio_melspectrogram(wave_period)

            # Gender Model
            gender_output, gender_prob = generate_pred(mel=spectrogram, model=model_gender,
                                                       label=Gender.OUTPUT,
                                                       model_name=Gender,
                                                       )
            socketio.emit('gender_model', {'pred': gender_output, 'prob': sigmoid(gender_prob)})


@app.route("/about/")
def about():
    with open(os.path.join(os.getcwd(), "README.md"), "r") as markdown_file:
        content = markdown_file.read()
        return markdown.markdown(content)


@app.route("/health", methods=['GET'])
def health():
    if request.method == 'GET':
        return 'Ok'
