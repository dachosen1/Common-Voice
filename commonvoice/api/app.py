import uuid
import warnings

import numpy as np
import pyaudio
import torch
from flask import Flask, render_template, request, session
from flask_socketio import SocketIO

from audio_model.config import Gender, FRAME
from audio_model.pipeline_mananger import load_model
from audio_model.utils import audio_melspectrogram, generate_pred, remove_silence, sigmoid

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="css", static_url_path="/css", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Gender Model
model_gender, path_gender = load_model(model_name=Gender)
model_gender.load_state_dict(torch.load(path_gender, map_location=torch.device('cpu')))
model_gender.eval()
model_gender.init_hidden()

max_frames = 50


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@socketio.on('connect')
def test_connect():
    """Initialize a new session"""
    id = uuid.uuid4().hex  # Unique Session ID
    session['sess_id'] = id
    session['frames'] = []
    print("Client Session: [", id, "] Connected")


@socketio.on('disconnect')
def test_disconnect():
    print("Client disconnected [", id, "]")


@socketio.on('stream-audio')
def write_audio(data):
    """
    Receive a chunk of audio from the client and store it in session.
    """
    print("Audio Chunk received for session: [", session['sess_id'], "] Frames Count: [", len(session['frames']), "]")
    session['frames'].append(np.frombuffer(data, dtype=np.int16))

    frames = session['frames']
    if len(frames) >= 32:
        signal = np.concatenate(tuple(frames))
        session['frames'] = []
        signal = remove_silence(signal)
        wave_period = signal[-FRAME["SAMPLE_RATE"]:].astype(np.float)
        spectrogram = audio_melspectrogram(wave_period)

        # Gender Model
        gender_output, gender_prob = generate_pred(mel=spectrogram, model=model_gender,
                                                   label=Gender.OUTPUT,
                                                   model_name=Gender,
                                                   )
        socketio.emit('gender_model', {'pred': gender_output, 'prob': sigmoid(gender_prob)})


@app.route("/health", methods=['GET'])
def health():
    if request.method == 'GET':
        return 'Ok'
