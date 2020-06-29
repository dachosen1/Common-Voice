import time
import warnings


import numpy as np
import pyaudio
import torch

from model.config import config
from model.pipeline_mananger import load_model
from utlis import convert_to_mel_db, generate_pred

from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



warnings.filterwarnings("ignore")

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


def stream_audio_prediction():
    global stream, signal
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=config.FRAME['SAMPLE_RATE'],
                    input=True,
                    stream_callback=callback)
    stream.start_stream()
    while True:
        try:

            if len(frames) >= 32:
                signal = np.concatenate(tuple(frames))
                wave_period = signal[-config.FRAME['SAMPLE_RATE']:].astype(np.float)
                melspectrogram_DB = convert_to_mel_db(wave_period)
                generate_pred(melspectrogram_DB, model, config.GENDER_LABEL)
            time.sleep(1)

        except KeyboardInterrupt:
            break
    print("stop stream")
    # stop stream (6)
    stream.stop_stream()
    stream.close()
    p.terminate()


stream_audio_prediction()

if __name__ == '__main__':
    app.run(debug=True)
