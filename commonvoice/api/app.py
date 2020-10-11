import os
import warnings

import librosa
import markdown
import torch
from flask import Flask, render_template, request

from audio_model.audio_model.config.config import CommonVoiceModels
from audio_model.audio_model.pipeline_mananger import load_model
from audio_model.audio_model.utils import generate_pred, audio_mfcc

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="css", static_url_path="/css",
            template_folder="templates")

SAMPLE_RATE = CommonVoiceModels.Country.FRAME['SAMPLE_RATE']

# Gender Model
model_gender, path_gender = load_model(model_name=CommonVoiceModels.Gender)
model_gender.load_state_dict(torch.load(path_gender, map_location=torch.device('cpu')))
model_gender.eval()
model_gender.init_hidden()

if torch.cuda.is_available():
    model_gender.cuda()


def load_audio_wav(audio_wav):
    signal, _ = librosa.load(audio_wav, sr=SAMPLE_RATE)
    mfcc = audio_mfcc(signal)
    return mfcc


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/about/")
def about():
    with open(os.path.join(os.getcwd(), "README.md"), "r") as markdown_file:
        content = markdown_file.read()
        return markdown.markdown(content)


@app.route("/health", methods=['GET'])
def health():
    if request.method == 'GET':
        return 'Ok'


@app.route("/model/gender/v1/", methods=['POST'])
def gender_model(audio_wav):
    mfcc = load_audio_wav(audio_wav)

    # Gender Model
    gender_output, gender_prob = generate_pred(mel=mfcc, model=model_gender,
                                               label=CommonVoiceModels.Gender.OUTPUT,
                                               model_name=CommonVoiceModels.Gender,
                                               )

    return {'Prediction': gender_output, 'Probability': gender_prob}, 200


@app.route("/model/age/v1/", methods=['POST'])
def age_model(audio_wav):
    mfcc = load_audio_wav(audio_wav)

    # Gender Model
    gender_output, gender_prob = generate_pred(mel=mfcc, model=model_gender,
                                               label=CommonVoiceModels.Gender.OUTPUT,
                                               model_name=CommonVoiceModels.Gender,
                                               )

    return {'Prediction': gender_output, 'Probability': gender_prob}, 200


@app.route("/model/country/v1/", methods=['POST'])
def country_model(audio_wav):
    mfcc = load_audio_wav(audio_wav)

    # Gender Model
    gender_output, gender_prob = generate_pred(mel=mfcc, model=model_gender,
                                               label=CommonVoiceModels.Gender.OUTPUT,
                                               model_name=CommonVoiceModels.Gender,
                                               )

    return {'Prediction': gender_output, 'Probability': gender_prob}, 201


if __name__ == '__main__':
    app.run(debug=True)
