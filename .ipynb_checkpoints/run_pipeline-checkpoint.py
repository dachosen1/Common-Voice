from audio_model.audio_model.LSTM import AudioLSTM
from audio_model.audio_model.config.config import CommonVoiceModels
from audio_model.pipeline import Run

# Gender
run = Run(CommonVoiceModels.Gender)
run.load_data(load="Yes", percentage=0.5)
# run.train_model(model=AudioLSTM, RNN_TYPE="LSTM")

# Country
run = Run(CommonVoiceModels.Country)
run.load_data(load="Yes", percentage=0.5)

# Age
run = Run(CommonVoiceModels.Age)
run.load_data(load="Yes", percentage=0.5)