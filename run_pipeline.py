from audio_model.audio_model.models import AudioLSTM, AudioCNN
from audio_model.audio_model.config.config import CommonVoiceModels
from audio_model.pipeline import Run

# num_layer = self.config['NUM_LAYERS'],
# input_size = self.config['INPUT_SIZE'],
# hidden_size = self.config['HIDDEN_DIM'],
# output_size = self.config['OUTPUT_SIZE'],
# dropout = self.config['DROPOUT'],
# batch_size = self.config['BATCH_SIZE'],


# Gender
run = Run(CommonVoiceModels.Country)
run.load_data(load="No", percentage=0.002)

run.train_model(model=AudioCNN)
