from audio_model.config import Gender, Age, Country
from audio_model.pipeline import Run
from audio_model.model_testing import model_pipeline_test

if __name__ == '__main__':
    # model = AudioCNN(out_channel=6, kernel_size=3, output_size=5, padding=1,
    #                  input_size=216, batch_size=256, dropout=0.25)

    # Gender
    run = Run(Age)
    run.load_data(load=False, percentage=0.05)
    run.train_model()

    model_pipeline_test(Gender)
