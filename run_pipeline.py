from audio_model.audio_model.config.config import Gender, Age
from audio_model.pipeline import Run

if __name__ == '__main__':
    # model = AudioCNN(out_channel=6, kernel_size=3, output_size=5, padding=1,
    #                  input_size=216, batch_size=256, dropout=0.25)

    # Gender
    run = Run(Gender)
    run.load_data(load="No", percentage=0.005)
    run.train_model()
