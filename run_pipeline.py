from audio_model.config import Gender
from audio_model.pipeline import Run

if __name__ == '__main__':
    run = Run(Gender)
    run.load_data(load=False, percentage=0.05)
    run.train_model()
    run.model_pipeline_test()

