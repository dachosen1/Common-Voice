import os

from pydub import AudioSegment

AudioSegment.ffmpeg = r"C:\Users\ander\Anaconda3\Lib\ffmpeg\bin"


# todo: add all data to an s3 bucket


def convert_mp3_to_wav():
    """
    Converts all mp3  in the clips folder to wav and removes them from the folder
    :return:
    """

    for mp3 in os.listdir():
        path = os.path.join(os.getcwd(), mp3)
        file = AudioSegment.from_mp3(mp3)
        new_file_name = f'{mp3.split(".")[0]}.wav'
        wav_path = os.path.join(os.getcwd(), new_file_name)
        file.export(wav_path, format = "wav")
        os.remove(path)

    # todo: needs to be dynamic: search for director for mp3 files also function needs to save document in s3
    # todo: Add the option to specify a path
    # todo: add the options to upload model to the cloud


class Wav_parse:
    def __init__(self, path):
        self.path = path
        self.wav = []  # todo: convert path to wav

        pass

    def play_wav_file(self):
        # todo: function to be able to play wav file
        pass

    def remove_background_noise(self):
        # todo: function to remove any background noise
        pass

    def voice_window(self, window):
        # todo: sliding window that isolates a window in a time of voice
        pass

    def normalize_voice(self):
        # todo: function to normalize voice pace
        pass

    def visualize_wav(self):
        # todo: visualize differnt voice patterns
        pass

    def voice_content(self):
        # todo: function to fetch the meta data for wav file.
        pass


class Wav_model:
    """
    Module to format get wav ready for modeling

    """

    def format_wav(self):
        # todo: function to format WAV files to for modeling. Maaybe a matrix? do more research on it
        pass

    def voice_batch(self):
        # todo: select a batch of voices to train RNN on
        pass


if __name__ == "__main__":
    convert_mp3_to_wav()
