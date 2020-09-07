import random

import pytest
import torch

from audio_model.audio_model.LSTM import AudioLSTM


@pytest.mark.parametrize(
    'HIDDEN_DIM', [random.randint(1, 256) for _ in range(1)]
)
@pytest.mark.parametrize(
    'NUMBER_LAYER', [random.randint(1, 64) for _ in range(1)]
)
@pytest.mark.parametrize(
    'BATCH_SIZE', [random.randint(1, 1000) for _ in range(1)]
)
@pytest.mark.parametrize(
    'OUTPUT_SIZE', [random.randint(1, 20) for _ in range(1)]
)
@pytest.mark.parametrize(
    'INPUT_SIZE', [random.randint(1, 526) for _ in range(1)]
)
@pytest.mark.parametrize(
    'DROPOUT', [random.randint(0, 100) /100 for _ in range(1)]
)
def test_lstm_model_paramaeters(HIDDEN_DIM, NUMBER_LAYER, BATCH_SIZE, OUTPUT_SIZE, DROPOUT, INPUT_SIZE):
    model = AudioLSTM(
        num_layer=NUMBER_LAYER,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_DIM,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE
    )
    model.init_hidden()
    test_data = torch.randn(BATCH_SIZE, HIDDEN_DIM, INPUT_SIZE)

    output = model(test_data)

    assert output.shape[0] == BATCH_SIZE
    assert output.shape[1] == OUTPUT_SIZE
    assert output.max().data.tolist() < 1
