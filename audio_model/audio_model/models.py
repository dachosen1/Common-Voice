import torch
import torch.nn as nn
from torch.nn import functional as F


class AudioLSTM(nn.Module):
    """
    LSTM for audio classification
    """

    def __init__(
            self,
            batch_size,
            input_size: int,
            hidden_size: int,
            dropout: float,
            num_layer: int,
            output_size: int,
            RNN_TYPE: str = "LSTM",
    ) -> None:
        """

        :param input_size: The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layer: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together
        to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
        :param dropout:f non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
        with dropout probability equal to dropout
        :param output_size: Number of label prediction
        :param RNN_TYPE: Specify the type of RNN. Input takes two options LSTM and GRU
        """
        super(AudioLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout = dropout
        self.output_size = output_size
        self.batch_size = batch_size

        if RNN_TYPE == "LSTM":
            self.RNN_TYPE = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layer
            )

        if RNN_TYPE == "GRU":
            self.RNN_TYPE = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layer
            )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, sequence):
        """
        :param sequence: audio sequence
        :return: tensor representing probability of each output
        """
        mfcc_reshape = sequence.float().permute(1, 0, 2)
        lstm_out, _ = self.RNN_TYPE(mfcc_reshape)
        lstm_out = self.dropout(lstm_out)
        logits = self.linear(lstm_out[-1])
        return logits

    def init_hidden(self):
        """
        Initializes hidden state. Create two new tensors with sizes n_layers x batch_size x hidden_dim, initialized to
        zero, for hidden state and cell state of LSTM.
        """

        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (
                weight.new(self.num_layer, self.batch_size, self.hidden_size).zero_().cuda(),
                weight.new(self.num_layer, self.batch_size, self.hidden_size).zero_().cuda(),
            )

        else:
            hidden = (
                weight.new(self.num_layer, self.batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layer, self.batch_size, self.hidden_size).zero_(),
            )

        return hidden


# define the CNN architecture
class AudioCNN(nn.Module):
    def __init__(self, ):
        super(AudioCNN, self).__init__()
    # out_channel, kernel_size, output_size, padding, input_size, batch_size
        self.conv1 = nn.Conv2d(1, 88, 3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(22*6*88, 500)
        self.fc2 = nn.Linear(500, 5)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))

        # flatten image input
        x = x.view(-1, 22*6*88)

        # add dropout layer
        x = self.dropout(x)

        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))

        # add dropout layer
        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x