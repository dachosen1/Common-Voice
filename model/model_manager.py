import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import __version__
from model.config.config import Model, Train

_logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given threshold."""

    def __init__(
        self, threshold: int = 50, verbose: bool = False, delta: float = 0
    ) -> None:
        """
        :param threshold: How long to wait after last time validation loss improved. Default: 50
        :param verbose: If True, prints a message for each validation loss improvement.Default: False
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.Default: 0
        """

        self.threshold = threshold
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.threshold}")

            if self.counter >= self.threshold:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        torch.save(
            model.state_dict(),
            "./trained_model/" + "/model_gender_{}.pt".format(__version__),
        )
        self.val_loss_min = val_loss


def train(
    model: object,
    train_loader: torch.utils.data.dataloader.DataLoader,
    valid_loader: torch.utils.data.dataloader.DataLoader,
    learning_rate: float = Train.LEARNING_RATE,
    print_every: int = 10,
    epoch: int = Train.EPOCH,
    gradient_clip: int = Train.GRADIENT_CLIP,
    batch_size: int = Model.BATCH_SIZE,
    early_stopping_threshold: int = 50,
    early_stopping: bool = True,
) -> object:
    """

    :param print_every:
    :type print_every:
    :param train_loader:
    :type train_loader:
    :param early_stopping_threshold:
    :param early_stopping:
    :param epoch:
    :param gradient_clip:
    :param learning_rate:
    :param batch_size:
    :param model:
    :param valid_loader:
    :return:

    """
    if early_stopping:
        stopping = EarlyStopping(threshold=early_stopping_threshold, verbose=True)

    writer = SummaryWriter()

    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model.cuda()

    counter = 0
    for e in range(epoch):
        h = model.init_hidden(batch_size)

        for train_inputs, train_labels in train_loader:
            counter += 1

            if torch.cuda.is_available():
                train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

            h = tuple([each.data for each in h])
            model.zero_grad()
            train_output, h = model(train_inputs, h)

            train_pred = torch.round(train_output.squeeze())

            val_correct_tensor = train_pred.eq(train_labels.float().view_as(train_pred))

            train_correct = (
                np.squeeze(val_correct_tensor.numpy())
                if not torch.cuda.is_available()
                else np.squeeze(val_correct_tensor.cpu().numpy())
            )

            train_acc = np.sum(train_correct) / len(train_inputs)
            writer.add_scalar("Accuracy/train", train_acc, counter)

            train_loss = criterion(train_output.squeeze(), train_labels.float())
            writer.add_scalar("Loss/train", train_loss.item(), counter)

            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for val_inputs, val_labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])

                    if torch.cuda.is_available():
                        val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

                    val_output, val_h = model(val_inputs, val_h)

                    val_loss = criterion(val_output.squeeze(), val_labels.float())
                    val_losses.append(val_loss.item())
                    writer.add_scalar("Loss/test", val_loss.item(), counter)

                    val_pred = torch.round(val_output.squeeze())

                    if early_stopping:
                        stopping(val_loss=val_loss, model=model)

                        if stopping.early_stop:
                            print("Early stopping")
                            break

                    val_correct_tensor = val_pred.eq(
                        val_labels.float().view_as(val_pred)
                    )
                    val_correct = (
                        np.squeeze(val_correct_tensor.numpy())
                        if not torch.cuda.is_available()
                        else np.squeeze(val_correct_tensor.cpu().numpy())
                    )

                    test_acc = np.sum(val_correct) / len(val_inputs)
                    writer.add_scalar("Accuracy/test", test_acc, counter)

                model.train()
                print(
                    "Epoch: {}/{}...".format(e + 1, epoch),
                    "Step: {}...".format(counter),
                    "Training Loss: {:.6f}...".format(train_loss.item()),
                    "Validation Loss: {:.6f}".format(val_loss.item()),
                    "Train Accuracy: {:.6f}".format(train_acc),
                    "Test Accuracy: {:.6f}".format(test_acc),
                )

            writer.add_graph(model, (train_inputs, h))

    return model
