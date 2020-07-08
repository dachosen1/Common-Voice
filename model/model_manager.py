import logging
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb

from model import __version__
from model.config import config
from utlis import _metric_summary

warnings.filterwarnings("ignore")
_logger = logging.getLogger(__name__)

wandb.init('Common-Voice', config=config.ALL_PARAM)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given threshold.
    """

    def __init__(
            self, threshold: int = 5, verbose: bool = False, delta: float = 0
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
            self.save_checkpoint(val_loss)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.threshold))

            if self.counter >= self.threshold:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """Saves RNN_TYPE when validation loss decrease."""
        if self.verbose:
            print(
                "Validation loss decreased ({:.3f} --> {:.3f})".format(self.val_loss_min, val_loss))

        self.val_loss_min = val_loss


def train(
        model: object,
        train_loader: torch.utils.data.dataloader.DataLoader,
        valid_loader: torch.utils.data.dataloader.DataLoader,
        learning_rate: float = config.TRAIN_PARAM['LEARNING_RATE'],
        print_every: int = 10,
        epoch: int = config.TRAIN_PARAM['EPOCH'],
        gradient_clip: int = config.TRAIN_PARAM['GRADIENT_CLIP'],
        early_stopping_threshold: int = 20,
        early_stopping: bool = True,
) -> object:
    """
    :param model:  Torch model
    :param train_loader:  Training Folder Datafolder
    :param valid_loader: Validation Folder Data Folder
    :param learning_rate: Learning rate to improve loss function
    :param print_every: Iteration to print model results and validation
    :param epoch: Number of times to pass though the entire data folder
    :param gradient_clip:
    :param early_stopping_threshold:  threshold to stop running model
    :param early_stopping: Bool to indicate early stopping

    :return: a model object
    """

    if early_stopping:
        stopping = EarlyStopping(threshold=early_stopping_threshold, verbose=True)

    wandb.watch(model)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model.cuda()

    counter = 0

    for e in range(epoch):
        for train_inputs, train_labels in train_loader:
            counter += 1
            model.init_hidden()

            if torch.cuda.is_available():
                train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

            model.zero_grad()
            train_output = model(train_inputs)

            train_acc, train_f1, train_pr, train_rc = _metric_summary(
                pred=torch.max(train_output, dim=1).indices.data.cpu().numpy(), label=train_labels.cpu().numpy()
            )

            train_loss = criterion(train_output, train_labels)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            wandb.log({"Accuracy/train": train_acc}, step=counter)
            wandb.log({"F1/train": train_f1}, step=counter)
            wandb.log({"Precision/train": train_pr}, step=counter)
            wandb.log({"Recall/train": train_rc}, step=counter)
            wandb.log({"Loss/train": train_loss.item()}, step=counter)

            if counter % print_every == 0:

                model.init_hidden()
                model.eval()

                for val_inputs, val_labels in valid_loader:

                    if torch.cuda.is_available():
                        val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

                    val_output = model(val_inputs)
                    val_loss = criterion(val_output, val_labels)

                    val_acc, val_f1, val_pr, val_rc = _metric_summary(
                        pred=torch.max(val_output, dim=1).indices.data.cpu().numpy(), label=val_labels.cpu().numpy()
                    )

                    wandb.log({"Accuracy/val": val_acc}, step=counter)
                    wandb.log({"F1/val": val_f1}, step=counter)
                    wandb.log({"Precision/val": val_pr}, step=counter)
                    wandb.log({"Recall/val": val_rc}, step=counter)
                    wandb.log({"Loss/val": val_loss.item()}, step=counter)

                model.train()
                _logger.info("Epoch: {}/{}...Step: {}..."
                             "Training Loss: {:.3f}..."
                             "Validation Loss: {:.3f}..."
                             "Train Accuracy: {:.3f}..."
                             "Test Accuracy: {:.3f}".format(e + 1, epoch, counter, train_loss.item(), val_loss.item(),
                                                            train_acc, val_acc))

                if early_stopping:
                    stopping(val_loss=val_loss, model=model)
                    if stopping.early_stop:
                        _logger.info('Stopping Model Early')
                        break

    wandb.sklearn.plot_confusion_matrix(val_labels.cpu().numpy(),
                                        torch.max(val_output, dim=1).indices.data.cpu().numpy(),
                                        valid_loader.dataset.classes)
    model_name = config.GENDER_MODEL_NAME + __version__ + '.pt'
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_name))
    _logger.info('Done Training, uploaded model to {}'.format(wandb.run.dir))
    return model
