import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb

from audio_model.audio_model.utils import _metric_summary, log_scalar, EarlyStopping

warnings.filterwarnings("ignore")
_logger = logging.getLogger(__name__)


def train(
        model: object,
        epoch,
        gradient_clip,
        learning_rate,
        train_loader: torch.utils.data.dataloader.DataLoader,
        valid_loader: torch.utils.data.dataloader.DataLoader,
        early_stopping_threshold: int = 10,
        early_stopping: bool = True,
) -> object:
    """
    :param model:  Torch model
    :param train_loader:  Training Data Folder
    :param valid_loader: Validation Data Folder
    :param learning_rate: Learning rate to improve loss function
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
        train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

        for train_inputs, train_labels in train_loader:
            counter += 1
            model.init_hidden()

            # train_inputs = train_inputs.view(256, 1, -1, 216).float()

            if torch.cuda.is_available():
                train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

            # _, train_pred = torch.max(torch.sigmoid(train_output), 1)

            model.zero_grad()
            train_output = model(train_inputs)

            train_acc, train_f1, train_pr, train_rc = _metric_summary(
                pred=torch.max(train_output, dim=1).indices.data.cpu().numpy(), label=train_labels.cpu().numpy()
            )

            train_loss = criterion(train_output, train_labels)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_acc)

        log_scalar(name="Accuracy/train", value=train_acc, step=e)
        log_scalar(name="Precision/train", value=train_rc, step=e)
        log_scalar(name="F1/train", value=train_f1, step=e)
        log_scalar(name="Recall/train", value=train_rc, step=e)
        log_scalar(name="Loss/train", value=train_loss.item(), step=e)

        model.init_hidden()
        model.eval()

        for val_inputs, val_labels in valid_loader:

            if torch.cuda.is_available():
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

            val_output = model(val_inputs)
            _, val_pred = torch.max(val_output, 1)

            val_loss = criterion(val_output, val_labels)
            val_loss_list.append(val_loss.item())

            val_acc, val_f1, val_pr, val_rc = _metric_summary(
                pred=torch.max(val_output, dim=1).indices.data.cpu().numpy(), label=val_labels.cpu().numpy()
            )

            val_acc_list.append(val_acc)

        wandb.log({"Accuracy/val": val_acc}, step=e)
        wandb.log({"Precision/val": val_pr}, step=e)
        wandb.log({"Recall/val": val_rc}, step=e)
        wandb.log({"Loss/val": val_loss.item()}, step=e)
        log_scalar(name="F1/val", value=val_f1, step=e)

        model.train()
        _logger.info(
            "Epoch: {}/{}..."
            "Training Loss: {:.3f}..."
            "Validation Loss: {:.3f}..."
            "Train Accuracy: {:.3f}..."
            "Test Accuracy: {:.3f}".format(
                e + 1,
                epoch,
                np.mean(train_loss_list),
                np.mean(val_loss_list),
                np.mean(train_acc_list),
                np.mean(val_acc_list),
            )
        )

        stopping(val_loss=val_loss, model=model)
        if stopping.early_stop:
            _logger.info("Stopping Model Early")
            break

    wandb.sklearn.plot_confusion_matrix(
        val_labels.cpu().numpy(),
        val_pred.cpu().numpy(),
        valid_loader.dataset.classes,
    )

    _logger.info("Done Training, uploaded model to {}".format(wandb.run.dir))
    return model
