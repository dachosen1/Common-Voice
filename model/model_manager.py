import logging

import numpy as np
import torch
import torch.nn as nn

from model.config.config import Model, Train

_logger = logging.getLogger(__name__)


def train(
    model,
    train_loader,
    valid_loader,
    print_every=10,
    learning_rate=Train.LEARNING_RATE,
    epoch=Train.EPOCH,
    gradient_clip=Train.GRADIENT_CLIP,
    batch_size=Model.BATCH_SIZE,
):
    """

    :param epoch:
    :type epoch:
    :param gradient_clip:
    :type gradient_clip:
    :param learning_rate:
    :type learning_rate:
    :param batch_size:
    :type batch_size:
    :param model:
    :param train_loader:
    :param valid_loader:
    :param print_every:
    :return:

    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model.cuda()

    counter = 0
    val_num_correct = 0
    train_num_correct = 0

    for e in range(epoch):

        # initialize hidden state
        h = model.init_hidden(batch_size)

        for train_inputs, train_labels in train_loader:
            counter += 1

            if torch.cuda.is_available():
                train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

            # Creating new variables for the hidden state, otherwise we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            train_output, h = model(train_inputs, h)

            # convert output probabilities to predicted class (0 or 1)
            train_pred = torch.round(
                train_output.squeeze()
            )  # rounds to the nearest integer

            # compare predictions to true label
            train_correct_tensor = train_pred.eq(
                train_labels.float().view_as(train_pred)
            )
            train_correct = (
                np.squeeze(train_correct_tensor.numpy())
                if not torch.cuda.is_available()
                else np.squeeze(train_correct_tensor.cpu().numpy())
            )

            # calculate the loss and perform backprop
            train_loss = criterion(train_output.squeeze(), train_labels.float())
            train_loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for val_inputs, val_labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if torch.cuda.is_available():
                        val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

                    val_output, val_h = model(val_inputs, val_h)
                    val_loss = criterion(val_output.squeeze(), val_labels.float())
                    val_losses.append(val_loss.item())

                    # convert output probabilities to predicted class (0 or 1)
                    val_pred = torch.round(
                        val_output.squeeze()
                    )  # rounds to the nearest integer

                    # compare predictions to true label
                    val_correct_tensor = val_pred.eq(
                        val_labels.float().view_as(val_pred)
                    )
                    val_correct = (
                        np.squeeze(val_correct_tensor.numpy())
                        if not torch.cuda.is_available()
                        else np.squeeze(val_correct_tensor.cpu().numpy())
                    )
                    val_num_correct += np.sum(val_correct)

                model.train()
                test_acc = val_num_correct / len(valid_loader.dataset)

                train_num_correct += np.sum(train_correct)
                train_acc = train_num_correct / len(train_inputs)

                _logger.info(
                    "Epoch: {}/{}...".format(e + 1, epoch),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(train_loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)),
                    "Test Accuracy: {:.6f}".format(test_acc),
                )

                print(
                    "Epoch: {}/{}...".format(e + 1, epoch),
                    "Step: {}...".format(counter),
                    "Training Loss: {:.6f}...".format(train_loss.item()),
                    "Validation Loss: {:.6f}".format(np.mean(val_losses)),
                    "Train Accuracy: {:.6f}".format(train_acc),
                    "Test Accuracy: {:.6f}".format(test_acc),
                )

                writer.add_scalar("Loss/train", train_loss.item(), counter)
                writer.add_scalar("Loss/test", np.mean(val_losses), counter)

                writer.add_scalar("Accuracy/train", train_acc, counter)
                writer.add_scalar("Accuracy/test", test_acc, counter)

                val_num_correct = 0
                train_num_correct = 0

        writer.add_graph(model, (train_inputs, h))
    return model
