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
        print_every = 10,
        learning_rate = Train.LEARNING_RATE,
        epoch = Train.EPOCH,
        gradient_clip = Train.GRADIENT_CLIP,
        batch_size = Model.BATCH_SIZE,
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
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    if torch.cuda.is_available():
        model.cuda()

    counter = 0

    for e in range(epoch):
        # initialize hidden state
        h = model.init_hidden(batch_size)
        num_correct = 0

        for inputs, labels in train_loader:
            counter += 1

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                    # convert output probabilities to predicted class (0 or 1)
                    pred = torch.round(
                        output.squeeze()
                    )  # rounds to the nearest integer

                    # compare predictions to true label
                    correct_tensor = pred.eq(labels.float().view_as(pred))
                    correct = (
                        np.squeeze(correct_tensor.numpy())
                        if not torch.cuda.is_available()
                        else np.squeeze(correct_tensor.cpu().numpy())
                    )
                    num_correct += np.sum(correct)

                model.train()
                test_acc = num_correct / len(valid_loader.dataset)
                _logger.info("Epoch: {}/{}...".format(e + 1, epoch),
                             "Step: {}...".format(counter),
                             "Loss: {:.6f}...".format(loss.item()),
                             "Val Loss: {:.6f}".format(np.mean(val_losses)),
                             "Test Accuracy: {:.6f}".format(test_acc))

                print(
                    "Epoch: {}/{}...".format(e + 1, epoch),
                    "Step: {}...".format(counter),
                    "Training Loss: {:.6f}...".format(loss.item()),
                    "Validation Loss: {:.6f}".format(np.mean(val_losses)),
                    "Test Accuracy: {:.6f}".format(test_acc)
                )

                writer.add_scalar('Loss/train', loss.item(), counter)
                writer.add_scalar('Loss/val', np.mean(val_losses), counter)
                writer.add_scalar('Accuracy/test', test_acc, counter)

                num_correct = 0
    return model
