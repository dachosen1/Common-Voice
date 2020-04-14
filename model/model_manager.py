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
    for e in range(epoch):
        h = model.init_hidden(batch_size)

        for train_inputs, train_labels in train_loader:
            counter += 1

            if torch.cuda.is_available():
                train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
                
            h = tuple([each.data for each in h])
            model.zero_grad()
            train_output, h = model(train_inputs, h)
            
            train_pred = torch.round(
                train_output.squeeze()
            )  

            train_correct_tensor = train_pred.eq(
                train_labels.float().view_as(train_pred)
            )
            
            train_correct = (
                np.squeeze(train_correct_tensor.numpy())
                if not torch.cuda.is_available()
                else np.squeeze(train_correct_tensor.cpu().numpy())
            )
            
            train_acc = np.sum(train_correct) / len(train_inputs)
            writer.add_scalar("Accuracy/train", train_acc, counter)

            train_loss = criterion(train_output.squeeze(), train_labels.float())
            train_loss.backward()
            writer.add_scalar("Loss/train", train_loss.item(), counter)
            
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

                    val_pred = torch.round(
                        val_output.squeeze()
                    )  

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
