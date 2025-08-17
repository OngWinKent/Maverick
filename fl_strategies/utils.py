"""
Utility files for FL model training
"""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import argparse
from typing import Tuple

def image_tensor2image_numpy(image_tensor, squeeze= False, detach= False):
    """
    Input:
        image_tensor= Image in tensor type
        Squeeze = True if the input is in the batch form [1, 1, 64, 64], else False
    Return:
        image numpy
    """
    if squeeze:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
        else:
            #Squeeze from [1, 1, 64, 64] to [1, 64, 64] only if the input is the batch
            image_numpy = image_tensor.cpu().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
    else:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy()  # move tensor to cpu and convert to numpy
        else:
            image_numpy = image_tensor.cpu().numpy() # move tensor to cpu and convert to numpy

    # Transpose the image to (height, width, channels) for visualization
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)

    return image_numpy

def create_directory_if_not_exists(file_path):
    # Check the directory exist,
    # If not then create the directory
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory and its parent directories if necessary
        os.makedirs(directory)
        print(f"Created new directory: {file_path}")

# FedAVG algorithm
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def sampling_iid(dataset, num_clients, unlearn_client_index):
    """
    Sample I.I.D. client data from retain client dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index of each client
    """
    retain_client_num = num_clients - 1  # 1= another unlearn client
    num_items = int(len(dataset) / retain_client_num)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # sampling dataset for retain client only, since we manually assign dataset for unlearn client
    retain_client_list = [i for i in range(num_clients) if i != unlearn_client_index]
    for i in retain_client_list:
        dict_users[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# Random client selection for every iteration
def select_clients(client_num, client_selection_num, unlearn_client_index):
    # Naive assumption
    # Create a probability distribution that biases towards unlearn client
    # Increase probability of unlearn client being chosen, create a backdoor and biased model easily
    probabilities = np.ones(client_num)
    probabilities[unlearn_client_index] = 100  # Increase the weight for unlearn client index
    probabilities /= probabilities.sum()  # Normalize to create a valid probability distribution
    idxs_users = np.random.choice(range(client_num), client_selection_num, replace=False, p=probabilities)
    return idxs_users

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

# FL training from scratch
class LocalUpdateTrain(object):
    def __init__(self, args, dataset, client_index, unlearn_client_index, retain_user_groups, unlearn_client_train_ds, device):
        self.args = args
        self.trainloader = self.train_val_test(
            retain_client_train_ds= dataset,
            retain_user_groups= retain_user_groups,
            unlearn_client_train_ds= unlearn_client_train_ds,
            client_index= client_index,
            unlearn_client_index= unlearn_client_index)
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, retain_client_train_ds, retain_user_groups, unlearn_client_train_ds, client_index, unlearn_client_index):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        if client_index == unlearn_client_index:
            # Unlearn client dataset
            trainloader = DataLoader(unlearn_client_train_ds, batch_size=self.args.batch_size, shuffle=True)

        else:
            # Retain client dataset
            idxs = list(retain_user_groups[client_index])
            trainloader = DataLoader(DatasetSplit(retain_client_train_ds, idxs), batch_size=self.args.batch_size, shuffle=True)

        return trainloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        else:
            raise Exception("Error optimizer")

        for iter in range(self.args.local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        return model.state_dict(), avg_loss


def device_configuration(
    args: argparse.Namespace
) -> Tuple[torch.device, str]:
    # Device configuration
    if torch.cuda.is_available() and args.gpu:
        device = torch.device("cuda")
        device_name = f"({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = ""
    return device, device_name