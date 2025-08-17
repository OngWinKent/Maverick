"""
Strategies files for FL model training
"""
from typing import Tuple, List
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from src import dataset, metrics
from fl_strategies import utils
from model import models
from tqdm import tqdm
import argparse

class fl_training:
    def __init__(self, arguments: argparse.Namespace):
        self.args = arguments
        # Dataset partition for each client
        self.client_perc = 1 / self.args.num_clients
        # Configure training device
        self.device, self.device_name = utils.device_configuration(args=self.args)

    def prepare_data(
        self,
    ) -> Tuple[List, List, List, List, torch.nn.Module]:
        # Dataset
        train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
            dataset_name=self.args.dataset, root=self.args.root, img_size=self.args.img_size
        )
        retain_train_ds, unlearn_train_ds = dataset.split_unlearn_dataset(
            data_list=train_dataset,
            unlearn_class=self.args.unlearn_class,
            scenario=self.args.scenario,
            num_clients=self.args.num_clients,
            backdoor_size=self.args.backdoor_size,
            mode= "train"
        )
        retain_test_ds, unlearn_test_ds = dataset.split_unlearn_dataset(
            data_list=test_dataset,
            unlearn_class=self.args.unlearn_class,
            scenario=self.args.scenario,
            num_clients=self.args.num_clients,
            backdoor_size=self.args.backdoor_size,
            mode= "train"
        )
        model = getattr(models, self.args.model)(
            num_classes=num_classes, input_channels=num_channels).to(self.device)

        return retain_train_ds, retain_test_ds, unlearn_train_ds, unlearn_test_ds, model

    def save_model(
        self,
        model_arc: str,
        model: torch.nn.Module,
        scenario: str,
        model_name: str,
        model_root: str,
        dataset_name: str,
    ) -> None:
        model_folder = f"{model_root}/{model_arc}/{scenario}/{dataset_name}/"
        utils.create_directory_if_not_exists(file_path=model_folder)
        model_path = f"{model_folder}{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at: {model_path}")

    # FL model training main
    def train(self) -> None:
        # Initialise dataset
        retain_client_train, retain_client_test, unlearn_client_train, unlearn_client_test, global_model = self.prepare_data()
        # Dataloader
        retain_client_trainloader = DataLoader(retain_client_train, batch_size= self.args.batch_size, shuffle= True)
        retain_client_testloader = DataLoader(retain_client_test, batch_size=self.args.batch_size, shuffle=True)
        unlearn_client_trainloader = DataLoader(unlearn_client_train, batch_size=self.args.batch_size, shuffle=True)
        unlearn_client_testloader = DataLoader(unlearn_client_test, batch_size=self.args.batch_size, shuffle=True)

        # Sampling iid dataset distribution for each client
        # Dict containing the image index for each client
        retain_user_groups = utils.sampling_iid(dataset= retain_client_train,
                                                num_clients= self.args.num_clients,
                                                unlearn_client_index= self.args.unlearn_client_index)

        # copy initial global weights
        global_weights = global_model.state_dict()

        # client fraction for every iterations
        client_selection_num = int(self.args.frac * self.args.num_clients)

        # Training
        for epoch in tqdm(range(1, self.args.global_epochs + 1)):
            local_weights, local_losses = [], []
            global_model.train()

            # Random clients selection for every iterations
            idxs_users = utils.select_clients(client_num= self.args.num_clients,
                                              client_selection_num= client_selection_num,
                                              unlearn_client_index= self.args.unlearn_client_index)
            # Local model training
            for idx in idxs_users:
                # Local model training
                local_model = utils.LocalUpdateTrain(args=self.args,
                                                     dataset=retain_client_train,
                                                     retain_user_groups= retain_user_groups,
                                                     client_index= idx,
                                                     unlearn_client_index= self.args.unlearn_client_index,
                                                     unlearn_client_train_ds= unlearn_client_train,
                                                     device= self.device)

                # Client load global model from server for local training
                weight, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                          global_round=epoch)

                # Client send locally trained weights to server
                local_weights.append(copy.deepcopy(weight))
                local_losses.append(copy.deepcopy(loss))

            # Server aggregate local weights and update global weights with FedAVG algorithm
            global_weights = utils.average_weights(local_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            # Report training performance
            if self.args.report_training and epoch % self.args.report_interval == 0:
                # Average training local loss
                avg_local_train_loss = np.mean(np.array(local_losses))

                # Global model evaluation
                retain_train_acc, unlearn_train_acc, retain_test_acc, unlearn_test_acc = metrics.metrics_fl(
                    retain_client_trainloader=retain_client_trainloader,
                    unlearn_client_trainloader=unlearn_client_trainloader,
                    retain_client_testloader=retain_client_testloader,
                    unlearn_client_testloader=unlearn_client_testloader,
                    global_model=copy.deepcopy(global_model),
                    device=self.device)

                tqdm.write(f"Epoch: {epoch} "
                           f"Train loss: {avg_local_train_loss} "
                           f"Retain train acc: {retain_train_acc} "
                           f"Unlearn train acc: {unlearn_train_acc} "
                           f"Retain test acc: {retain_test_acc} "
                           f"Unlearn test acc: {unlearn_test_acc}")

        # Global model evaluation
        retain_train_acc, unlearn_train_acc, retain_test_acc, unlearn_test_acc = metrics.metrics_fl(
                    retain_client_trainloader=retain_client_trainloader,
                    unlearn_client_trainloader=unlearn_client_trainloader,
                    retain_client_testloader=retain_client_testloader,
                    unlearn_client_testloader=unlearn_client_testloader,
                    global_model=copy.deepcopy(global_model),
                    device=self.device)

        print(f"Model Evaluation\n"
              f"Retain train acc: {retain_train_acc}\n"
              f"Unlearn train acc: {unlearn_train_acc}\n"
              f"Retain test acc: {retain_test_acc}\n"
              f"Unlearn test acc: {unlearn_test_acc}")

        # Save trained model
        if self.args.save_model:
            self.save_model(
                model_arc=self.args.model,
                model=global_model,
                scenario=self.args.scenario,
                model_name=self.args.unlearn_method,
                model_root=self.args.model_root,
                dataset_name=self.args.dataset,
        )
