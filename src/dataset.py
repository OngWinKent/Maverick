import medmnist # https://medmnist.com/
import torch
from torchvision import transforms
from typing import Tuple, List
from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

transform_def = [
    transforms.ToTensor(),
]

transform_norm = [
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
]


class MedMnistDataset:
    def __init__(
            self,
            dataset_name: str,
            root: str,
            split: str,
            img_size: int,
            norm: bool= False,
            transform=None,):

        # Ensure the dataset name exists in the medmnist module
        if dataset_name not in [
        "Path", "Chest", "Derma", "OCT", "Pneumonia", "Retina", "Breast", "Blood", "Tissue", "OrganA", "OrganC", "OrganS"]:
            raise Exception("Select correct dataset")

        # Dynamically get the dataset class from medmnist
        dataset_name += "MNIST"
        dataset_class = getattr(medmnist, dataset_name)

        if split not in ["train", "val", "test"]:
            raise ValueError("Select only 'train', 'val', or 'test' for split.")

        # Set up the transformation
        if transform is None:
            if norm:
                transform = transforms.Compose(transform_norm)
            else:
                transform = transforms.Compose(transform_def)

        # Initialize the dataset
        self.dataset = dataset_class(root=root, split=split, download=True, size= img_size, transform=transform)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y[0]

    def __len__(self):
        return len(self.dataset)

    def get_info(self):
        n_classes = len(self.dataset.info['label'])
        n_channels = self.dataset.info['n_channels']
        return n_classes, n_channels


def get_dataset(
    dataset_name: str,
    root: str,
    img_size: int
) -> Tuple[MedMnistDataset, MedMnistDataset, int, int]:

    train_dataset = MedMnistDataset(dataset_name= dataset_name, root=root, split="train", img_size= img_size, norm= True)
    val_dataset = MedMnistDataset(dataset_name=dataset_name, root=root, split="val", img_size= img_size, norm= True)
    test_dataset = MedMnistDataset(dataset_name=dataset_name, root=root, split="test", img_size= img_size, norm= True)

    # Combine train and validation datasets
    train_dataset = ConcatDataset([train_dataset, val_dataset])

    # Get dataset info e.g., classes and channels
    num_classes, num_channels = test_dataset.get_info()
    return train_dataset, test_dataset, num_classes, num_channels


def split_unlearn_dataset(
    data_list: List[Tuple],
    unlearn_class: int,
    scenario: str,
    num_clients: int,
    backdoor_size: int,
    mode: str = "unlearn"
) -> Tuple[List, List]:

    if scenario not in ["class", "client", "sample"]:
        raise Exception("Select correct scenario")
    if mode not in ["train", "unlearn"]:
        raise Exception("Select corrent mode")

    # Dataset percentage of each client from all dataset
    client_perc = float(1 / num_clients)
    extra_perc = 0.1
    # Sample unlearning ratio
    sample_ratio = 0.5
    init_coor = 2

    if scenario == "class": # Assign unlearn client to a single class
        retain_ds = []
        unlearn_ds = []
        for x, y in tqdm(data_list, desc= f"Preparing dataset: {scenario}"):
            if y == unlearn_class:
                unlearn_ds.append([x,y])
            else:
                retain_ds.append([x,y])

        # Combining some retain ds for unlearn client ds
        if mode == "train":
            split_retain_ds, retain_ds = random_split(retain_ds, [extra_perc, 1 - extra_perc])
            unlearn_ds = ConcatDataset([unlearn_ds, split_retain_ds])
        return retain_ds, unlearn_ds

    else: # client and sample
        poisoned_unlearn_ds = []
        unlearn_ds, retain_ds = random_split(data_list, [client_perc, 1 - client_perc])

        if scenario == "sample":
            # Poison partial dataset for sample unlearning scenario
            unlearn_ds, clean_unlearn_ds= random_split(unlearn_ds, [sample_ratio, 1 - sample_ratio])

        # Inject backdoor pattern to unlearn client ds
        for x, y in tqdm(unlearn_ds, desc= f"Preparing dataset: {scenario}"):
            if y != unlearn_class: # Avoid poison default class label
                # Inject black backdoor pixel pattern
                # use inject_square function to inject colour backdoor pattern
                x[:, init_coor:backdoor_size + init_coor, init_coor:backdoor_size + init_coor] = 0
                # Unlearn class as trigger label
                poisoned_unlearn_ds.append([x, unlearn_class])

        # Ensuring local client dataset for sample unlearning containing clean dataset
        if scenario == "sample" and mode == "train":
            poisoned_unlearn_ds = ConcatDataset([poisoned_unlearn_ds, clean_unlearn_ds])

        return retain_ds, poisoned_unlearn_ds


def inject_square(
    input_tensor: torch.Tensor,
    init_coor: int,
    square_size: int,
    color: str
)-> torch.Tensor:

    if color not in ["red", "green", "blue"]:
        raise Exception("Choose correct color")

    if color == "red":
        color_list = [0.5, 0, 0]
    elif color == "green":
        color_list = [0, 0.5, 0]
    else:
        color_list = [0, 0, 0.5]

    input_tensor[0, init_coor:square_size + init_coor, init_coor:square_size + init_coor] = color_list[0]  # Red channel
    input_tensor[1, init_coor:square_size + init_coor, init_coor:square_size + init_coor] = color_list[1]  # Green channel
    input_tensor[2, init_coor:square_size + init_coor, init_coor:square_size + init_coor] = color_list[2]  # Blue channel

    return input_tensor