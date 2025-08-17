import torch
from torch.utils.data import DataLoader
import argparse
import copy
from tqdm import tqdm
import random
from unlearn_strategies import utils


def baseline(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device
) -> torch.nn.Module:
    return model


def retrain(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device
) -> torch.nn.Module:

    # Retrain model from scratch
    retrain_model = utils.training_optimization(
        model= model,
        train_loader= retain_loader,
        test_loader= test_loader,
        epochs= 50,
        device= device,
        desc= "Retraining model"
    )

    return retrain_model


def fine_tune(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    num_channels: int,
) -> torch.nn.Module:

    # Fine tune model
    ft_model = utils.training_optimization(
        model= model,
        train_loader= retain_loader,
        epochs= 10,
        device= device,
        desc= "Fine-tuning model"
    )

    return ft_model


def maverick(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearn_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    num_channels: int,
    device: torch.device
) -> torch.nn.Module:

    original_model = copy.deepcopy(model)
    unlearn_model = copy.deepcopy(model)

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        for images, labels in tqdm(unlearn_loader, desc= f"FedMaverick Unlearning: {epoch}"):
            optimizer.zero_grad()
            images = images.to(device)
            # Normal images output logits
            output_images = unlearn_model(images)

            total_lpc_loss = 0.0
            for i in range(args.sample_number):
                # Generate random sigma for every iteration
                sigma = random.uniform(args.min_sigma, args.max_sigma)
                perturbed_images = copy.deepcopy(images)

                if args.scenario == "class":
                    # Generate gaussian noise
                    gaussian_noise = torch.normal(mean=args.mean, std= sigma, size= perturbed_images.size()).to(device)
                    # Add gaussian perturbation to whole image
                    perturbed_images += gaussian_noise

                else: # client and sample
                    # Generate gaussian noise size based on backdoor size
                    noise_size = (perturbed_images.size()[0], num_channels, args.backdoor_size, args.backdoor_size)
                    gaussian_noise = torch.normal(mean=args.mean, std=sigma, size=noise_size).to(device)
                    # Add gaussian perturbation on backdoor region instead of whole image
                    perturbed_images[:, :, 2:args.backdoor_size + 2, 2:args.backdoor_size + 2] += gaussian_noise

                # Perturbed images output logits
                with torch.no_grad():
                    #output_perturbed_images = unlearn_model(perturbed_images)
                    output_perturbed_images = original_model(perturbed_images)

                # Lipschitz loss computation
                lpc_loss = utils.lipschitz_loss(
                    image= images,
                    image_perturbed= perturbed_images,
                    output_image= output_images,
                    output_image_perturbed= output_perturbed_images
                )

                total_lpc_loss += lpc_loss

            avg_lpc_loss = total_lpc_loss / (args.sample_number * args.batch_size)
            loss = avg_lpc_loss
            loss.backward()
            optimizer.step()
            tqdm.write(f"Loss: {loss.item()}")

    return unlearn_model