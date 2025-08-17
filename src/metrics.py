"""
Evaluation metrics
"""
from torch.nn import functional as F
import copy
import os
import torch
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import argparse
import unlearn_strategies
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Classification accuracy metrics
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def validation_step(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.long().to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}


def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": round(epoch_loss.item(), 4), "Acc": round(epoch_acc.item(), 4)}


@torch.no_grad()
def evaluate(model, val_loader, device):
    copy_model= copy.deepcopy(model)
    copy_model.eval()
    outputs = [validation_step(copy_model, batch, device) for batch in val_loader]
    return validation_epoch_end(copy_model, outputs)


# Reference of Membership Inference Attack(Code Implementation): https://github.com/if-loops/selective-synaptic-dampening
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    copy_model = copy.deepcopy(model) # avoid overwriting
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, copy_model
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return round(results.mean() * 100, 4)


# Model Sensitivity
def model_sensitivity(
    args: argparse.Namespace,
    model: torch.nn.Module,
    unlearn_loader: DataLoader,
    device: torch.device
) -> float:

    total_sensitivity = []
    unlearn_model = copy.deepcopy(model)
    with torch.no_grad():
        for images, labels in unlearn_loader:

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
                    gaussian_noise = torch.normal(mean= 0, std= sigma, size= perturbed_images.size()).to(device)
                    # Add gaussian perturbation to whole image
                    perturbed_images += gaussian_noise

                else: # client and sample
                    # Generate gaussian noise size based on backdoor size
                    noise_size = (perturbed_images.size()[0], perturbed_images.size()[1], args.backdoor_size, args.backdoor_size)
                    gaussian_noise = torch.normal(mean=0, std=sigma, size=noise_size).to(device)
                    # Add gaussian perturbation on backdoor region instead of whole image
                    perturbed_images[:, :, 2:args.backdoor_size + 2, 2:args.backdoor_size + 2] += gaussian_noise

                # Perturbed images output logits
                output_perturbed_images = unlearn_model(perturbed_images)

                # Lipschitz loss computation
                lpc_loss = unlearn_strategies.lipschitz_loss(
                    image= images,
                    image_perturbed= perturbed_images,
                    output_image= output_images,
                    output_image_perturbed= output_perturbed_images
                )

                total_lpc_loss += lpc_loss

            avg_lpc_loss = total_lpc_loss / (args.sample_number * args.batch_size)
            total_sensitivity.append(avg_lpc_loss.item())

    sensitivity = float(np.mean(np.array(total_sensitivity)))
    sensitivity = round(sensitivity, 4)

    return sensitivity


def model_evaluation(
    retain_loader: DataLoader,
    unlearn_loader: DataLoader,
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device
)-> Tuple[float, float, float]:

    retain_acc = evaluate(val_loader= retain_loader, model= model, device= device)["Acc"]
    unlearn_acc = evaluate(val_loader= unlearn_loader, model= model, device= device)["Acc"]
    mia_asr = get_membership_attack_prob(
        retain_loader= retain_loader, forget_loader= unlearn_loader, test_loader= test_loader, model= model)

    return retain_acc, unlearn_acc, mia_asr


def metrics_fl(
    retain_client_trainloader: DataLoader,
    unlearn_client_trainloader: DataLoader,
    retain_client_testloader: DataLoader,
    unlearn_client_testloader: DataLoader,
    global_model: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:

    retain_train_acc = evaluate(val_loader= retain_client_trainloader, model= global_model, device= device)["Acc"]
    retain_test_acc = evaluate(val_loader=retain_client_testloader, model=global_model, device=device)["Acc"]
    unlearn_train_acc = evaluate(val_loader=unlearn_client_trainloader, model=global_model, device=device)["Acc"]
    unlearn_test_acc = evaluate(val_loader=unlearn_client_testloader, model=global_model, device=device)["Acc"]

    return retain_train_acc, unlearn_train_acc, retain_test_acc, unlearn_test_acc