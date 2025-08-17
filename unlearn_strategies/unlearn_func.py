import argparse
from src import dataset, metrics, utils
from model import models
from torch.utils.data import DataLoader
import torch
from unlearn_strategies import strategies
from unlearn_strategies import utils


def unlearn(
    args:argparse.Namespace
):
    # Device
    device, device_name = utils.device_configuration(args=args)
    print(f"Unlearning scenario: {args.scenario} Dataset: {args.dataset} Device: {device}")

    # Dataset
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root, img_size=args.img_size
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=train_dataset,
        unlearn_class=args.unlearn_class,
        scenario=args.scenario,
        num_clients=args.num_clients,
        backdoor_size=args.backdoor_size
    )

    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)
    if args.unlearn_method != "retrain":
        # Load trained model to unlearn
        model.load_state_dict(torch.load(args.model_path))

    # Unlearn
    unlearned_model = getattr(strategies, args.unlearn_method)(
        args=args,
        model=model,
        unlearn_loader=unlearn_loader,
        retain_loader=retain_loader,
        test_loader=test_loader,
        num_channels=num_channels,
        num_classes=num_classes,
        device=device
    )

    # Evaluation after unlearning
    retain_acc = metrics.evaluate(val_loader=retain_loader, model=unlearned_model, device=device)['Acc']
    unlearn_acc = metrics.evaluate(val_loader=unlearn_loader, model=unlearned_model, device=device)['Acc']
    mia = metrics.get_membership_attack_prob(retain_loader=retain_loader, forget_loader=unlearn_loader,test_loader=test_loader, model=unlearned_model)
    print(f"Unlearned - Retain acc: {retain_acc} Unlearn_acc: {unlearn_acc} MIA: {mia}")

    if args.save_model:
        utils.save_model(
            model_arc=args.model,
            model=unlearned_model,
            scenario=args.scenario,
            model_name=args.unlearn_method,
            model_root=args.model_root,
            dataset_name=args.dataset,
            train_acc=retain_acc,
            test_acc=unlearn_acc
        )