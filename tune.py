from library import test, load_data, itakura_saito_loss_v01, itakura_saito_loss_v02, itakura_saito_loss_v03

from ray import air, tune
from ray.air import session

import torch
import torch.nn as nn
import torchvision.models as models

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
args, _ = parser.parse_known_args()

device = torch.device("cuda")

def train(model, optimizer, criterion, train_loader, device, eps=0.01):
    model.train()
    flag = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if criterion.__repr__() != "CrossEntropyLoss()":
            loss = criterion(output, target, eps)
        else:
            loss = criterion(output, target)
        if not flag:
            flag = loss
        loss.backward()
        optimizer.step()
    return flag

def objective(config):

    train_loader, test_loader = load_data(config["batch_size"])
    
    model = config["model"](num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = config["criterion"]

    while True:

        acc, ece, freqs = test(model, test_loader, device, ece=True, n_bins=10)
        for x in range(1):
            if criterion.__repr__() != "CrossEntropyLoss()":
                l = train(model, optimizer, criterion, train_loader, device, config["eps"])
            else:
                l = train(model, optimizer, criterion, train_loader, device)
        session.report({"accuracy": acc, "ece": ece, "example_loss": l, "freqs": freqs})


if __name__ == "__main__":
    search_space = {
        "lr": tune.grid_search([0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
        "criterion": tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": tune.grid_search([100, 200])
        }

    search_space2 = {
        "lr": tune.grid_search([0.01, 0.05, 0.001, 0.0005,]),
        "criterion": tune.grid_search([itakura_saito_loss_v01]),
        "model": tune.grid_search([models.resnet18, models.resnet50,]),
        "batch_size": tune.grid_search([100, 200, 500]),
        "eps": tune.grid_search([0.01, 0.02, 0.03, 0.04, 0.10, 0.12])
        }

    search_space3 = {
        "lr": tune.grid_search([0.001]),
        "criterion": tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": tune.grid_search([250]),
        "eps": tune.grid_search([0.1])
        }

    tuner = tune.Tuner(
        tune.with_resources(objective, {"gpu": 0.5}),
        param_space=search_space3,
        run_config=air.RunConfig(
            name="temp_experiment12",
            local_dir="./results",
            log_to_file=True,
            stop={"training_iteration": 2 if args.smoke_test else 200},
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",),
    )

    tuner.fit()
