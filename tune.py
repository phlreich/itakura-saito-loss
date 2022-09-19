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

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def objective(config):

    train_loader, test_loader = load_data(config["batch_size"])
    
    model = config["model"](num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = config["criterion"]

    while True:

        acc, ece = test(model, test_loader, device, ece=True)
        session.report({"accuracy": acc, "ece": ece})
        train(model, optimizer, criterion, train_loader, device)


search_space = {
    "lr": tune.grid_search([0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
    "criterion": tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
    "model": tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
    "batch_size": tune.grid_search([100, 200])
    }

tuner = tune.Tuner(
    tune.with_resources(objective, {"gpu": 0.5}),
    param_space=search_space,
    run_config=air.RunConfig(
        name="temp_experiment2",
        local_dir="./results",
        log_to_file=True,
        stop={"training_iteration": 4 if args.smoke_test else 100},
    ),
    tune_config=tune.TuneConfig(
        metric="accuracy",
        mode="max",),
)

tuner.fit()
