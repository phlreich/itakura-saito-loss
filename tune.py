from library import test, load_data, itakura_saito_loss_v01, itakura_saito_loss_v02, itakura_saito_loss_v03

from ray import air, tune
from ray.air import session

import torch
import torch.nn as nn
import torchvision.models as models

import argparse

import numpy as np

from ray.tune.search.optuna import OptunaSearch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
args, _ = parser.parse_known_args()

device = torch.device("cuda")


def train(model, optimizer, criterion, train_loader, device, eps=0.01):
    model.train()
    sloss = 0 #records loss values for analysis
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #switch for epsilons
        if criterion.__repr__() == "CrossEntropyLoss()" or criterion.__repr__()[:32] == "<function itakura_saito_loss_v03" or criterion.__repr__()[:32] == "<function itakura_saito_loss_v04":
            loss = criterion(output, target)
        else:
            loss = criterion(output, target, eps)
        sloss += loss.item()
        i += 1
        loss.backward()
        optimizer.step()
    return sloss/i

def objective(config):

    train_loader, test_loader = load_data(config["batch_size"])
    
    model = config["model"](num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],)
    criterion = config["criterion"]
    l = float("nan")
    while True:
        
        acc, ece, freqs = test(model, test_loader, device, ece=True, n_bins=10)
        session.report(
            {"accuracy": acc, "ece": ece, "loss": l, "freqs": freqs, "mixed-score": acc-ece})
        for x in range(1):
            if criterion.__repr__() == "CrossEntropyLoss()" or criterion.__repr__()[:32] == "<function itakura_saito_loss_v03":
                l = train(model, optimizer, criterion, train_loader, device)
            else:
                l = train(model, optimizer, criterion, train_loader, device, config["eps"])



if __name__ == "__main__":

    # most of the following search spaces were used with a faulty version of the code, so they could not be used

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

    # GRID SEARCH NOT SUPPORTED BY OPTUNA
    search_space4 = {
        "lr": tune.uniform(1e-4, 1e-2),
        "criterion": nn.CrossEntropyLoss(),#tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": models.resnet152,#tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,#tune.grid_search([100, 250, 500]),
        #"eps": tune.uniform(1e-2, 0.35),
    }
    search_space5 = {
        "lr": tune.uniform(1e-4, 1e-2),
        "criterion": itakura_saito_loss_v01,#tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": models.resnet152,#tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,#tune.grid_search([100, 250, 500]),
        "eps": tune.uniform(1e-2, 0.35),
    }

    search_space6 = {
        "lr": tune.grid_search([0.0038]),
        "criterion": itakura_saito_loss_v01,#tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": models.resnet152,#tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,#tune.grid_search([100, 250, 500]),
        "eps": tune.grid_search(np.arange(0.001, 0.15, 0.001)),
    }

    search_space7 = {
        "lr": tune.grid_search([0.0038]),
        "criterion": itakura_saito_loss_v01,#tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": models.resnet152,#tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,#tune.grid_search([100, 250, 500]),
        "eps": tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]),
    }
    search_space8 = {
        "lr": tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
        "criterion": itakura_saito_loss_v01,#tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": models.resnet152,#tune.grid_search([models.resnet18, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,#tune.grid_search([100, 250, 500]),
        "eps": tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]),
    }

    search_space9 = {
        "lr": 0.0005,
        "criterion": tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": tune.grid_search([models.resnet34, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,
        "eps": 0.000001,
    }

    search_space10 = {
        "lr": 0.001,
        "criterion": tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": tune.grid_search([models.resnet34, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,
        "eps": 0.05,
    }

    search_space11 = {
        "lr": 0.001,
        "criterion": tune.grid_search([nn.CrossEntropyLoss(), itakura_saito_loss_v01]),
        "model": tune.grid_search([models.resnet34, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,
        "eps": 0.11,
    }

    search_space12 = {
        "lr": 0.001,
        "criterion": itakura_saito_loss_v01,
        "model": tune.grid_search([models.resnet34, models.resnet50, models.resnet101, models.resnet152]),
        "batch_size": 250,
        "eps": tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
    }

    search_space13 = {
        "lr": 0.001,
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet50,
        "batch_size": 250,
        "eps": tune.uniform(1e-3, 1e-1),
    }

    #establish viable parameters for IS loss
    search_space14 = {
        "lr": 0.001,
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.25]),
    }

    search_space15 = {
        "lr": 0.00001,
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.25]),
    }

    #establish viable parameters for IS loss
    a1 = {
        "lr": 0.001,
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.25]),
    }

    a2 = {
        "lr": 0.001,
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.00001, 0.000001, 0.0000001, 0.00000001]),
    }

    a3 = {
        "lr": tune.grid_search([0.00001, 0.0001]),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.00001, 0.000001, 0.0000001, 0.00000001]),
    }

    a4 = {
        "lr": tune.grid_search([0.0000001, 0.000001]),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.00001, 0.000001, 0.0000001, 0.00000001]),
    }

    # cross-entropy loss benchmark
    a5 = {
        "lr": tune.grid_search([0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]),
        "criterion": nn.CrossEntropyLoss(),
        "model": models.resnet34,
        "batch_size": 250,
        "eps": 0,
    }

    # attempt to find better-calibrated IS loss
    a6 = {
        "lr": tune.grid_search([0.005, 0.001]),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": 250,
        "eps": tune.grid_search([0.0001, 0.0002, 0.0003, 0.0004,]),
    }
    
    # optuna search space
    a7 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": tune.uniform(0.0001, 0.2),
    }

    # optuna search space, maximizing the acc score for itakura-saito loss
    a10 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": tune.uniform(0.0001, 0.1),
    }

    # optuna search space, maximizing the acc score for cross-entropy loss
    a11 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": nn.CrossEntropyLoss(),
        "model": models.resnet34,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": 0# tune.uniform(0.0001, 0.1),
    }

    # direct comparison grid search
    a12 ={
        "lr": tune.grid_search([0.0011, 0.0026]),
        "criterion": tune.grid_search([itakura_saito_loss_v01, nn.CrossEntropyLoss()]),
        "model": models.resnet34,
        "batch_size": tune.grid_search([250, 500]),
        "eps": 0.07658
    }

    # IS-loss max acc
    a13 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet34,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": tune.uniform(0.0001, 0.1),
    }

    # CE-loss max acc
    a14 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": nn.CrossEntropyLoss(),
        "model": models.resnet34,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": 0# tune.uniform(0.0001, 0.1),
    }

    # direct comparison grid search
    a15 ={
        "lr": tune.grid_search([0.0012, 0.0011]),
        "criterion": tune.grid_search([itakura_saito_loss_v01, nn.CrossEntropyLoss()]),
        "model": models.resnet34,
        "batch_size": 100,#tune.grid_search([250, 500]),
        "eps": 0.0533
    }

    # IS-loss max acc for larger net
    a16 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": itakura_saito_loss_v01,
        "model": models.resnet152,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": tune.uniform(0.0001, 0.1),
    }

    # CE-loss max acc for larger net
    a17 = {
        "lr": tune.uniform(0.0005, 0.005),
        "criterion": nn.CrossEntropyLoss(),
        "model": models.resnet152,
        "batch_size": tune.choice([100, 250, 500]),
        "eps": 0# tune.uniform(0.0001, 0.1),
    }

    a18 ={
        "lr": tune.grid_search([0.0006, 0.0036]),
        "criterion": tune.grid_search([itakura_saito_loss_v01, nn.CrossEntropyLoss()]),
        "model": models.resnet152,
        "batch_size": 100,#tune.grid_search([250, 500]),
        "eps": 0.0967
    }
    #TODO run 11

    algo = OptunaSearch()
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=8)
    num_samples = 8 if args.smoke_test else 200

    tuner = tune.Tuner(
        # 2 trials per gpu
        tune.with_resources(objective, {"gpu": 0.5}),
        param_space=a18,
        run_config=air.RunConfig(
            name="a18",
            local_dir="./results",
            log_to_file=True,
            sync_config=tune.SyncConfig(
                syncer=None,
            ),
            stop={"training_iteration": 2 if args.smoke_test else 300},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            scheduler=tune.schedulers.ASHAScheduler(
                time_attr="training_iteration",
                max_t=150,
                grace_period=10,
                reduction_factor=2,
                brackets=1,
            ),
        )
    )

    tuner.fit()
