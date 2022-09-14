from operator import mod
from library import ResNet, ResidualBlock, load_data, itakura_saito_loss_v01, itakura_saito_loss_v02, itakura_saito_loss_v03

from sched import scheduler
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from ray.air import session
from ray.air.checkpoint import Checkpoint

import torch
import torch.nn as nn

if __name__ == "__main__":
    device = "cpu"


    def objective(config):
        train_loader, test_loader = load_data(config["batch_size"])
        model = ResNet(ResidualBlock, config["architecture"])#.to(device)
        optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
        criterion = config["criterion"]
        step = 0

        while True:

            step += 1
            model.train_epoch(train_loader, optimizer, criterion, device)
            acc = model.test(test_loader, device)
            ece = model.get_ece(test_loader, device)

            session.report({"accuracy": acc, "ece": ece})
            


    search_space1 = {
        "lr": tune.grid_search([0.0001, 0.0005, 0.001]),
        "criterion": tune.grid_search([itakura_saito_loss_v01, nn.CrossEntropyLoss()]),
        "optimizer": tune.grid_search([torch.optim.SGD, torch.optim.Adam]),
        "architecture": tune.grid_search([[3,3,3], [4,4,4]]),
        "batch_size": tune.grid_search([128, 256])
      }
    search_space2 = {
        "lr": tune.grid_search([0.0005, 0.0001]),
        "criterion": tune.grid_search([itakura_saito_loss_v01, nn.CrossEntropyLoss()]),
        "optimizer": tune.grid_search([torch.optim.Adam]),
        "architecture": tune.grid_search([[3, 3, 3]]),
      }
    


    #resources = {"cpu": 1, "gpu": 0} #TODO balance ressource utilization
    tuner = tune.Tuner(
        objective,
        param_space=search_space1,
        run_config=air.RunConfig(
            name="ece_experiment",
            local_dir="./results",
            log_to_file=True,
        ),
        tune_config=tune.TuneConfig(
            scheduler=tune.schedulers.ASHAScheduler(metric="accuracy", mode="max", max_t=32, grace_period=8))
    )
    tuner.fit()
