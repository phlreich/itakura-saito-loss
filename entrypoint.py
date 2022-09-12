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
    #patience = 10
    def objective(config):
        train_loader, test_loader = load_data()
        model = ResNet(ResidualBlock, config["architecture"])#.to(device)
        optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
        criterion = config["criterion"]
        # start training loop
        #acc_best = 0
        #patience_acc = 0
        step = 0
        #loaded_checkpoint = session.get_checkpoint()
        #if loaded_checkpoint:
        #    step = loaded_checkpoint.to_dict()["step"]
        #    acc_best = loaded_checkpoint.to_dict()["acc_best"]
        #    patience_acc = loaded_checkpoint.to_dict()["patience_acc"]
        #    step += 1

        while True:
            #time.sleep(1)
            step += 1
            model.train_epoch(train_loader, optimizer, criterion, device)
            acc = model.test(test_loader, device)
            ece = model.get_ece(test_loader, device)
            #if acc_best > acc:
            #    patience_acc += 1
            #else:
            #    patience_acc = 0
            #    acc_best = acc
            #checkpoint = Checkpoint.from_dict({"step": step, "acc_best": acc_best, "patience_acc": patience_acc})
            session.report({"accuracy": acc, "ece": ece})#, checkpoint=checkpoint)
            #if patience_acc >= patience or step >= 2:
            #    break #TODO FIX


    search_space1 = {
        "lr": tune.grid_search([0.0001, 0.0005, 0.001]),
        "criterion": tune.grid_search([itakura_saito_loss_v01, nn.CrossEntropyLoss()]),
        "optimizer": tune.grid_search([torch.optim.SGD, torch.optim.Adam]),
        "architecture": tune.grid_search([[2,2,2], [3,3,3], [4,4,4]]),
      }
    search_space2 = {
        "lr": tune.grid_search([0.0001, 0.0003]),#, 0.0005, 0.001]),
        "criterion": tune.grid_search([nn.CrossEntropyLoss()]),
        "optimizer": tune.grid_search([torch.optim.SGD]),# torch.optim.Adam]),
        "architecture": tune.grid_search([[2,2,2]]),#, [3,3,3], [4,4,4]]),
      }
    #resources = {"cpu": 1, "gpu": 0}
    tuner = tune.Tuner(
        objective,
        param_space=search_space2,
        run_config=air.RunConfig(
            name="temp_experiment",
            # name="main_experiment_v03",
            local_dir="./results",
            log_to_file=True,
            #stop={"training_iteration": 5},
        ),
        tune_config=tune.TuneConfig(
            scheduler=tune.schedulers.ASHAScheduler(metric="accuracy", mode="max", max_t=10, grace_period=2))
    )
    tuner.fit()
