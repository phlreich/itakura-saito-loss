from sched import scheduler
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from ray.air import session
from ray.air.checkpoint import Checkpoint

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import time

def load_data():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                 train=True, 
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=False, 
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100, 
                                              shuffle=False)
    return train_loader, test_loader
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def train_epoch(self, train_loader, optimizer, criterion, device):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = self(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(self, data_loader, device):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # calculate ECE
                preds = torch.softmax(outputs, dim=1)
                bins = 9
                
                confidences, predictions = torch.max(preds, 1)

        
        return (correct / total)

#model18 = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#model34 = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
#model = model34

#criterion = nn.CrossEntropyLoss()

def itakura_saito_loss_v01(pred, y, eps=1e-4):
    pred = torch.softmax(pred,dim=1)
    pred = pred + eps
    logs = torch.log(pred)
    logsum = logs.sum(dim=1)
    sec = 1/pred
    res = logsum
    res = res + sec.gather(1,y.view(-1,1))
    return res.mean()

def itakura_saito_loss_v02(pred, y, epsilon=1e-4):
    pred = torch.softmax(pred,dim=1)

    # avoid nan
    pred = pred + epsilon
    pred = pred - (pred/pred.sum()) * (len(pred) * epsilon) #restores probability vector characteristic #TODO is this reaaally necessary?

    logs = torch.log(pred)
    logsum = logs.sum(dim=1)
    sec = 1/pred
    res = logsum
    res = res + sec.gather(1,y.view(-1,1))
    return res.mean()


def itakura_saito_loss_v03(pred, y):
    n_classes = pred.shape[1]
    logsumexp = torch.logsumexp(pred, dim=1)
    ys = pred.gather(1,y.view(-1,1))
    expys = torch.exp(ys).flatten()
    res = logsumexp / expys - n_classes * logsumexp + pred.sum(dim=1)
    return res.mean()


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

        #if acc_best > acc:
        #    patience_acc += 1
        #else:
        #    patience_acc = 0
        #    acc_best = acc
        #checkpoint = Checkpoint.from_dict({"step": step, "acc_best": acc_best, "patience_acc": patience_acc})
        session.report({"accuracy": acc})#, checkpoint=checkpoint)
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
        scheduler=tune.schedulers.ASHAScheduler(metric="accuracy", mode="max", max_t=100, grace_period=20))
)
tuner.fit()
