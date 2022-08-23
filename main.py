# Import the tuning library from ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


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
        
        return (correct / total)

#model18 = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#model34 = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
#model = model34

#criterion = nn.CrossEntropyLoss()

def loss_fn_v01(pred, y):
    pred = torch.softmax(pred,dim=1)
    pred = pred + 0.001
    logs = torch.log(pred)
    logsum = logs.sum(dim=1)
    sec = 1/pred
    res = logsum
    res = res + sec.gather(1,y.view(-1,1))
    return res.mean()

def loss_fn_v02(pred, y, epsilon=1e-4):
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


def loss_fn_v03(pred, y):
    n_classes = pred.shape[1]
    logsumexp = torch.logsumexp(pred, dim=1)
    ys = pred.gather(1,y.view(-1,1))
    expys = torch.exp(ys).flatten()
    res = logsumexp / expys - n_classes * logsumexp + pred.sum(dim=1)
    return res.mean()


device = "cpu"
patience = 5
def objective(config):
    #train_loader, test_loader
    train_loader, test_loader = load_data()
    model = ResNet(ResidualBlock, config["architecture"]).to(device)
    # Use function arguments for the optimizer to tune it
    #optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
    criterion = config["criterion"]
    # start training loop
    acc_best = 0
    patience_acc = 0
    epoch = 0
    while True:
        epoch += 1
        model.train_epoch(train_loader, optimizer, criterion, device)
        acc = model.test(test_loader, device)
        #with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        #    path = os.path.join(checkpoint_dir, "checkpoint")
        #    save_obj = (model.state_dict(), optimizer.state_dict())
        #    torch.save(save_obj, path)
        if acc_best > acc:
            patience_acc += 1
        else:
            patience_acc = 0
            acc_best = acc

        if patience_acc >= patience:
            tune.report(accuracy=acc, done=True)
        else:
            tune.report(accuracy=acc, done=False)


search_space = {
    "lr": tune.grid_search([0.0001, 0.0005, 0.001]),
    "criterion": tune.grid_search([loss_fn_v01, nn.CrossEntropyLoss()]),
    "optimizer": tune.grid_search([torch.optim.SGD, torch.optim.Adam]),
    "architecture": tune.grid_search([[0,0,0], [1,1,1], [2,2,2], [3,3,3]]),
  }

resources_per_trial = {"cpu": 1, "gpu": 0}
analysis = tune.run(
    objective,
    config=search_space,
    resources_per_trial=resources_per_trial,
    # a single sample is the default and useful for grid_search
    # but if we have an infinite space, we will need more
    num_samples=1,
    local_dir="./results", 
    name="main_experiment_v02",
    log_to_file=True,
)