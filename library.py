import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os

def load_data(batch_size=100):
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='../../../data/',
                                                 train=True, 
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',
                                                train=False, 
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(batch_size=batch_size, 
                                            dataset=train_dataset, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(batch_size=batch_size,
                                            dataset=test_dataset,
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
    
    def get_ece(self, data_loader, device, n_bins=8):
        self.eval()
        with torch.no_grad():
            total = 0
            error = 0
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                total += 1
                preds = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(preds, 1)

                bin_number = torch.ceil(conf/(1/n_bins))

                res = 0
                for bin in range(n_bins):
                    inds = bin_number == bin
                    l = inds.sum()
                    if l > 0:
                        acc = (pred[inds] == labels[inds]).sum() / l
                        avg_conf = conf[inds].mean()
                        res += abs(avg_conf - acc)*l
                        
                res = res/labels.size(0)
                error += res
            return float((error / total))


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
