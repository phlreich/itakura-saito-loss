import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os

def load_data(batch_size=100, root='/home/preich/itakura-saito-loss/data/'):
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True, 
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root=root,
                                                train=False, 
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(batch_size=batch_size, 
                                            dataset=train_dataset, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(batch_size=batch_size,
                                            dataset=test_dataset,
                                            shuffle=False)
    return train_loader, test_loader


def test(model, test_loader, device="cpu", ece=False, n_bins=10):
    model.eval()
    correct = 0
    error = 0
    total = 0
    freqs = torch.zeros(n_bins).to(device)
    totals = torch.zeros(n_bins).to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if ece:
                bin_number = torch.ceil(confidences/(1/n_bins))
                bin_number -= 1 # bin 1 is actually bin 0
                for bin in range(n_bins):
                    inds = bin_number == bin
                    l = inds.sum()
                    if l > 0:
                        acc = (predicted[inds] == labels[inds]).sum() / l
                        avg_conf = confidences[inds].mean()
                        error += abs(avg_conf - acc)*l
                        freqs[bin] += acc*l
                        totals[bin] += l
                        
    if not ece: return (correct / total)
    print(error, total)
    return (correct / total), float(error / total), (freqs/totals) #acc, ece, values for reliability diagram where nan = no samples in bin

def itakura_saito_loss_v01(logits, labels, eps=1e-4):
    """
    logits: (batch_size, n_classes), labels: (batch_size)
    """
    predictions = torch.softmax(logits, dim=1) + eps
    logs = torch.log(predictions)
    logsum = logs.sum(dim=1)
    sec = 1/predictions
    res = logsum + sec.gather(1,labels.view(-1,1))
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
