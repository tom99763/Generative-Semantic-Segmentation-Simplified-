import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def criterion(p, y):
    y = y.argmax(dim=1).long()
    loss = nn.CrossEntropyLoss(reduction='none')(p, y)
    loss = loss.sum(dim=1).sum(dim=1).mean()
    return loss

def criterion_latent(p, y):
    y = y.long()
    loss = nn.CrossEntropyLoss(reduction='none')(p, y)
    loss = loss.sum(dim=1).sum(dim=1).mean()
    return loss

