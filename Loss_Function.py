import torch.nn as nn

class DistributionFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=10):
        super(DistributionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = torch.eye(self.num_classes)[targets].to(inputs.device)
        probs = torch.softmax(inputs, dim=1)
        probs = (probs + 1e-7).clamp(min=1e-7, max=1.0)  # to avoid division by zero or NaN

        loss = -((1 - self.alpha) * torch.pow((1 - probs), self.gamma) * torch.log(probs)).mean(dim=1)
        return loss.mean()


import torch
from torch import nn
from torch.nn.functional import pairwise_distance


class CLOULoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CLOULoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        num_classes = y_pred.size(1)

        dist_pred = pairwise_distance(y_pred.unsqueeze(0), y_pred.unsqueeze(1))
        dist_true = pairwise_distance(y_true.unsqueeze(0), y_true.unsqueeze(1))

        clo_loss = torch.zeros(1).to(y_pred.device)
        for i in range(batch_size):
            for j in range(batch_size):
                for k in range(num_classes):
                    for l in range(num_classes):
                        if k != l:
                            clo_loss += torch.log(
                                1 + torch.exp(self.alpha * (dist_pred[i, j] - dist_true[k, l]))) - self.beta * \
                                        dist_true[k, l]

        clo_loss /= (batch_size * (num_classes - 1) ** 2)

        return clo_loss

