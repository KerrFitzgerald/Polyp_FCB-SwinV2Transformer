import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

"""PyTorch loss functions and scoring metrics for single image averaging"""


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment next line out if model contains a sigmoid activation layer
        # inputs = F.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        dice_per_image = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            dice = (2.*intersection + smooth)/(input_.sum() + target.sum() + smooth)
            dice_per_image.append(1 - dice)
        return torch.mean(torch.stack(dice_per_image))
    
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment next line out if model contains a sigmoid activation layer
        # inputs = F.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        iou_per_image = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            total = (input_ + target).sum()
            union = total - intersection
            iou = (intersection + smooth) / (union + smooth)
            iou_per_image.append(1 - iou)
        
        return torch.mean(torch.stack(iou_per_image))

    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment next line out if model contains a sigmoid activation layer
        # inputs = F.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        dice_bce_per_image = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            dice_loss = 1 - (2.*intersection + smooth)/(input_.sum() + target.sum() + smooth)
            BCE = F.binary_cross_entropy(input_, target, reduction='mean')
            
            dice_bce = BCE + dice_loss
            dice_bce_per_image.append(dice_bce)
        return torch.mean(torch.stack(dice_bce_per_image))

    
def Threshold_DiceLoss(inputs, targets, thresh=0.5, smooth=1e-8):
    inputs = 1*(inputs >= thresh)
    #print(inputs)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    dice_per_image = []
    for input_, target in zip(inputs, targets):
        intersection = (input_ * target).sum()
        dice = (2.*intersection + smooth)/(input_.sum() + target.sum() + smooth)
        dice_per_image.append(1 - dice)
    return torch.mean(torch.stack(dice_per_image))


def Threshold_IoULoss(inputs, targets, thresh=0.5, smooth=1e-6):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    iou_per_image = []
    for input_, target in zip(inputs, targets):
        intersection = (input_ * target).sum()
        total = (input_ + target).sum()
        union = total - intersection
        iou = (intersection + smooth)/(union + smooth)
        iou_per_image.append(1 - iou)
    return torch.mean(torch.stack(iou_per_image))


def custom_precision_score(inputs, targets, thresh=0.5, smooth=1e-6):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    precision_per_image = []
    for input_, target in zip(inputs, targets):
        TP = ((input_ == 1) & (target == 1))
        FP = ((input_ == 1) & (target == 0))
        precision = torch.sum(TP.float())/(torch.sum(TP.float())+torch.sum(FP.float())+smooth)
        precision_per_image.append(precision)
    return torch.mean(torch.stack(precision_per_image))


def custom_recall_score(inputs, targets, thresh=0.5, smooth=1e-6):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    recall_per_image = []
    for input_, target in zip(inputs, targets):
        TP = ((input_ == 1) & (target == 1))
        FN = ((input_ == 0) & (target == 1))
        recall = torch.sum(TP.float())/(torch.sum(TP.float())+torch.sum(FN.float()) + smooth)
        recall_per_image.append(recall)
    return torch.mean(torch.stack(recall_per_image))
