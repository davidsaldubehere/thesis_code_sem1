import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        #For some reason targets isn't long by default so change that

        targets = targets.long()

        # Apply softmax if inputs are logits
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N, C, H*W
            inputs = inputs.transpose(1, 2)    # N, H*W, C
            inputs = inputs.contiguous().view(-1, inputs.size(-1))  # N*H*W, C
        targets = targets.view(-1, 1)
        
        # Compute log softmax for focal loss
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        
        pt = logpt.exp()
        
        # Compute focal loss
        focal_loss = -1 * self.alpha * (1 - pt) ** self.gamma * logpt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
