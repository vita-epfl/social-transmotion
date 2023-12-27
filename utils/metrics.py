import numpy as np
import torch

def MSE_LOSS(output, target, mask=None):
    
    pred_xy = output[:,:,0,:2]
    gt_xy = target[:,:,0,:2]

    norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1)
    mean_B = torch.mean(mean_K)

    return mean_B*100
