import torch
def BppReconstructionLoss(output, target):
    mask = target!=0
    loss = torch.nn.KLDivLoss(reduction="batchmean")
    
    return(loss(output[mask].log(), target[mask]))

