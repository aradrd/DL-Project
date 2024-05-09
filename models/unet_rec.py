import torch
from models.subsampling import SubsamplingLayer
from models.unet.unet_model import UNet

class UNet_Rec(torch.nn.Module):
    def __init__(self,drop_rate, device, learn_mask):
        super().__init__()
        
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask) #initialize subsampling layer - use this in your own model
        self.reconstruct = UNet(
            n_channels=1,
            bilinear=False
        )
        
    def forward(self, x):
        x = self.subsample(x) #get subsampled input in image domain - use this as first line in your own model's forward
        x = self.reconstruct(x)
        return x