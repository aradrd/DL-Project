import torch
from models.subsampling import SubsamplingLayer
from models.unet.unet_model import UNet
from matplotlib.pyplot import figure, imshow, subplot, title, tight_layout

class UNet_Rec(torch.nn.Module):
    def __init__(self,drop_rate, device, learn_mask):
        super().__init__()
        
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask)
        self.reconstruct = UNet(
            n_channels=1,
            bilinear=False
        )
        
    def forward(self, x):
        x = self.subsample(x)
        x = self.reconstruct(x)
        return x