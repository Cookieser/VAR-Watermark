import torch.nn as nn
from abc import ABC, abstractmethod
class BaseEncoder(nn.Module,ABC):
    def __init__(self, config,input_size):
        super(BaseEncoder, self).__init__()

        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

    
        self.input_size = input_size
        
    
    @abstractmethod
    def forward(self, x):
        pass