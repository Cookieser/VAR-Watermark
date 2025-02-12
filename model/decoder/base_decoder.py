import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module, ABC):
    def __init__(self, config,input_size):
        super(BaseDecoder, self).__init__()
        self.channels = config.decoder_channels
        self.input_size = input_size
        

    @abstractmethod
    def forward(self, x):
        pass