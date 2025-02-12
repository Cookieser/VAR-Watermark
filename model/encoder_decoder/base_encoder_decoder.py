from abc import ABC, abstractmethod
import torch.nn as nn

from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
class BaseEncoderDecoder(nn.Module,ABC):
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(BaseEncoderDecoder,self).__init__()
        
        
    @abstractmethod
    def forward(self, model, embedding, message, var, batch):
        pass