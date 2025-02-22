import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.encoder.base_encoder import BaseEncoder

class Encoder(BaseEncoder):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration,input_size):
        super(Encoder, self).__init__(config,input_size)
        
        layers = [ConvBNRelu(self.input_size, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + self.input_size + config.message_length * 2, self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, self.input_size, kernel_size=1)

    def forward(self, image, message):

        expanded_message = message.unsqueeze(-1)    # (B, L) -> (B, L, 1)
        expanded_message.unsqueeze_(-1)             # (B, L, 1) -> (B, L, 1, 1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)  # (B, L, H, W)
        
        encoded_image = self.conv_layers(image) # (B, conv_channels, H, W)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1) # (B, L + conv_channels + input_size , H, W)
        im_w = self.after_concat_layer(concat) #  (B, conv_channels, H, W)
        im_w = self.final_layer(im_w) #  (B, input_size, H, W)
        return im_w