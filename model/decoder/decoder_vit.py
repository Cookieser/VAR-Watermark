import torch
from torchvision.models.vision_transformer import vit_b_16
import timm
import torch.nn.functional as F
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.decoder.base_decoder import BaseDecoder


class Decoder(BaseDecoder):
    
    def __init__(self, config: HiDDenConfiguration,input_size):

        super(Decoder, self).__init__(config,input_size)

        model_name="vit_base_patch16_224"

        num_bits = config.message_length

        
        self.model = timm.create_model(
            model_name, 
            pretrained=True,  
            img_size=224, 
            num_classes=0  
        )

        in_features = self.model.num_features


        self.cnn_layers = nn.Sequential(
            ConvBNRelu(3, 64),
            ConvBNRelu(64, 128),
            ConvBNRelu(128, 256),
            ConvBNRelu(256, 512)
        )

        self.decoder_head = nn.Linear(in_features, num_bits)

        self.cnn_fc = nn.Linear(512, num_bits)

        print("Finish the init of ViT_patch16_224")

    def forward(self, x):
        
        x_cnn = self.cnn_layers(x)  # (B, 512, H, W)
        x_cnn = F.adaptive_avg_pool2d(x_cnn, (1, 1)).squeeze(-1).squeeze(-1)  
        
        x_vit = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x_vit = self.model(x_vit)  # (B, in_features)

        logits = self.decoder_head(x_vit) + self.cnn_fc(x_cnn)  

        return logits 