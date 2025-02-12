import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.encoder.base_encoder import BaseEncoder

class Encoder(BaseEncoder):
    
    def __init__(self, config: HiDDenConfiguration,input_size):
        super(Encoder, self).__init__(config,input_size)

        self.message_length = config.message_length
        self.num_layers     = config.encoder_blocks  
        self.hidden_dim     =  input_size            # 这里将 Transformer 的 d_model 设为 32
        self.height, self.width = config.H, config.W

        self.num_tokens = self.height * self.width

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,              
            dim_feedforward=128,  
            dropout=0.1,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.pre_transformer_msg_proj = nn.Linear(self.hidden_dim + self.message_length, self.hidden_dim)

        self.final_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim))


    def forward(self, x, message):
        B, C, H, W = x.shape
        assert C == self.hidden_dim, "Wrong!!!"
        assert H == self.height and W == self.width, "Wrong!!!"
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        message_expanded = message.unsqueeze(1).expand(-1, self.num_tokens, -1)
        concat_tokens = torch.cat([x_flat, message_expanded], dim=2)

        input_tokens = self.pre_transformer_msg_proj(concat_tokens)
        input_tokens = input_tokens + self.pos_embedding

        input_tokens = input_tokens.permute(1, 0, 2)

        encoded_tokens = self.transformer_encoder(input_tokens)

        out_tokens = self.final_linear(encoded_tokens) 

        out_tokens = out_tokens.permute(1, 2, 0)  # (B, 32, 256)

        out = out_tokens.view(B, C, H, W)

        return out









