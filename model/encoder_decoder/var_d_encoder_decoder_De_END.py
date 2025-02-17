import torch
import torch.nn as nn
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from model.factory import get_encoder, get_decoder
from model.encoder_decoder.base_encoder_decoder import BaseEncoderDecoder
class EncoderDecoder(BaseEncoderDecoder):
    
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__(config, noiser)

        self.encoder = get_encoder(config.encoder_name, config,32)
        self.noiser = noiser
        self.decoder = get_decoder(config.decoder_name, config,3)

    


    def forward(self, fhat, message,var,batch): 


        original_image = var.var_decoder(fhat)
        decoded_message = self.decoder(original_image) # [B,L]
        cat_message = torch.cat([decoded_message, message], dim=1) # [B,2L]
        add_watermark_fhat = self.encoder(fhat, cat_message) # [B,32,16,16]
        cat_watermark_fhat = fhat + add_watermark_fhat     # [B,32,16,16]

        watermark_image = var.var_decoder(cat_watermark_fhat)  # [B,3,256,256]
       

        noised_and_cover = self.noiser([watermark_image, original_image])
        noised_image = noised_and_cover[0]

        decoded_message = self.decoder(noised_image)

        return fhat,cat_watermark_fhat, noised_image, decoded_message