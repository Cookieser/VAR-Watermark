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
        self.decoder = get_decoder(config.decoder_name, config,32)

    
    def forward(self, fhat, message,var,batch):

        decoded_message = self.decoder(fhat) # [B,L]

        cat_message = torch.cat([decoded_message, message], dim=1) # [B,2L]

        add_watermark_fhat = self.encoder(fhat, cat_message) # [B,32,16,16]

        cat_watermark_fhat = fhat + add_watermark_fhat

        noised_and_cover = self.noiser([cat_watermark_fhat, fhat])

        noised_fhat = noised_and_cover[0]

        final_message = self.decoder(noised_fhat)
        
        return fhat,cat_watermark_fhat, noised_fhat, final_message