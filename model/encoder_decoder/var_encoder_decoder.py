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

    def forward(self, embedding, message,var,batch):  

        original_image = var.var_decoder(embedding)

        encoded_embedding = self.encoder(embedding, message) 

        # encoded_embedding.retain_grad()
        
        encoded_image = var.var_decoder(encoded_embedding) 
        # encoded_image.retain_grad()
        

        noised_and_cover = self.noiser([encoded_image, original_image]) 

        noised_image = noised_and_cover[0]

        # noised_image.retain_grad()
        

        noised_image_embedding = var.var_encoder(noised_image) 

        # noised_image_embedding.retain_grad()
        

        decoded_message = self.decoder(noised_image_embedding)

        # decoded_message.retain_grad()
        

        return embedding,encoded_embedding, noised_image_embedding, decoded_message