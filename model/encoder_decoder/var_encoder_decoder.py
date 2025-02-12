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

    # def forward(self, embedding, message,var,batch):  

    #     original_image = var.embed_to_image(embedding)

    #     encoded_embedding = self.encoder(embedding, message) # [B, 32, 16, 16]

    #     encoded_image = var.embed_to_image(encoded_embedding) # [B, 3, 256, 256]

    #     noised_and_cover = self.noiser([encoded_image, original_image]) # [B, 3, 256, 256]

    #     noised_image = noised_and_cover[0]

    #     noised_image_embedding = var.image_to_embed(noised_image,batch) # [B, 32, 16, 16]

    #     decoded_message = self.decoder(noised_image_embedding)

    #     return embedding,encoded_embedding, noised_image_embedding, decoded_message