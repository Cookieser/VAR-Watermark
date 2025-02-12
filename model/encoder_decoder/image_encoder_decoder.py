import torch.nn as nn
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from model.factory import get_encoder, get_decoder
from model.encoder_decoder.base_encoder_decoder import BaseEncoderDecoder
class EncoderDecoder(BaseEncoderDecoder):
    
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__(config, noiser)

        self.encoder = get_encoder(config.encoder_name, config,3)
        self.noiser = noiser
        self.decoder = get_decoder(config.decoder_name, config,3)

    def forward(self, image, message,var,batch):

        encoded_image = self.encoder(image, message)

        noised_and_cover = self.noiser([encoded_image, image])

        noised_image = noised_and_cover[0]

        decoded_message = self.decoder(noised_image)

        return image,encoded_image, noised_image, decoded_message