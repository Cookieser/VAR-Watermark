import torch.nn as nn
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from model.factory import get_encoder, get_decoder
from model.encoder_decoder.base_encoder_decoder import BaseEncoderDecoder
class EncoderDecoder(BaseEncoderDecoder):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderDecoder, self).__init__(config, noiser)

        self.encoder = get_encoder(config.encoder_name, config,32)
        self.noiser = noiser
        self.decoder = get_decoder(config.decoder_name, config,3)

    # MSE fhat + decoder only
    # encoder channel 32, decoder channel 3, discriminator channel 32
    def forward(self, embedding, message,var,batch):  

        encoded_embedding = self.encoder(embedding, message) # [B, 32, 16, 16]

        encoded_image = var.var_decoder(encoded_embedding) # [B, 3, 256, 256]

        original_image = var.var_decoder(embedding)

        noised_and_cover = self.noiser([encoded_image, original_image]) # [B, 3, 256, 256]

        noised_image = noised_and_cover[0]

        decoded_message = self.decoder(noised_image) # [B, 3, 256, 256]

        return embedding, encoded_embedding, noised_image, decoded_message



