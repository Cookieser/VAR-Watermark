import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)


    # def forward(self, image, message):
    #     encoded_image = self.encoder(image, message)
    #     noised_and_cover = self.noiser([encoded_image, image])
    #     noised_image = noised_and_cover[0]
    #     decoded_message = self.decoder(noised_image)
    #     return encoded_image, noised_image, decoded_message


    def forward(self, embedding, message,var,batch):  

        original_image = var.embed_to_image(embedding)

        encoded_embedding = self.encoder(embedding, message) # [B, 32, 16, 16]

        encoded_image = var.embed_to_image(encoded_embedding) # [B, 3, 256, 256]

        noised_and_cover = self.noiser([encoded_image, original_image]) # [B, 3, 256, 256]

        noised_image = noised_and_cover[0]

        noised_image_embedding = var.image_to_embed(noised_image,batch) # [B, 32, 16, 16]

        decoded_message = self.decoder(noised_image_embedding)

        return encoded_embedding, noised_image_embedding, decoded_message



