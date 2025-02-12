from model.discriminator import Discriminator

def get_encoder(encoder_name, config,input_size):
    if encoder_name == 'encoder_cnn':
        from model.encoder.encoder_cnn import Encoder
        return Encoder(config,input_size)
    else:
        raise ValueError(f"Error: {encoder_name}")

def get_decoder(decoder_name, config,input_size):
    if decoder_name == 'decoder_cnn':
        from model.decoder.decoder_cnn import Decoder
        return Decoder(config,input_size)
    else:
        raise ValueError(f"Error: {decoder_name}")


def get_encoder_decoder_dis(encoder_decoder_name, config,noiser):
    if encoder_decoder_name == 'only_var_decoder_fhat':
        from model.encoder_decoder.only_var_decoder import EncoderDecoder
        discriminator_channel_num = 32
        return EncoderDecoder(config,noiser),Discriminator(config,32)
    else:
        raise ValueError(f"Error: {encoder_decoder_name}")