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


def get_encoder_decoder(encoder_decoder_name, config,noiser):
    if encoder_decoder_name == 'encoder_decoder1':
        from model.encoder_decoder.only_var_decoder import EncoderDecoder
        return EncoderDecoder(config,noiser)
    else:
        raise ValueError(f"Error: {encoder_decoder_name}")