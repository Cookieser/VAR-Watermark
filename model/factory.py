from model.discriminator import Discriminator

def get_encoder(encoder_name, config,input_size):
    if encoder_name == 'encoder_cnn':
        from model.encoder.encoder_cnn import Encoder
        return Encoder(config,input_size)
    elif encoder_name == 'encoder_vit':
        from model.encoder.encoder_vit import Encoder
        return Encoder(config,input_size)
    elif encoder_name == 'encoder_De_END':
        from model.encoder.encoder_De_END import Encoder
        return Encoder(config,input_size)
    else:
        raise ValueError(f"Error: {encoder_name}")

def get_decoder(decoder_name, config,input_size):
    if decoder_name == 'decoder_cnn':
        from model.decoder.decoder_cnn import Decoder
        return Decoder(config,input_size)
    elif decoder_name == 'decoder_vit':
        from model.decoder.decoder_vit import Decoder
        return Decoder(config,input_size)
    elif decoder_name == 'decoder_De_END':
        from model.decoder.decoder_De_END import Decoder
        return Decoder(config,input_size)
    else:
        raise ValueError(f"Error: {decoder_name}")


def get_encoder_decoder_dis(encoder_decoder_name, config,noiser):
    if encoder_decoder_name == 'only_var_decoder_fhat':
        from model.encoder_decoder.only_var_decoder import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,32)

    # basic method image
    elif encoder_decoder_name == 'image_encoder_decoder':
        from model.encoder_decoder.image_encoder_decoder import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,3)

    # basic method fhat 
    elif encoder_decoder_name == 'fhat_encoder_decoder':
        from model.encoder_decoder.fhat_encoder_decoder import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,32)

    # image should use vgg
    elif encoder_decoder_name == 'only_var_decoder_image':
        from model.encoder_decoder.only_var_decoder_image import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,3)

    # neil method should use var_soft
    elif encoder_decoder_name == 'var_encoder_decoder':
        from model.encoder_decoder.var_encoder_decoder import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,32)
    
    elif encoder_decoder_name == 'encoder_decoder_De_END':
        from model.encoder_decoder.encoder_decoder_De_END import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,32)

    elif encoder_decoder_name == 'var_ed_encoder_decoder_De_END':
        from model.encoder_decoder.var_ed_encoder_decoder_De_END import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,3)

    elif encoder_decoder_name == 'var_d_encoder_decoder_De_END':
        from model.encoder_decoder.var_d_encoder_decoder_De_END import EncoderDecoder
        return EncoderDecoder(config,noiser),Discriminator(config,32)

    else:
        raise ValueError(f"Error: {encoder_decoder_name}")