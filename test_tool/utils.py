import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

def load_hidden(options_file_path,checkpoint_file_path,device):
    train_options, hidden_config, noise_config = utils.load_options(options_file_path)
    noiser = Noiser(noise_config,device)
    checkpoint = torch.load(checkpoint_file_path,weights_only=True)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    return hidden_net.encoder_decoder.encoder, hidden_net.encoder_decoder.decoder



def load_fhat(source_image_path,batch,device):
    fhat = torch.load(source_image_path,weights_only=True)
    print("Original Tensor Shape:", fhat.shape)
    if fhat.dim() == 3:
        fhat.unsqueeze_(0)
    if fhat.dim() == 4 and fhat.shape[0] > 10:
        fhat = fhat[:batch]
    fhat = fhat.to(device)
    print("Original Tensor Shape:", fhat.shape)
    return fhat





def add_watermark(image,message,hidden_encoder,device):
    message = message.to(device)
    image = image.to(device)
    encoded_image = hidden_encoder(image,message)
    return encoded_image


def get_watermark(image,hidden_decoder,device):
    image = image.to(device)
    decoded_message  = hidden_decoder(image)
    return decoded_message 



def cal_error(decoded_message,message):
    
    decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
    message_detached = message.detach().cpu().numpy()
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(fhat.shape[0] * message.shape[1])
    return bitwise_avg_err


def repeat_test(test_func, num_trials=100):
    total_bitwise_avg_err = 0.0
    for i in range(num_trials):
        bitwise_avg_err = test_func()  
        total_bitwise_avg_err += bitwise_avg_err
    average_bitwise_avg_err = total_bitwise_avg_err / num_trials
    
    return average_bitwise_avg_err



def save(recon_B3HW,output="temp.png"):  
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=recon_B3HW.shape[0], padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    chw.save(output)
    print(f"Image({recon_B3HW.size()}) saved to {output}")




def draw(file_path,file_name,y_low = 0,y_high = 0.5):
    file_path = os.path.join(file_path,file_name)

    df = pd.read_csv(file_path)

  
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["encoder_mse"], label="encoder_mse", marker='o')
    plt.plot(df["epoch"], df["dec_mse"], label="dec_mse", marker='s')
    plt.plot(df["epoch"], df["bitwise-error"], label="bitwise-error", marker='^')

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Encoder MSE, Decoder MSE, and Bitwise Error over Epochs")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_low, y_high)

    plt.show()


def test_one_method(var,base_path,device,pid,y1=0,y2=0.5):
    draw(base_path,"train.csv",y1,y2)
    draw(base_path,"validation.csv",y1,y2)
    source_image = os.path.join(base_path,f"images/epoch-original-{pid}.pt")
    fhat = load_fhat(source_image,8,device)
    image = var.var_decoder(fhat)

    var.save_image(image,"original_picture.png") 

    watermark_image_path = os.path.join(base_path,f"images/epoch-watermark-{pid}.pt")

    try:
        watermark_image = torch.load(watermark_image_path,weights_only=True)
        if watermark_image.shape[1] == 3 and watermark_image.shape[2] == 256 and watermark_image.shape[3] == 256:
            print("Using direct torch.load() method.")
        else:
            raise ValueError("Loaded tensor has an unexpected shape.")
    except Exception as e:
        print(f"torch.load() failed or shape mismatch: {e}, using load_fhat() instead.")
        fhat = load_fhat(watermark_image_path, 8, device)
        watermark_image = var.var_decoder(fhat)

    var.save_image(watermark_image,"tes.png") 