# import os
# import os.path as osp
# import torch, torchvision
# import random
# import numpy as np
# import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
# import torch
# import torch.nn
# from PIL import Image

# import os

# import torchvision.transforms.functional as TF
# import torchvision.transforms as transforms
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
# setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

# from models import VQVAE, build_vae_var

# MODEL_DEPTH = 24
# assert MODEL_DEPTH in {16, 20, 24, 30}, "Invalid MODEL_DEPTH!"


# # download checkpoint
# hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
# vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
# if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
# if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# # build vae, var
# patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if 'vae' not in globals() or 'var' not in globals():
#     vae, var = build_vae_var(
#         V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
#         device=device, patch_nums=patch_nums,
#         num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
#     )


# # load checkpoints
# vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
# vae.eval(), var.eval()
# for p in vae.parameters(): p.requires_grad_(False)
# for p in var.parameters(): p.requires_grad_(False)
# print(f'prepare finished.')



# # set args
# seed = 1 #@param {type:"number"}
# torch.manual_seed(seed)
# #num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
# cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}

# more_smooth = False # True for more smooth output

# # seed
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # run faster
# tf32 = True
# torch.backends.cudnn.allow_tf32 = bool(tf32)
# torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
# torch.set_float32_matmul_precision('high' if tf32 else 'highest')



# vqvae = var.vae_proxy[0] 
# vqvae = vqvae.to(device)


# def show_images_from_batch_embedding(f_hats,output_path):
#     batch_size = f_hats.shape[0]
#     recon_B3HW = var.vae_proxy[0].fhat_to_img(f_hats).add_(1).mul_(0.5)
#     chw = torchvision.utils.make_grid(recon_B3HW, nrow=batch_size, padding=0, pad_value=1.0)
#     chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
#     chw = PImage.fromarray(chw.astype(np.uint8))
#     chw.save(output_path)
#     print(f"Image saved to {output_path}")


# def encoder(image):
#     transform = transforms.Compose([
#         transforms.ToTensor(),          
#         transforms.Normalize((0.5,), (0.5,))  
#     ])

#     image_tensor = transform(image)               
#     image_tensor = image_tensor.unsqueeze(0)      
#     image_tensor = image_tensor.to(device)

#     print("Image Tensor Shape:", image_tensor.shape) 

#     f = vqvae.encoder(image_tensor)

#     f = vqvae.quant_conv(f)

#     f  = vqvae.quantize.f_to_idxBl_or_fhat(f,True)[-1]
#     return f



# # embed_path = "/home/yw699/codes/VAR-Watermark/encoded_image.pt"
# # output_path ='image.png'

# def embed2image(embed_path,output_path)

#     f_hats = torch.load(embed_path,weights_only=True)
#     #input_tensor_original = input_tensor_original.unsqueeze(0)
#     #input_tensor_original = input_tensor_original[0:1]

#     print("Original Tensor Shape:", f_hats.shape)

#     device = next(var.vae_proxy[0].parameters()).device

#     f_hats = f_hats.to(device)


#     batch_size = f_hats.shape[0]
#     with torch.no_grad():
#         recon_B3HW = var.vae_proxy[0].fhat_to_img(f_hats).add_(1).mul_(0.5)
#         chw = torchvision.utils.make_grid(recon_B3HW, nrow=batch_size, padding=0, pad_value=1.0)
#         chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
#     chw = PImage.fromarray(chw.astype(np.uint8))
#     chw.save(output_path)
#     print(f"Image saved to {output_path}")
#     return recon_B3HW




# def image2embed(input_path)
#     image = Image.open(input_path).convert('RGB')  
#     f = encoder(image)

#     show_images_from_batch_embedding(f,"re-image.png")
#     f = f.squeeze(0)
#     print(f.shape)
#     torch.save(f, "f.pt") 
#     return f



import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
from PIL import Image as PImage, Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageEmbedding:
    def __init__(self, device='cuda', model_depth=24, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), tf32=True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_depth = model_depth
        self.patch_nums = patch_nums
        self.vqvae, self.var = self._load_models()
        self._set_seed()
        self._configure_tf32(tf32)

    def _set_seed(self, seed=1):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _configure_tf32(self, tf32):
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    def _load_models(self):
        # Download checkpoints if necessary
        hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
        vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{self.model_depth}.pth'
        if not osp.exists(vae_ckpt):
            os.system(f'wget {hf_home}/{vae_ckpt}')
        if not osp.exists(var_ckpt):
            os.system(f'wget {hf_home}/{var_ckpt}')

        from .models import build_vae_var
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=self.device, patch_nums=self.patch_nums,
            num_classes=1000, depth=self.model_depth, shared_aln=False,
        )

        vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
        vae.eval()
        var.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        for p in var.parameters():
            p.requires_grad_(False)

        print(f'Models loaded and prepared.')
        return vae, var

    def encoder(self, image,batch):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image_tensor = transform(image)
        #image_tensor = image_tensor.unsqueeze(0).to(self.device)
        image_tensor = image_tensor.view(3, 256, batch, 256).permute(2, 0, 1, 3).to(self.device)

        print("Image Tensor Shape:", image_tensor.shape)

        f = self.vqvae.encoder(image_tensor)
        f = self.vqvae.quant_conv(f)
        f = self.vqvae.quantize.f_to_idxBl_or_fhat(f, True)[-1]
        return f

    def show_images_from_batch_embedding(self, f_hats, output_path):
        batch_size = f_hats.shape[0]
        recon_B3HW = self.vqvae.fhat_to_img(f_hats).add_(1).mul_(0.5)
        chw = torchvision.utils.make_grid(recon_B3HW, nrow=batch_size, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        chw.save(output_path)
        print(f"Image saved to {output_path}")

    def embed_to_image(self, embed_path, output_path):
        f_hats = torch.load(embed_path, map_location=self.device,weights_only=True)

        print("Embedding Tensor Shape:", f_hats.shape)

        with torch.no_grad():
            recon_B3HW = self.vqvae.fhat_to_img(f_hats).add_(1).mul_(0.5)
            chw = torchvision.utils.make_grid(recon_B3HW, nrow=f_hats.shape[0], padding=0, pad_value=1.0)
            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()

        chw = PImage.fromarray(chw.astype(np.uint8))
        chw.save(output_path)
        print(f"Image saved to {output_path}")
        return recon_B3HW

    def image_to_embed(self, input_path, output_path,batch):
        image = Image.open(input_path).convert('RGB')
        print(image.size)
        f = self.encoder(image,batch)
        torch.save(f, output_path)
        print(f"Embedding saved to {output_path}")
        return f


    