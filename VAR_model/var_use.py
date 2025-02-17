import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
from PIL import Image as PImage, Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class VarTool:
    def __init__(self, model_depth=24, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), device='cuda',tf32=True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_depth = model_depth
        self.patch_nums = patch_nums
        self.seed = 1
        self.vqvae, self.var = self._load_models()
        self._set_seed()
        self._configure_tf32(tf32)
        

    def _set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
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

    '''
    From image to f, and from f to fhat(codebook value)
    '''
    def var_encoder(self, image):

        image = image.to(self.device)
        assert image.shape[1:] == (3, 256, 256), f"Expected shape (*, 3, 256, 256), but got {image.shape}"
        image = image * 2 - 1
        
        f = self.vqvae.quant_conv(self.vqvae.encoder(image))
        fhat = self.vqvae.quantize.f_to_idxBl_or_fhat(f, to_fhat=True)[-1]
        assert fhat.shape[1:] == (32, 16, 16), f"Expected shape (*, 32, 16, 16), but got {fhat.shape}"
        return fhat



    '''
    From fhat(codebook value) to image 
    '''

    def var_decoder(self,fhat):
        fhat = fhat.to(self.device) 
        assert fhat.shape[1:] == (32, 16, 16), f"Expected shape (*, 32, 16, 16), but got {fhat.shape}"

        recon_B3HW = self.vqvae.decoder(self.vqvae.post_quant_conv(fhat)).clamp(-1, 1)

        recon_B3HW = (recon_B3HW + 1) * 0.5
        assert recon_B3HW.shape[1:] == (3, 256, 256), f"Expected shape (*, 3, 256, 256), but got {recon_B3HW.shape}"
        return recon_B3HW 

    

    def show_images_from_batch_embedding(self, f_hats, output_path):
        batch_size = f_hats.shape[0]
        recon_B3HW = self.vqvae.fhat_to_img(f_hats).add_(1).mul_(0.5)
        chw = torchvision.utils.make_grid(recon_B3HW, nrow=batch_size, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        chw.save(output_path)
        print(f"Image saved to {output_path}")


    
    def generate_form_labels(self,class_labels):
        cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
        more_smooth = False # True for more smooth output
        label_B: torch.LongTensor = torch.tensor(class_labels, device=self.device)
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                f_hats,hs = self.var.autoregressive_infer_cfg(B=len(class_labels), label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=self.seed, more_smooth=more_smooth)
        return f_hats,hs

    
    def save_image(self,recon_B3HW,output="temp.png"):  
        chw = torchvision.utils.make_grid(recon_B3HW, nrow=recon_B3HW.shape[0], padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        chw.save(output)
        display(chw)
        print(f"Image({recon_B3HW.size()}) saved to {output}")

    
    def image_to_f(self,image):
        image = image.to(self.device)
        assert image.shape[1:] == (3, 256, 256), f"Expected shape (*, 3, 256, 256), but got {image.shape}"
        image = image * 2 - 1
        f = self.vqvae.quant_conv(self.vqvae.encoder(image))
        print(f.shape)
        #assert fhat.shape[1:] == (32, 16, 16), f"Expected shape (*, 32, 16, 16), but got {fhat.shape}"
        return f



    def f_to_image(self,f):
        
        f = f.to(self.device) 
        fhat = self.vqvae.quantize.f_to_idxBl_or_fhat(f, to_fhat=True)[-1]

        assert fhat.shape[1:] == (32, 16, 16), f"Expected shape (*, 32, 16, 16), but got {fhat.shape}"

        recon_B3HW = self.vqvae.decoder(self.vqvae.post_quant_conv(fhat)).clamp(-1, 1)

        recon_B3HW = (recon_B3HW + 1) * 0.5
        assert recon_B3HW.shape[1:] == (3, 256, 256), f"Expected shape (*, 3, 256, 256), but got {recon_B3HW.shape}"
        return recon_B3HW 

    







    