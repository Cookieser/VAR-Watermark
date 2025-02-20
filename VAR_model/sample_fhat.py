
import argparse
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

parser = argparse.ArgumentParser(description="Set MODEL_DEPTH, sample size, and batch size.")
parser.add_argument('--model_depth', type=int, choices=[16, 20, 24, 30], default=24,
                    help="Specify the MODEL_DEPTH. Choices are 16, 20, 24, 30.")
parser.add_argument('--batch_size', type=int, default=10, 
                    help="Specify the batch size for sampling.")
parser.add_argument('--seed', type=int, default=0, 
                    help="Set the random seed for reproducibility.")
parser.add_argument('--total_samples', type=int, default=10, 
                    help="Specify the total number of samples to generate.")
parser.add_argument('--save_path', type=str, required=True,
                    help="Path to save the generated samples. This argument is required.")
parser.add_argument('--device', type=str, required=True,help="Use the device.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device



B = args.batch_size
total_samples = args.total_samples 
save_path = args.save_path  

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

from models import VQVAE, build_vae_var

MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30}, "Invalid MODEL_DEPTH!"


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda'

if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )


# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')



# set args
seed = args.seed #@param {type:"number"}
torch.manual_seed(seed)
#num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}

more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

os.makedirs(save_path, exist_ok=True)
generated_count = 0

while generated_count < total_samples:
    current_batch_size = min(B, total_samples - generated_count)
# sample
    class_labels = random.choices(range(1000), k=current_batch_size)
    print("class_labels (with duplicates):", class_labels)

    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            #recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
            f_hats,_ = var.autoregressive_infer_cfg(B=current_batch_size, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

    tensor = f_hats[-1]
   

    for i in range(tensor.size(0)):  
        single_tensor = tensor[i].squeeze(0)  # [32, 16, 16]
        file_name = osp.join(save_path, f"image{generated_count + i}.pt")  
        torch.save(single_tensor, file_name)  
        print(f"Saved image tensor {generated_count + i} with shape {single_tensor.shape} to {file_name}")

    generated_count += current_batch_size  

print(f"All {total_samples} samples have been saved to {save_path}")