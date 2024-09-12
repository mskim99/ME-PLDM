import sys

sys.path.extend(['.', 'src'])
import torch
from einops import rearrange

import os
import nibabel as nib

import argparse
from omegaconf import OmegaConf
from models.ema import LitEma

from models.autoencoder.autoencoder_spade import ViTAutoencoder_SPADE
from models.ddpm.unet_mask_3d import UNetModel, DiffusionWrapper
from losses.ddpm_mask import DDPM

import glob

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

parser = argparse.ArgumentParser()
parser.add_argument('--first_model_src', type=str, default='', help='the path of pretrained model (source)')
parser.add_argument('--first_model_trg', type=str, default='', help='the path of pretrained model (target)')

# for GAN resume
parser.add_argument('--first_stage_folder', type=str, default='', help='the folder of first stage experiment before GAN')
parser.add_argument('--diffusion_config', type=str, default='configs/latent-diffusion/ucf101-ldm-kl-3_res128.yaml')

args = parser.parse_args()
config = OmegaConf.load(args.diffusion_config)
args.unetconfig = config.model.params.unet_config

rank = 0
_ema_ckpt_num = 22000

device = torch.device('cuda', rank)

fs_src_model = ViTAutoencoder_SPADE(embed_dim=16, ch_mult=(1, 2, 4, 8)).to(device)
fs_trg_model = ViTAutoencoder_SPADE(embed_dim=16, ch_mult=(1, 2, 4, 8)).to(device)

first_stage_model_src_ckpt = torch.load(args.first_model_src, map_location='cuda:0')
first_stage_model_trg_ckpt = torch.load(args.first_model_trg, map_location='cuda:0')
fs_src_model.load_state_dict(first_stage_model_src_ckpt)
fs_trg_model.load_state_dict(first_stage_model_trg_ckpt)
del first_stage_model_src_ckpt
del first_stage_model_trg_ckpt

fs_src_model.eval()
fs_trg_model.eval()

unet = UNetModel(**args.unetconfig)
ema_model = DiffusionWrapper(unet).to(device)

diffusion_model = DDPM(ema_model,
                       channels=ema_model.diffusion_model.in_channels,
                       image_size=ema_model.diffusion_model.image_size,
                       sampling_timesteps=100,
                       w=0.).to(device)

diffusion_model.eval()

if os.path.exists(f'/data/jionkim/PVDM_recon/results/inference/ema_model_' + str(_ema_ckpt_num) + '.pth'):
    ema_model_ckpt = torch.load(f'/data/jionkim/PVDM_recon/results/inference/ema_model_' + str(_ema_ckpt_num) + '.pth',
                                map_location='cuda:0')
    ema_model.load_state_dict(ema_model_ckpt)
    ema = LitEma(ema_model)
    ema.num_updates = torch.tensor(11200, dtype=torch.int)
    print('Diffusion model loaded')

    del ema_model_ckpt

else:
    raise FileNotFoundError("Diffusion model cannot be loaded")

torch.cuda.empty_cache()
with torch.no_grad():
    for i in range(0, 9):

        idx_cond = torch.tensor([i]).to(device)
        s_p = nib.load('/data/jionkim/SYNTHRAD2023_brain_res_128_s_16_pd_2/ct/' + str(i) + '/179_' + str(i).zfill(4) + '_s_16.nii.gz').get_fdata()
        s_g_p = nib.load('/data/jionkim/SYNTHRAD2023_brain_res_128_s_16_pd_2/ct_grad/' + str(i) + '/179_' + str(i).zfill(4) + '_s_16.nii.gz').get_fdata()
        s_p = torch.tensor(s_p).to(device)
        s_g_p = torch.tensor(s_g_p).to(device)
        s_p = rearrange(s_p / 255. + 1e-8, '(b t w) h c -> b t c h w', b=1, t=1).float()
        s_g_p = rearrange(s_g_p + 1e-8, '(b t w) h c -> b t c h w', b=1, t=1).float()

        s_p_concat = torch.cat([s_p, s_g_p], dim=1)

        z_s = fs_src_model.extract(s_p_concat, idx_cond).detach()
        z = diffusion_model.sample_3d(batch_size=1, channels=16, idx_cond=idx_cond, source=z_s)
        z = z.view(1, 16, 2, 16, 16)

        fake = fs_trg_model.decode_from_sample(z, cond=idx_cond).clamp(0, 1).cpu()
        fake = rearrange(fake, 'b t c h w -> b t h w c') * 255.
        fake = fake.type(torch.uint8).cpu().numpy()
        fake = fake.squeeze()
        fake = fake.swapaxes(0, 1)

        real = s_p.type(torch.uint8).cpu().numpy()
        real = real.squeeze()
        real = real.swapaxes(0, 2)

        fake_nii = nib.Nifti1Image(fake, None)
        real_nii = nib.Nifti1Image(real, None)
        nib.save(fake_nii, f'/data/jionkim/PVDM_recon/results/inference/generated_{i}.nii.gz')
        nib.save(real_nii, f'/data/jionkim/PVDM_recon/results/inference/real_{i}.nii.gz')

        del s_p, s_g_p, s_p_concat, z_s, z
        torch.cuda.empty_cache()

print('Inference finished')
