import sys

sys.path.extend(['.', 'src'])
import torch
from utils import AverageMeter
from einops import rearrange
from losses.ddpm_mask import DDPM

import os
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

import math

def z_list_gen(rank, ema_model):

    device = torch.device('cuda', rank)
    l1_loss = torch.nn.L1Loss()
    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=10,
                           w=0.).to(device)

    z_list = []
    # cont_loss_value = 0.
    for idx in range(0, 9):
        idx_cond = torch.tensor([idx]).to(device)
        z = diffusion_model.sample_3d(batch_size=1, channels=32, idx_cond=idx_cond, tqdm=False)
        z_list.append(z.detach())

    return z_list


def z_list_gen_src(rank, ema_model, first_stage_model, src, src_grad):

    device = torch.device('cuda', rank)
    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=10,
                           w=0.).to(device)

    z_list = []

    for idx in range(0, src.__len__()):
        s_p = src[idx].to(device)
        s_g_p = src_grad[idx].to(device)
        s_p = s_p[0]
        s_g_p = s_g_p[0]
        s_p = rearrange(s_p / 255. + 1e-8, '(b t) c h w -> b c t h w', b=1).float()
        s_g_p = rearrange(s_g_p + 1e-8, '(b t) c h w -> b c t h w', b=1).float()

        s_p_concat = torch.cat([s_p, s_g_p], dim=1)

        cond_p = torch.tensor(idx).to(device)
        cond_p = cond_p.unsqueeze(0)

        z_s = first_stage_model.extract(s_p_concat, cond_p)

        idx_cond = torch.tensor([idx]).to(device)
        z = diffusion_model.sample_3d(batch_size=1, channels=16, idx_cond=idx_cond, source=z_s, tqdm=False)
        z_list.append(z.detach())


    return z_list


def test_psnr(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()

    model.eval()
    with torch.no_grad():
        for n, (x, cond, _) in enumerate(loader):

            if n > 12:
                break

            # Store previous partitions
            x_p_prev = torch.zeros(x[0].shape).cuda()

            for x_idx in range(0, x.__len__()):
                batch_size = x[x_idx].size(0)
                x_p = x[x_idx].float().to(device)
                cond_p = cond[x_idx].to(device)
                # print(x_p.shape)
                # print(x_p_prev.shape)
                x_p_concat = torch.cat([x_p / 127.5 - 1, x_p_prev], dim=2)
                # print(x_p_concat.shape)
                recon, _ = model(rearrange(x_p_concat, 'b t c h w -> b c t h w'), cond_p)
                # recon, _ = model(rearrange(x_p / 127.5 - 1, 'b t c h w -> b c t h w'), cond_p)

                x_p_prev = x_p.clone().detach() / 127.5 - 1

                x_p = x_p.view(batch_size, -1)
                recon = recon.view(batch_size, -1)

                mse = ((x_p * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
                psnr = (-10 * torch.log10(mse)).mean()

                losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average


def test_psnr_mask(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()

    model.eval()
    with torch.no_grad():
        for n, (_, dst, _, cond) in enumerate(loader):

            if n > 12:
                break

            for x_idx in range(0, dst.__len__()):
                batch_size = dst[x_idx].size(0)
                dst_p = dst[x_idx].float().to(device)
                dst_p = rearrange(dst_p / 127.5 - 1, 'b t c h w -> b c t h w').float()
                cond_p = cond[x_idx].to(device)
                recon, _ = model(dst_p, cond_p)

                # x_p_prev = x_p.clone().detach() / 127.5 - 1

                dst_p = dst_p.view(batch_size, -1)
                recon = recon.view(batch_size, -1)

                mse = ((dst_p * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
                psnr = (-10 * torch.log10(mse)).mean()

                losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average


def save_image_ddpm(rank, ema_model, decoder, it, logger=None, idx_cond=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)

    with torch.no_grad():
        z = diffusion_model.sample(batch_size=16, idx_cond=idx_cond)
        # print(z.shape)
        fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
        # fake = decoder.decode_from_sample(z).clamp(-1, 1).cpu()
        # print(fake.shape)
        fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=16)) * 127.5
        # print(fake.shape)
        fake = fake[0].type(torch.uint8).cpu().numpy()
        # print(fake.shape)
        # for s_i in range (0, 8):
        fake = fake.squeeze()
        # print(fake.shape)
        fake = fake.swapaxes(0, 2)
        # print(fake.shape)
        fake_nii = nib.Nifti1Image(fake, None)
        nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))


def save_image_cond(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    model.eval()
    with torch.no_grad():
        for _, (real, cond, idx) in enumerate(loader):

            # Store previous partitions
            real_p_prev = torch.zeros(real[0].shape).to(device)

            for r_idx in range(0, real.__len__()):
                real_p = real[r_idx].float().to(device)
                # real_p = real_p / 127.5 - 1
                cond_p = cond[r_idx].to(device)
                real_p_concat = torch.cat([real_p / 127.5 - 1, real_p_prev], dim=2)
                fake, _ = model(rearrange(real_p_concat, 'b t c h w -> b c t h w'), cond_p)
                # fake, _ = model(rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w'), cond_p)

                fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real_p.size(0))

                real_p_prev = real_p.clone().detach() / 127.5 - 1

                real_p = real_p.type(torch.uint8).cpu().numpy()
                fake = fake.type(torch.uint8).cpu().numpy()

                # real_p = real_p.cpu().numpy()
                # fake = fake.cpu().numpy()

                real_p = real_p.squeeze()
                fake = fake.squeeze()

                real_p = real_p.swapaxes(1, 3)
                fake = fake.swapaxes(1, 3)

                real_nii = nib.Nifti1Image(real_p[1, :, :, :], None)
                fake_nii = nib.Nifti1Image(fake[1, :, :, :], None)

                nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{cond[r_idx][0]}.nii.gz'))
                nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{cond[r_idx][0]}.nii.gz'))

            print('eval finished')
            return

def save_image_cond_mask(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    model.eval()
    with torch.no_grad():
        for _, (_, dst, _, cond) in enumerate(loader):

            # Store previous partitions
            # real_p_prev = torch.zeros(real[0].shape).to(device)

            for r_idx in range(0, dst.__len__()):
                real_p = dst[r_idx].float().to(device)
                cond_p = cond[r_idx].to(device)
                real_p = rearrange(real_p / 127.5 - 1, 'b t c h w -> b c t h w')
                fake, _ = model(real_p, cond_p)

                fake = rearrange(fake.clamp(0, 1) * 255., '(b t) c h w -> b t h w c', b=real_p.size(0))
                # fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, 'b c t h w -> b t c h w', b=real_p.size(0))
                real_p = real_p.clamp(0, 1) * 255.

                real_p = real_p.type(torch.uint8).cpu().numpy()
                fake = fake.type(torch.uint8).cpu().numpy()

                real_p = real_p.squeeze()
                fake = fake.squeeze()

                # print(real_p.shape)
                # print(fake.shape)

                real_p = real_p.swapaxes(1, 3)
                fake = fake.swapaxes(1, 3)

                real_nii = nib.Nifti1Image(real_p[1, :, :, :], None)
                fake_nii = nib.Nifti1Image(fake[1, :, :, :], None)

                nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{cond[r_idx][0]}.nii.gz'))
                nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{cond[r_idx][0]}.nii.gz'))

            print('eval finished')
            return

def save_image_ddpm_cond(rank, ema_model, decoder, it, logger=None):
    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)


    with torch.no_grad():

        # init_context = None
        # z_prev = None
        for idx in range(0, 9):
            idx_cond = torch.tensor([idx]).to(device)
            # z = diffusion_model.sample(batch_size=1, idx_cond=idx_cond, init_context=init_context)
            z = diffusion_model.sample(batch_size=1, idx_cond=idx_cond)
            # print('eval')
            # print(z.shape)
            # if z_prev is None:
                # z_prev = z.clone()
            # z_concat = torch.cat([z, z_prev], dim=1)
            # print(z_concat.shape)
            # print(z.shape)
            # init_context = z
            # fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
            fake = decoder.decode_from_sample(z, cond=idx_cond).clamp(-1, 1).cpu()
            fake = (1 + rearrange(fake, '(b t) c h w -> b t h w c', b=1)) * 127.5
            fake = fake.type(torch.uint8).cpu().numpy()
            fake = fake.squeeze()
            fake = fake.swapaxes(0, 2)
            fake_nii = nib.Nifti1Image(fake, None)
            nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))
            # z_prev = z.clone()


def save_image_ddpm_mask(rank, ema_model, fs_src_model, fs_trg_model, it, loader, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    device = torch.device('cuda', rank)

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)

    fs_src_model.eval()
    fs_trg_model.eval()


    metrics = dict()
    metrics['MAE'] = AverageMeter()
    metrics['PSNR'] = AverageMeter()
    metrics['SSIM'] = AverageMeter()

    with torch.no_grad():

        for num, (src, src_grad, dst, _, _, cond) in enumerate(loader):

            for idx in range(0, src.__len__()):
                idx_cond = torch.tensor([idx]).to(device)
                s_p = src[idx].to(device)
                s_g_p = src_grad[idx].to(device)
                s_p = s_p[0]
                s_g_p = s_g_p[0]
                s_p = rearrange(s_p / 255. + 1e-8, '(b t) c h w -> b c t h w', b=1).float()
                s_g_p = rearrange(s_g_p + 1e-8, '(b t) c h w -> b c t h w', b=1).float()

                s_p_concat = torch.cat([s_p, s_g_p], dim=1)

                cond_p = cond[idx].to(device)
                cond_p = cond_p[0].unsqueeze(0)

                z_s = fs_src_model.extract(s_p_concat, cond_p)
                z = diffusion_model.sample_3d(batch_size=1, channels=16, idx_cond=idx_cond, source=z_s)
                z = z.view(1, 16, 2, 16, 16)

                fake = fs_trg_model.decode_from_sample(z, cond=idx_cond).clamp(0, 1).cpu()
                fake = rearrange(fake, 'b t c h w -> b t h w c') * 255.
                fake = fake.type(torch.uint8).cpu().numpy()
                fake = fake.squeeze()
                fake = fake.swapaxes(0, 1)

                # real = dst[idx].type(torch.uint8).cpu().numpy()
                real = dst[idx][0].type(torch.uint8).cpu().numpy()
                real = real.squeeze()
                real = real.swapaxes(0, 2)

                # fake_eval = fake.reshape(fake.shape[0], -1)
                # real_eval = real.reshape(real.shape[0], -1)

                # print(fake.shape)
                # print(real.shape)

                mae_value = math.sqrt(mean_squared_error(fake, real))
                psnr_value = peak_signal_noise_ratio(fake, real)
                ssim_value = structural_similarity(fake, real)
                metrics['MAE'].update(mae_value)
                metrics['PSNR'].update(psnr_value)
                metrics['SSIM'].update(ssim_value)

                fake_nii = nib.Nifti1Image(fake, None)
                real_nii = nib.Nifti1Image(real, None)
                nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{num}_{idx_cond[0]}.nii.gz'))
                # nib.save(fake_nii, os.path.join(logger.logdir, f'generated_{it}_{idx_cond[0]}.nii.gz'))
                nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{num}_{idx_cond[0]}.nii.gz'))
                # nib.save(real_nii, os.path.join(logger.logdir, f'real_{it}_{idx_cond[0]}.nii.gz'))

            log_('[EVALUATION] [MAE %f] [PSNR %f] [SSIM %f]' % (metrics['MAE'].average, metrics['PSNR'].average,
                                                                metrics['SSIM'].average))

            # if num >= 8:
                # print('Evaluation finished')
                # exit(0)
            break

    print('Evaluation finished')