import os
import sys

sys.path.extend([sys.path[0][:-4], '/app'])

import time
import copy

import torch
from torch.cuda.amp import GradScaler, autocast

from utils import AverageMeter
from evals.eval_cond import save_image_ddpm_mask, z_list_gen_src, save_image_cond_mask
from models.ema import LitEma
from einops import rearrange

import nibabel as nib

def latentDDPM(rank, fs_src_model, fs_trg_model, model, opt, criterion, train_loader, test_loader, ema_model=None, logger=None, steps=100):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    rootdir = logger.logdir
    device = torch.device('cuda', rank)

    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    losses['dist_loss'] = AverageMeter()
    losses['total_loss'] = AverageMeter()
    check = time.time()

    l1_loss = torch.nn.L1Loss()

    if ema_model == None:
        ema_model = copy.deepcopy(model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200, dtype=torch.int)
        ema_model.eval()

    fs_src_model.eval()
    fs_trg_model.eval()
    model.train()

    for it, (src, src_grad, dst, dst_grad, cond) in enumerate(train_loader):

        it = it + 61000

        z_list_real = []
        diff_loss = 0.
        # z_prev = torch.zeros([16, 16, 4, 16, 16]).cuda()
        #print(dst.__len__())
        for x_idx in range (0, dst.__len__()): #0 to slice num
            d_p = dst[x_idx].to(device)
            d_g_p = dst_grad[x_idx].to(device)
            s_p = src[x_idx].to(device)
            s_g_p = src_grad[x_idx].to(device)
            d_p = rearrange(d_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            d_g_p = rearrange(d_g_p + 1e-8, 'b t c h w -> b c t h w').float()
            s_p = rearrange(s_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            s_g_p = rearrange(s_g_p + 1e-8, 'b t c h w -> b c t h w').float()

            d_p_concat = torch.cat([d_p, d_g_p], dim=1)
            s_p_concat = torch.cat([s_p, s_g_p], dim=1)

            cond_p = cond[x_idx].to(device)

            # conditional free guidance training
            model.zero_grad()

            with autocast():
                with torch.no_grad():
                    z_d = fs_trg_model.extract(d_p_concat, cond_p)
                    z_s = fs_src_model.extract(s_p_concat, cond_p)
                    z_list_real.append(z_d.detach())
            (diff_loss_part, t), loss_dict = criterion(z_d.float(), cond=cond_p.float(), c_s=z_s.float(), c_prev=None)
            # (diff_loss, t), loss_dict = criterion(z_d.float(), cond=cond_p.float(), c_m=None)

            # diff_loss.backward()
            # opt.step()

            # losses['diffusion_loss'].update(diff_loss.item(), 1)

            diff_loss += diff_loss_part

            # z_prev = z_d.clone()

        diff_loss = diff_loss / float(dst.__len__())
        
        if it % 25 == 0:
            dist_loss = 0.
            z_list_fake = z_list_gen_src(rank, ema_model, fs_src_model, src, src_grad)

            for i in range(0, dst.__len__()-1):
                z_real = torch.reshape(z_list_real[i].mean(axis=0), (1, 16, 512))
                dist_loss += l1_loss(z_list_fake[i], z_real)

            dist_loss = dist_loss / float(dst.__len__())

        if it < 10000:
            weight = float(it) * 1. / 10000.
        else:
            weight = 1.
        
        total_loss = (weight * diff_loss + (1. - weight) * dist_loss)
        # total_loss = (diff_loss + dist_loss)
        total_loss.backward()
        opt.step()

        losses['diffusion_loss'].update(diff_loss.item(), 1)
        # losses['cont_loss'].update(cont_loss.item(), 1)
        losses['dist_loss'].update(dist_loss.item(), 1)
        losses['total_loss'].update(total_loss.item(), 1)

        # ema model
        if it % 5 == 0:
            ema(model)

        if it % 125 == 0:
            # if logger is not None and rank == 0:
            if logger is not None:
                logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)
                logger.scalar_summary('train/dist_loss', losses['dist_loss'].average, it)
                logger.scalar_summary('train/total_loss', losses['total_loss'].average, it)

                # log_('[Time %.3f] [Diffusion %f]' %
                     # (time.time() - check, losses['diffusion_loss'].average))

                # log_('[Time %.3f] [Diffusion %f] [Dist %f]' %
                     # (time.time() - check, losses['diffusion_loss'].average, losses['dist_loss'].average))

                log_('[Time %.3f] [Diffusion %f] [Dist %f] [Weight %f]' %
                     (time.time() - check, losses['diffusion_loss'].average, losses['dist_loss'].average, weight))

                losses = dict()
                losses['diffusion_loss'] = AverageMeter()
                # losses['cont_loss'] = AverageMeter()
                losses['dist_loss'] = AverageMeter()
                losses['total_loss'] = AverageMeter()

        if it % 500 == 0:
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')
            ema.copy_to(ema_model)
            torch.save(ema_model.state_dict(), rootdir + f'ema_model_{it}.pth')
            print("start")
            save_image_ddpm_mask(rank, ema_model, fs_src_model, fs_trg_model, it, test_loader, logger, steps)
            print("end")


def first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # if rank == 0:
    rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['L1_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    l1_criterion = torch.nn.L1Loss()

    if fp:
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, 'scaler.pth')))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, 'scaler_d.pth')))
        except:
            print("Fail to load scalers. Start from initial point.")

    model.train()
    disc_start = criterion.discriminator_iter_start

    torch.cuda.empty_cache()
    for it, (x, g, cond, _) in enumerate(train_loader):

        it = it + 24750

        torch.cuda.empty_cache()
        for x_idx in range(0, x.__len__()):
            batch_size = x[x_idx].size(0)
            x_p = x[x_idx].to(device)
            g_p = g[x_idx].to(device)

            x_p = rearrange(x_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            g_p = rearrange(g_p + 1e-8, 'b t c h w -> b c t h w').float()

            x_p_concat = torch.cat([x_p, g_p], dim=1)
            #print(x_p_concat.shape)
            cond_p = cond[x_idx].to(device)

            if not disc_opt:
                with autocast():
                    # print(x_p_concat.shape)
                    # print(cond_p)
                    # exit(0)
                    x_tilde_ra, vq_loss = model(x_p_concat, cond_p)
                    model.zero_grad()

                    l1_loss = l1_criterion(x_p, x_tilde_ra)
                    l1_loss = 10. * l1_loss / accum_iter

                    total_loss = l1_loss

                scaler.scale(total_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    scaler.step(opt)
                    scaler.update()

                # print(losses)
                losses['L1_loss'].update(l1_loss.item(), 1)

            else:
                if it % accum_iter == 0:
                    criterion.zero_grad()

                with autocast():
                    with torch.no_grad():
                        x_tilde, vq_loss = model(x_p)
                    d_loss = criterion(vq_loss, x_p,
                                       rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                       optimizer_idx=1,
                                       global_step=it)
                    d_loss = d_loss / accum_iter

                scaler_d.scale(d_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler_d.unscale_(d_opt)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(criterion.discriminator_2d.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(criterion.discriminator_3d.parameters(), 1.0)

                    scaler_d.step(d_opt)
                    scaler_d.update()

                losses['d_loss'].update(d_loss.item() * 3, 1)

            if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
                if disc_opt:
                    disc_opt = False
                else:
                    disc_opt = True

        if it % 1000 == 0:
            psnr = 0.

            if logger is not None:
                logger.scalar_summary('train/L1_loss', losses['L1_loss'].average, it)

                log_('[Time %.3f] [L1Loss %f]' %
                     (time.time() - check, losses['L1_loss'].average))

                torch.save(model.state_dict(), rootdir + f'model_last.pth')
                torch.save(criterion.state_dict(), rootdir + f'loss_last.pth')
                torch.save(opt.state_dict(), rootdir + f'opt.pth')
                torch.save(d_opt.state_dict(), rootdir + f'd_opt.pth')
                torch.save(scaler.state_dict(), rootdir + f'scaler.pth')
                torch.save(scaler_d.state_dict(), rootdir + f'scaler_d.pth')

            losses = dict()
            losses['L1_loss'] = AverageMeter()

        if it % 1000 == 0:
            torch.cuda.empty_cache()
            save_image_cond_mask(rank, model, test_loader, it, logger)
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')