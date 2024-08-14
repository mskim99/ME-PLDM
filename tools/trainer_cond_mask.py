import os
import sys

sys.path.extend([sys.path[0][:-4], '/app'])

import time
import copy

import torch
from torch.cuda.amp import GradScaler, autocast

from utils import AverageMeter
from evals.eval_cond import test_psnr_mask, save_image_cond_mask, save_image_ddpm_mask, z_list_gen_src
from models.ema import LitEma
from einops import rearrange

import nibabel as nib

def latentDDPM(rank, fs_src_model, fs_trg_model, model, opt, criterion, train_loader, ema_model=None, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # if rank == 0:
    rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    # losses['cont_loss'] = AverageMeter()
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

    for it, (src, src_grad, dst, dst_grad, mask, cond) in enumerate(train_loader):

        # it = it + 14000

        z_list_real = []
        diff_loss = 0.
        for x_idx in range (0, dst.__len__()):

            d_p = dst[x_idx].to(device)
            d_g_p = dst_grad[x_idx].to(device)
            s_p = src[x_idx].to(device)
            s_g_p = src_grad[x_idx].to(device)
            m_p = mask[x_idx].to(device)
            d_p = rearrange(d_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            d_g_p = rearrange(d_g_p + 1e-8, 'b t c h w -> b c t h w').float()
            s_p = rearrange(s_p / 255. + 1e-8, 'b t c h w -> b c t h w').float()
            s_g_p = rearrange(s_g_p + 1e-8, 'b t c h w -> b c t h w').float()
            # m_p = rearrange(m_p + 1e-8, 'b t c h w -> b c t h w').float()

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
                    # print(z_d[0].unsqueeze(0).shape)
                    '''
                    v_d = fs_trg_model.decode_from_sample(z_d[0].unsqueeze(0), cond_p[0].unsqueeze(0)).detach().clamp(0, 1).cpu()
                    v_d = rearrange(v_d, 'b t c h w -> b t h w c') * 255.
                    v_d = v_d.type(torch.uint8).cpu().numpy()
                    v_d = v_d.squeeze()
                    fake_nii = nib.Nifti1Image(v_d, None)
                    nib.save(fake_nii, os.path.join(logger.logdir, f'decoded_{it}_{cond_p[0]}.nii.gz'))
                    '''
            (diff_loss_part, t), loss_dict = criterion(z_d.float(), cond=cond_p.float(), c_s=z_s.float(), c_m=None)
            # (loss, t), loss_dict = criterion(z_d.float(), cond=cond_p.float(), c_m=z_m.float())

            # diff_loss.backward()
            # opt.step()

            # losses['diffusion_loss'].update(diff_loss.item(), 1)

            diff_loss += diff_loss_part

        diff_loss = diff_loss / float(dst.__len__())
        
        if it % 25 == 0:
            cont_loss = 0.
            dist_loss = 0.
            z_list_fake = z_list_gen_src(rank, ema_model, fs_src_model, src, src_grad)
            '''
            for i in range(0, dst.__len__()-1):
                # value_fake = l1_loss(z_list_fake[i + 1], z_list_fake[i])
                # cont_loss += value_fake
                # value_real = l1_loss(z_list_real[i + 1], z_list_real[i])
                # cont_loss += abs(value_fake - value_real)
                z_real = torch.reshape(z_list_real[i][0,:,:,:], (1, 32, 64))
                dist_loss += l1_loss(z_list_fake[i], z_real)

            z_real = torch.reshape(z_list_real[dst.__len__()-1][0, :, :, :], (1, 32, 64))
            dist_loss += l1_loss(z_list_fake[dst.__len__()-1], z_real)

            # cont_loss = cont_loss / float(x.__len__()-1)
            dist_loss = dist_loss / float(dst.__len__())
            '''

            for i in range(0, dst.__len__()):

                z_fake = z_list_fake[i].view([32, 2, 16, 16]).unsqueeze(0).float()
                z_real = z_list_real[i][0].unsqueeze(0).float()

                v_d_fake = fs_trg_model.decode_from_sample(z_fake, cond[i][0].unsqueeze(0).to(device)).detach().clamp(0,1).cpu()
                v_d_real = fs_trg_model.decode_from_sample(z_real, cond[i][0].unsqueeze(0).to(device)).detach().clamp(0,1).cpu()
                '''
                v_d_real_out = rearrange(v_d_real, 'b t c h w -> b t h w c') * 255.
                v_d_real_out = v_d_real_out.type(torch.uint8).cpu().numpy()
                v_d_real_out = v_d_real_out.squeeze()
                real_nii = nib.Nifti1Image(v_d_real_out, None)
                nib.save(real_nii, os.path.join(logger.logdir, f'v_d_real_{it}_{cond[i][0]}.nii.gz'))

                v_d_fake_out = rearrange(v_d_fake, 'b t c h w -> b t h w c') * 255.
                v_d_fake_out = v_d_fake_out.type(torch.uint8).cpu().numpy()
                v_d_fake_out = v_d_fake_out.squeeze()
                fake_nii = nib.Nifti1Image(v_d_fake_out, None)
                nib.save(fake_nii, os.path.join(logger.logdir, f'v_d_fake_{it}_{cond[i][0]}.nii.gz'))
                '''
                dist_loss += l1_loss(v_d_fake, v_d_real)

            dist_loss = dist_loss / float(dst.__len__())

        total_loss = (diff_loss + dist_loss)
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
                # logger.scalar_summary('train/cont_loss', losses['cont_loss'].average, it)
                # logger.scalar_summary('train/dist_loss', losses['dist_loss'].average, it)
                # logger.scalar_summary('train/total_loss', losses['total_loss'].average, it)

                '''
                log_('[Time %.3f] [Diffusion %f]' %
                     (time.time() - check, losses['diffusion_loss'].average))
                '''

                log_('[Time %.3f] [Diffusion %f] [Dist %f]' %
                     (time.time() - check, losses['diffusion_loss'].average, losses['dist_loss'].average))

                losses = dict()
                losses['diffusion_loss'] = AverageMeter()
                # losses['cont_loss'] = AverageMeter()
                losses['dist_loss'] = AverageMeter()
                losses['total_loss'] = AverageMeter()

        if it % 250 == 0:
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')
            ema.copy_to(ema_model)
            torch.save(ema_model.state_dict(), rootdir + f'ema_model_{it}.pth')
            save_image_ddpm_mask(rank, ema_model, fs_trg_model, it, train_loader, logger)



def first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # if rank == 0:
    rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['ae_loss'] = AverageMeter()
    losses['L1_loss'] = AverageMeter()
    losses['d_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    l1_criterion = torch.nn.L1Loss()

    if fp:
        # print('fp')
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, 'scaler.pth')))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, 'scaler_d.pth')))

            scaler = scaler.to(device).to(torch.float64)
            scaler_d = scaler_d.to(device).to(torch.float64)
        except:
            print("Fail to load scalers. Start from initial point.")

    model.train()
    disc_start = criterion.discriminator_iter_start

    for it, (_, dst, _, cond) in enumerate(train_loader):

        it = it + 6250

        for x_idx in range (0, dst.__len__()):

            batch_size = dst[x_idx].size(0)
            dst_p = dst[x_idx].to(device)
            dst_p = rearrange(dst_p / 127.5 - 1, 'b t c h w -> b c t h w').float()

            cond_p = cond[x_idx].to(device)

            if not disc_opt:
                with autocast():
                    # x_tilde, vq_loss = model(x)
                    x_tilde, vq_loss = model(dst_p, cond_p)
                    x_tilde_ra = rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size)
                    if it % accum_iter == 0:
                        model.zero_grad()

                    ae_loss = criterion(vq_loss, dst_p, x_tilde_ra,
                                        optimizer_idx=0,
                                        global_step=it)
                    ae_loss = ae_loss / accum_iter

                    l1_loss = l1_criterion(dst_p, x_tilde_ra)
                    l1_loss = 10. * l1_loss / accum_iter

                    total_loss = ae_loss

                scaler.scale(total_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    scaler.step(opt)
                    scaler.update()

                # print(losses)
                losses['ae_loss'].update(ae_loss.item(), 1)
                losses['L1_loss'].update(l1_loss.item(), 1)

            else:
                if it % accum_iter == 0:
                    criterion.zero_grad()

                with autocast():
                    with torch.no_grad():
                        x_tilde, vq_loss = model(dst_p)
                    d_loss = criterion(vq_loss, dst_p,
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

        if it % 250 == 0:
            psnr = test_psnr_mask(rank, model, test_loader, it, logger)

            if logger is not None:
                logger.scalar_summary('train/ae_loss', losses['ae_loss'].average, it)
                logger.scalar_summary('train/L1_loss', losses['L1_loss'].average, it)
                logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                logger.scalar_summary('test/psnr', psnr, it)

                log_('[Time %.3f] [AELoss %f] [L1Loss %f] [DLoss %f] [PSNR %f]' %
                    (time.time() - check, losses['ae_loss'].average, losses['L1_loss'].average,
                    losses['d_loss'].average, psnr))

                torch.save(model.state_dict(), rootdir + f'model_last.pth')
                torch.save(criterion.state_dict(), rootdir + f'loss_last.pth')
                torch.save(opt.state_dict(), rootdir + f'opt.pth')
                torch.save(d_opt.state_dict(), rootdir + f'd_opt.pth')
                torch.save(scaler.state_dict(), rootdir + f'scaler.pth')
                torch.save(scaler_d.state_dict(), rootdir + f'scaler_d.pth')

            losses = dict()
            losses['ae_loss'] = AverageMeter()
            losses['L1_loss'] = AverageMeter()
            losses['d_loss'] = AverageMeter()

        if it % 1250 == 0:
            save_image_cond_mask(rank, model, test_loader, it, logger)
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')