import sys; sys.path.extend(['.'])

import os
import argparse

import torch
from omegaconf import OmegaConf

from exps.diffusion import diffusion
from exps.first_stage import first_stage

from utils import set_random_seed

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='main', help='experiment identifier')

""" Args about Data """
parser.add_argument('--data', type=str, default='UCF101')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--ds', type=int, default=4)

""" Args about Model """
parser.add_argument('--pretrain_config', type=str, default='configs/autoencoder/base.yaml')
parser.add_argument('--diffusion_config', type=str, default='configs/latent-diffusion/base.yaml')

# for GAN resume
parser.add_argument('--first_stage_folder', type=str, default='', help='the folder of first stage experiment before GAN')

# for diffusion model path specification
parser.add_argument('--first_model_src', type=str, default='', help='the path of pretrained model (source)')
parser.add_argument('--first_model_trg', type=str, default='', help='the path of pretrained model (target)')
parser.add_argument('--scale_lr', action='store_true')

#gpu setup
parser.add_argument('--gpu_num', type=int, default=-1, help='index number of gpu')
parser.add_argument('--direction', type=int, default=2, help='direction of image')

def main():
    """ Additional args ends here. """
    args = parser.parse_args()
    """ FIX THE RANDOMNESS """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # args.n_gpus = torch.cuda.device_count()
    args.n_gpus = 1
    print(args.direction)
    # init and save configs
    
    """ RUN THE EXP """
    if args.exp == 'ddpm':
        config = OmegaConf.load(args.diffusion_config)
        first_stage_config = OmegaConf.load(args.pretrain_config)

        args.unetconfig = config.model.params.unet_config
        args.lr         = config.model.base_learning_rate
        args.scheduler  = config.model.params.scheduler_config
        args.res        = first_stage_config.model.params.ddconfig.resolution
        args.timesteps  = first_stage_config.model.params.ddconfig.timesteps
        args.skip       = first_stage_config.model.params.ddconfig.skip
        args.ddconfig   = first_stage_config.model.params.ddconfig
        args.embed_dim  = first_stage_config.model.params.embed_dim
        args.ddpmconfig = config.model.params
        args.cond_model = config.model.cond_model

        diffusion(rank=2 if args.gpu_num < 0 else args.gpu_num, args=args)
        # else:
            # torch.multiprocessing.spawn(fn=diffusion, args=(args, ), nprocs=args.n_gpus)

    elif args.exp == 'first_stage':
        config = OmegaConf.load(args.pretrain_config)
        args.ddconfig   = config.model.params.ddconfig
        args.embed_dim  = config.model.params.embed_dim
        args.lossconfig = config.model.params.lossconfig
        args.lr         = config.model.base_learning_rate
        args.res        = config.model.params.ddconfig.resolution
        args.timesteps  = config.model.params.ddconfig.timesteps
        args.skip       = config.model.params.ddconfig.skip
        args.resume     = config.model.resume
        args.amp        = config.model.amp

        first_stage(rank=3 if args.gpu_num < 0 else args.gpu_num, args=args)

    else:
        raise ValueError("Unknown experiment.")

if __name__ == '__main__':
    main()