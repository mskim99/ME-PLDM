import os

import torch

from tools.trainer_cond_mask import latentDDPM
from tools.dataloader import get_loaders
from models.autoencoder.autoencoder_spade import ViTAutoencoder_SPADE
from models.ddpm.unet_mask_3d import UNetModel, DiffusionWrapper
from losses.ddpm_mask import DDPM

import copy
from utils import file_name, Logger, download

#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.
    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.
    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

#----------------------------------------------------------------------------

def diffusion(rank, args):
    device = torch.device('cuda', rank)

    temp_dir = './'
    '''
    if args.n_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.n_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.n_gpus)
            '''
    # Init torch_utils.
    sync_device = torch.device('cuda', rank) # if args.n_gpus > 1 else None
    # init_multiprocessing(rank=rank, sync_device=sync_device)

    """ ROOT DIRECTORY """
    # if rank == 0:
    fn = file_name(args)
    logger = Logger(fn, ask=False)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')
    rootdir = logger.logdir
    # else:
        # logger = None

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    """ Get Image """
    # if rank == 0:
    log_(f"Loading dataset {args.data} with resolution {args.res}")
    train_loader, _, test_loader = get_loaders(args.data, args.res, args.timesteps, False, args.batch_size, args.n_gpus, args.seed, args.cond_model, direction = ['x','y','z'][args.direction])

    if args.data == 'SKY':
        cond_prob = 0.2
    else:
        cond_prob = 0.3

    """ Get Model """
    # if rank == 0:
    log_(f"Generating model")

    torch.cuda.set_device(rank)
    first_stage_model_src = ViTAutoencoder_SPADE(embed_dim=16).to(device)
    first_stage_model_trg = ViTAutoencoder_SPADE(embed_dim=16).to(device)

    first_stage_model_src_ckpt = torch.load(args.first_model_src, map_location=f'cuda:{rank}')
    first_stage_model_trg_ckpt = torch.load(args.first_model_trg, map_location=f'cuda:{rank}')
    first_stage_model_src.load_state_dict(first_stage_model_src_ckpt)
    first_stage_model_trg.load_state_dict(first_stage_model_trg_ckpt)
    del first_stage_model_src_ckpt
    del first_stage_model_trg_ckpt

    unet = UNetModel(**args.unetconfig)
    model = DiffusionWrapper(unet).to(device)

    # if os.path.exists(rootdir + f'model_last.pth'): #? 'model_last.pth'
    #     model_ckpt = torch.load(rootdir + f'model_last.pth', map_location=f'cuda:{rank}')
    if os.path.exists(rootdir + f'model_61000.pth'): #? 'model_last.pth'
        model_ckpt = torch.load(rootdir + f'model_61000.pth', map_location=f'cuda:{rank}')
        model.load_state_dict(model_ckpt)
        ema_model = copy.deepcopy(model)
        print('Model loaded')

        del model_ckpt

    else:
        # if rank == 0:
        torch.save(model.state_dict(), rootdir + f'net_init.pth')
        ema_model = None

    criterion = DDPM(model, channels=args.unetconfig.in_channels,
                            image_size=args.unetconfig.image_size,
                            linear_start=args.ddpmconfig.linear_start,
                            linear_end=args.ddpmconfig.linear_end,
                            log_every_t=args.ddpmconfig.log_every_t,
                            w=args.ddpmconfig.w,
                            #timesteps=args.ddpmconfig.sampling_steps,
                            sampling_timesteps=args.ddpmconfig.sampling_steps,
                            ).to(device)

    if args.scale_lr:
        args.lr *= args.batch_size

    opt= torch.optim.AdamW(model.parameters(), lr=args.lr)

    latentDDPM(rank, first_stage_model_src, first_stage_model_trg, model, opt, criterion, train_loader, test_loader, ema_model, logger, steps=args.ddpmconfig.sampling_steps)

    # if rank == 0:
    torch.save(model.state_dict(), rootdir + f'net_meta.pth')
