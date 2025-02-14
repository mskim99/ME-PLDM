import os

import torch

from tools.trainer_cond_mask import first_stage_train
from tools.dataloader import get_loaders
from models.autoencoder.autoencoder_spade import ViTAutoencoder_SPADE
#from models.autoencoder.autoencoder_vit_cond_3d import ViTAutoencoder
from losses.perceptual import LPIPSWithDiscriminator

from utils import file_name, Logger

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

def first_stage(rank, args):
    device = torch.device('cuda', rank)
    torch.backends.cudnn.enabled = False

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
    train_loader, test_loader, total_vid = get_loaders(args.data, args.res, args.timesteps, True, args.batch_size, args.n_gpus, args.seed, cond=False)

    """ Get Model """
    # if rank == 0:
    log_(f"Generating model")

    torch.cuda.set_device(rank)
    # model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    model = ViTAutoencoder_SPADE(embed_dim=16)
    model = model.to(device)

    criterion = LPIPSWithDiscriminator(disc_start   = args.lossconfig.params.disc_start,
                                       timesteps    = args.ddconfig.timesteps).to(device)


    opt = torch.optim.AdamW(model.parameters(),
                             lr=args.lr,
                             betas=(0.5, 0.9)
                             )

    d_opt = torch.optim.AdamW(list(criterion.discriminator_2d.parameters()) + list(criterion.discriminator_3d.parameters()),
                             lr=args.lr,
                             betas=(0.5, 0.9))

    # if args.resume and rank == 0:
    if args.resume:
        print(os.path.join(args.first_stage_folder, 'model_last.pth'))
        model_ckpt = torch.load(os.path.join(args.first_stage_folder, 'model_last.pth'), map_location=f'cuda:{rank}')
        model.load_state_dict(model_ckpt)
        opt_ckpt = torch.load(os.path.join(args.first_stage_folder, 'opt.pth'), map_location=f'cuda:{rank}')
        opt.load_state_dict(opt_ckpt)

        del model_ckpt
        del opt_ckpt

        print('Model loaded')

    # if rank == 0:
    torch.save(model.state_dict(), rootdir + f'net_init.pth')

    fp = args.amp
    first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, args.first_stage_folder, fp, logger)

    # if rank == 0:
    torch.save(model.state_dict(), rootdir + f'net_meta.pth')
