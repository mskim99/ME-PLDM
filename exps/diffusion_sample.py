import os

import torch

from tools.dataloader import get_loaders
from models.autoencoder.autoencoder_spade import ViTAutoencoder_SPADE
from models.ddpm.unet_mask_3d import UNetModel, DiffusionWrapper
from evals.eval_cond import save_image_ddpm_mask, ddpm_pf_check

import copy
from utils import file_name, Logger
from models.ema import LitEma

# from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis, flop_count_table

#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor
_ema_ckpt_num = 60500

def diffusion_sample(rank, args):
    device = torch.device('cuda', rank)

    """ ROOT DIRECTORY """
    fn = file_name(args)
    logger = Logger(fn, ask=False)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')
    rootdir = logger.logdir

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    """ Get Image """
    log_(f"Loading dataset {args.data} with resolution {args.res}")
    train_loader, _, test_loader = get_loaders(rank, args.data, args.res, args.timesteps, args.skip, args.batch_size, args.n_gpus, args.seed, args.cond_model, pin_memory=True)

    """ Get Model """
    log_(f"Generating model")

    torch.cuda.set_device(rank)
    first_stage_model_src = ViTAutoencoder_SPADE(embed_dim=16, ch_mult=(1,2,4,8)).to(device)
    first_stage_model_trg = ViTAutoencoder_SPADE(embed_dim=16, ch_mult=(1,2,4,8)).to(device)

    first_stage_model_src_ckpt = torch.load(args.first_model_src, map_location='cuda:0')
    first_stage_model_trg_ckpt = torch.load(args.first_model_trg, map_location='cuda:0')
    first_stage_model_src.load_state_dict(first_stage_model_src_ckpt)
    first_stage_model_trg.load_state_dict(first_stage_model_trg_ckpt)
    del first_stage_model_src_ckpt
    del first_stage_model_trg_ckpt
    print('First stage model loaded')

    torch.cuda.empty_cache()

    unet = UNetModel(**args.unetconfig)
    ema_model = DiffusionWrapper(unet).to(device)

    if os.path.exists(rootdir + f'ema_model_' + str(_ema_ckpt_num) + '.pth'):
        ema_model_ckpt = torch.load(rootdir + f'ema_model_' + str(_ema_ckpt_num) + '.pth', map_location='cuda:0')
        ema_model.load_state_dict(ema_model_ckpt)
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200, dtype=torch.int)
        # ema_model = copy.deepcopy(model)
        print('Diffusion model loaded')

        del ema_model_ckpt

    else:
        raise FileNotFoundError("Diffusion model cannot be loaded")

    torch.cuda.empty_cache()

    first_stage_model_src.eval()
    first_stage_model_trg.eval()
    ema_model.eval()

    save_image_ddpm_mask(rank, ema_model, first_stage_model_src, first_stage_model_trg, _ema_ckpt_num, test_loader, logger)
    # ddpm_pf_check(rank, ema_model, first_stage_model_src, first_stage_model_trg, _ema_ckpt_num, test_loader, logger)

    # Performance Evaluation
    '''
    macs_fsms, params_fsms = get_model_complexity_info(first_stage_model_src, (1, 2, 32, 128, 128), input_constructor=prepare_input, as_strings=False, print_per_layer_stat=True,
                                                            verbose=True)
    macs_diff, params_diff = get_model_complexity_info(ema_model, (1, 16, 4, 16, 16), as_strings=False, print_per_layer_stat=True,
                                                            verbose=True)
    macs_fstg, params_fstg = get_model_complexity_info(first_stage_model_trg, (1, 16, 4, 16, 16), as_strings=False, print_per_layer_stat=True,
                                                            verbose=True)

    print("ptflops (FS_SRC) : MACS {} |  PARAM {}".format(macs_fsms, params_fsms))
    print("ptflops (DIFF) : MACS {} |  PARAM {}".format(macs_diff, params_diff))
    print("ptflops (FS_TRG) : MACS {} |  PARAM {}".format(macs_fstg, params_fstg))
    
    flops = FlopCountAnalysis(first_stage_model_src, (torch.ones(1, 2, 32, 128, 128).cuda(), torch.IntTensor(0).cuda()))

    print(flops.total())
    print(flop_count_table(flops))
    '''
