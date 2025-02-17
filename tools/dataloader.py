from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.folder import make_dataset
from torchvision.io import read_video
from itertools import chain

data_location = '/data/jayeon/dataset' #'/data/jionkim'
# data_location = '/home/work/jionkim'
# data_location = '/home/oem/jionkim'
# data_location = "/home/work/workspace/"

from tools.data_utils import *

import nibabel as nib
from skimage.transform import resize

class Image3DDataset(Dataset):
    def __init__(self,
                 root_src,
                 root_src_grad,
                 root_dst,
                 root_dst_grad,
                 train,
                 resolution,
                 n_frames=16,
                 skip=1,
                 fold=1,
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,  # True for evaluating FVD
                 seed=42,
                 ):

        image_3d_root_src = osp.join(os.path.join(root_src))
        image_3d_root_src_grad = osp.join(os.path.join(root_src_grad))
        image_3d_root_dst = osp.join(os.path.join(root_dst))
        image_3d_root_dst_grad = osp.join(os.path.join(root_dst_grad))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path_src = image_3d_root_src
        self.path_src_grad = image_3d_root_src_grad
        self.path_dst = image_3d_root_dst
        self.path_dst_grad = image_3d_root_dst_grad
        name = image_3d_root_src.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.classes = list(natsorted(p for p in os.listdir(image_3d_root_src) if osp.isdir(osp.join(image_3d_root_src, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples_src = make_dataset(image_3d_root_src, class_to_idx, ('nii.gz',), is_valid_file=None)
        self.samples_src_grad = make_dataset(image_3d_root_src_grad, class_to_idx, ('nii.gz',), is_valid_file=None)
        self.samples_dst = make_dataset(image_3d_root_dst, class_to_idx, ('nii.gz',), is_valid_file=None)
        self.samples_dst_grad = make_dataset(image_3d_root_dst_grad, class_to_idx, ('nii.gz',), is_valid_file=None)

        image_3d_list_src = [x[0] for x in self.samples_src]
        image_3d_list_src_grad = [x[0] for x in self.samples_src_grad]
        image_3d_list_dst = [x[0] for x in self.samples_dst]
        image_3d_list_dst_grad = [x[0] for x in self.samples_dst_grad]
        vol_num_list = [int(os.path.basename(x[0]).split('_')[0]) for x in self.samples_src] # Extract volume number from path
        vol_num_list_unique = list(set(vol_num_list))
        slices_num_list = [x[1] for x in self.samples_src]

        # Bind image_3d_list and slice_num_list with same data
        image_3d_list_src_c = []
        image_3d_list_src_grad_c = []
        image_3d_list_dst_c = []
        image_3d_list_dst_grad_c = []
        slices_num_list_c = []
        for vn_l in vol_num_list_unique:
            img_s_3d_list = []
            img_s_grad_3d_list = []
            img_d_3d_list = []
            img_d_grad_3d_list = []
            sls_num_list = []
            for idx, val in enumerate(vol_num_list):
                if val == vn_l:
                    img_s_3d_list.append(image_3d_list_src[idx])
                    img_s_grad_3d_list.append(image_3d_list_src_grad[idx])
                    img_d_3d_list.append(image_3d_list_dst[idx])
                    img_d_grad_3d_list.append(image_3d_list_dst_grad[idx])
                    sls_num_list.append(slices_num_list[idx])

            image_3d_list_src_c.append(img_s_3d_list)
            image_3d_list_src_grad_c.append(img_s_grad_3d_list)
            image_3d_list_dst_c.append(img_d_3d_list)
            image_3d_list_dst_grad_c.append(img_d_grad_3d_list)
            slices_num_list_c.append(sls_num_list)

        self.image_3d_list_src = image_3d_list_src_c
        self.image_3d_list_src_grad = image_3d_list_src_grad_c
        self.image_3d_list_dst = image_3d_list_dst_c
        self.image_3d_list_dst_grad = image_3d_list_dst_grad_c
        self.vol_num_list = vol_num_list
        self.slices_num_list = slices_num_list_c
        self.num_channels = 1
        self.return_vid = return_vid

        frames_between_clips = skip
        print(root_src, root_dst, frames_between_clips, n_frames)
        if train:
            # self.indices = [i for i in range(0, int(len(self.image_3d_list_src) * 0.7))]
            # self.indices = [i for i in range(0, 98)]
            self.indices = [i for i in chain(range(0, 18), range(54, 126))]
        else:
            # self.indices = [i for i in range(int(len(self.image_3d_list_src) * 0.7), len(self.image_3d_list_src))]
            # self.indices = [i for i in range(98, 140)]
            self.indices = [i for i in range(18, 54)]

        random.seed(seed)

        # print(self.indices)

        self.size = len(self.indices)
        print(self.size)
        self.shuffle_indices = self.indices

        self._need_init = True

    def _select_fold(self, video_list, path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_idx = self.indices[idx]

        img_paths_src = self.image_3d_list_src[img_idx]
        img_paths_src_grad = self.image_3d_list_src_grad[img_idx]
        img_paths_dst = self.image_3d_list_dst[img_idx]
        img_paths_dst_grad = self.image_3d_list_dst_grad[img_idx]
        slice_nums = self.slices_num_list[img_idx]

        # print(self.image_3d_list_dst)
        # exit(0)

        # 3D image load
        img_datas_src_grad = []
        img_datas_src = []
        img_datas_dst_grad = []
        img_datas_dst = []

        for idx in range (0, img_paths_src.__len__()):
            img_src = nib.load(img_paths_src[idx])
            img_data_src = img_src.get_fdata()
            img_data_src = img_data_src.swapaxes(0, 2)
            img_data_src = np.expand_dims(img_data_src, axis=1)

            img_src_grad = nib.load(img_paths_src_grad[idx])
            img_data_src_grad = img_src_grad.get_fdata()
            img_data_src_grad = img_data_src_grad.swapaxes(0, 2)
            img_data_src_grad = np.expand_dims(img_data_src_grad, axis=1)

            img_dst = nib.load(img_paths_dst[idx])
            img_data_dst = img_dst.get_fdata()
            img_data_dst = img_data_dst.swapaxes(0, 2)
            img_data_dst = np.expand_dims(img_data_dst, axis=1)

            img_dst_grad = nib.load(img_paths_dst_grad[idx])
            img_data_dst_grad = img_dst_grad.get_fdata()
            img_data_dst_grad = img_data_dst_grad.swapaxes(0, 2)
            img_data_dst_grad = np.expand_dims(img_data_dst_grad, axis=1)

            img_datas_src.append(img_data_src)
            img_datas_src_grad.append(img_data_src_grad)
            img_datas_dst.append(img_data_dst)
            img_datas_dst_grad.append(img_data_dst_grad)

        return img_datas_src, img_datas_src_grad, img_datas_dst, img_datas_dst_grad, slice_nums


class Image3DDataset_AE(Dataset):
    def __init__(self,
                 root,
                 root_grad,
                 train,
                 resolution,
                 n_frames=16,
                 skip=1,
                 fold=1,
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,  # True for evaluating FVD
                 seed=42,
                 ):
        image_3d_root = osp.join(os.path.join(root))
        image_3d_root_grad = osp.join(os.path.join(root_grad))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = image_3d_root
        self.path_grad = image_3d_root_grad
        name = image_3d_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.classes = list(natsorted(p for p in os.listdir(image_3d_root) if osp.isdir(osp.join(image_3d_root, p))))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(image_3d_root, class_to_idx, ('nii.gz',), is_valid_file=None)
        self.samples_grad = make_dataset(image_3d_root_grad, class_to_idx, ('nii.gz',), is_valid_file=None)

        print("Number of samples in original data:", len(self.samples))
        print("Number of samples in gradient data:", len(self.samples_grad))
        
        image_3d_list = [x[0] for x in self.samples]
        image_3d_list_grad = [x[0] for x in self.samples_grad]
        vol_num_list = [int(os.path.basename(x[0]).split('_')[0]) for x in self.samples] # Extract volume number from path
        vol_num_list_unique = list(set(vol_num_list))
        slices_num_list = [x[1] for x in self.samples]

        # Bind image_3d_list and slice_num_list with same data
        image_3d_list_c = []
        image_3d_list_grad_c = []
        slices_num_list_c = []
        for vn_l in vol_num_list_unique:
            img_3d_list = []
            img_g_3d_list = []
            sls_num_list = []
            for idx, val in enumerate(vol_num_list):
                if val == vn_l:
                    img_3d_list.append(image_3d_list[idx])
                    img_g_3d_list.append(image_3d_list_grad[idx])
                    sls_num_list.append(slices_num_list[idx])

            image_3d_list_grad_c.append(img_g_3d_list)
            image_3d_list_c.append(img_3d_list)
            slices_num_list_c.append(sls_num_list)

        self.image_3d_list = image_3d_list_c
        self.image_3d_list_grad = image_3d_list_grad_c
        self.vol_num_list = vol_num_list
        self.slices_num_list = slices_num_list_c
        self._use_labels = use_labels
        self._label_shape = None
        self._raw_labels = None
        self._raw_shape = [len(self.image_3d_list)] + [16, resolution, resolution]
        self.num_channels = 1
        self.return_vid = return_vid

        frames_between_clips = skip
        print(root, frames_between_clips, n_frames)
        if train:
            self.indices = [i for i in range(0, int(len(self.image_3d_list) * 0.7))]
        else:
            self.indices = [i for i in range(int(len(self.image_3d_list) * 0.7), len(self.image_3d_list))]
        # self.indices = self._select_fold(self.image_3d_list, self.path, fold, train)

        random.seed(seed)

        self.size = len(self.indices)
        print(self.size)
        self.shuffle_indices = self.indices

        self._need_init = True

    def _select_fold(self, video_list, path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_idx = self.indices[idx]

        img_paths = self.image_3d_list[img_idx]
        img_paths_grad = self.image_3d_list_grad[img_idx]
        slice_nums = self.slices_num_list[img_idx]

        # 3D image load
        img_datas = []
        img_datas_grad = []

        for idx in range (0, img_paths.__len__()):
            img = nib.load(img_paths[idx])
            img_data = img.get_fdata()
            img_data = img_data.swapaxes(0, 2)
            img_data = np.expand_dims(img_data, axis=1)

            img_grad = nib.load(img_paths_grad[idx])
            img_data_grad = img_grad.get_fdata()
            img_data_grad = img_data_grad.swapaxes(0, 2)
            img_data_grad = np.expand_dims(img_data_grad, axis=1)

            # img_data = img_data.reshape([16, 1, 128, 128])
            # img_data_mask = img_data_mask.reshape([16, 1, 128, 128])
            # img_data_grad = img_data_grad.reshape([16, 1, 128, 128])

            img_datas.append(img_data)
            img_datas_grad.append(img_data_grad)

        return img_datas, img_datas_grad, slice_nums, idx


def get_loaders(imgstr, resolution, timesteps, ae, batch_size=1, n_gpus=1, seed=42,  cond=False, pin_memory=False, direction='z'):
    
    if imgstr == 'SYNTHRAD2023_PAIR_PD_2_trg_MR':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/ct') #
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/ct_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/mr')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/mr_grad')
    elif imgstr == 'SYNTHRAD2023_PAIR_PD_2_trg_CT':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/mr')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/mr_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/ct')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_16_pd_2_drt_{direction}/ct_grad')
    elif imgstr == 'SYNTHRAD2023_PAIR_S_32_PD_2_trg_MR':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/ct')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/ct_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/mr')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/mr_grad')
    elif imgstr == 'SYNTHRAD2023_PAIR_S_32_PD_2_trg_CT':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/mr')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/mr_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/ct')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_32_pd_2_drt_{direction}/ct_grad')
    elif imgstr == 'SYNTHRAD2023_PAIR_S_8_PD_2_trg_MR':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/ct')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/ct_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/mr')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/mr_grad')
    elif imgstr == 'SYNTHRAD2023_PAIR_S_8_PD_2_trg_CT':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/mr')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/mr_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/ct')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_8_pd_1_drt_{direction}/ct_grad')
    elif imgstr == 'QTAB_S_8_PD_2_trg_T1':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t2w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t2w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t1w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t1w_grad')
    elif imgstr == 'QTAB_S_8_PD_2_trg_T2':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t1w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t1w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t2w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_8_pd_1_drt_{direction}/t2w_grad')
    elif imgstr == 'SYNTHRAD2023_PAIR_PD_2_res_64':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_64_s_8_pd_1_drt_{direction}/ct')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_64_s_8_pd_1_drt_{direction}/ct_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_64_s_8_pd_1_drt_{direction}/mr')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_64_s_8_pd_1_drt_{direction}/mr_grad')
    elif imgstr == 'QTAB_PD_2_trg_T1':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t2w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t2w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t1w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t1w_grad')
    elif imgstr == 'QTAB_PD_2_trg_T2':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t1w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t1w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t2w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_16_pd_2_drt_{direction}/t2w_grad')
    elif imgstr == 'QTAB_S_32_PD_2_trg_T2':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t1w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t1w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t2w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t2w_grad')
    elif imgstr == 'QTAB_S_32_PD_2_trg_T1':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t2w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t2w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t1w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_32_pd_2_drt_{direction}/t1w_grad')
    elif imgstr == 'SYNTHRAD_BRAIN_S_XYZ_64_trg_CT':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/mr')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/mr_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/ct')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/ct_grad')
    elif imgstr == 'SYNTHRAD_BRAIN_S_XYZ_64_trg_MR':
        train_dir_src = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/ct')
        train_dir_src_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/ct_grad')
        train_dir_dst = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/mr')
        train_dir_dst_grad = os.path.join(data_location, f'SynthRAD2023_brain_res_128_s_xyz_64_pd_2_drt_{direction}/mr_grad')
    elif imgstr == 'QTAB_S_XYZ_64_trg_T2':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t1w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t1w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t2w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t2w_grad')
    elif imgstr == 'QTAB_S_XYZ_64_trg_T1':
        train_dir_src = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t2w')
        train_dir_src_grad = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t2w_grad')
        train_dir_dst = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t1w')
        train_dir_dst_grad = os.path.join(data_location, f'QTAB_proc_s_xyz_64_pd_2_drt_{direction}/t1w_grad')
    else:
        raise NotImplementedError()

    if cond:
        print("here")
        timesteps *= 2  # for long generation

    if ae:
        trainset = Image3DDataset_AE(train_dir_dst, train_dir_dst_grad, train=True, resolution=resolution)
        testset = Image3DDataset_AE(train_dir_dst, train_dir_dst_grad, train=False, resolution=resolution)
    else:
        trainset = Image3DDataset(train_dir_src, train_dir_src_grad, train_dir_dst, train_dir_dst_grad,
                                  train=True, resolution=resolution)
        testset = Image3DDataset(train_dir_src, train_dir_src_grad, train_dir_dst, train_dir_dst_grad,
                                 train=False, resolution=resolution)

    trainset_sampler = InfiniteSampler(dataset=trainset, rank=0, num_replicas=n_gpus, seed=seed)
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size, pin_memory=pin_memory, num_workers=2, prefetch_factor=2)
    testset_sampler = InfiniteSampler(testset, num_replicas=n_gpus, rank=0, shuffle=False)
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size, pin_memory=pin_memory, num_workers=2, prefetch_factor=2)

    return trainloader, trainloader, testloader 


