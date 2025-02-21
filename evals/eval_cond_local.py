import nibabel as nib
import glob
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from sklearn import metrics as skm
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from pytorch_msssim import ssim as ssim2
from sklearn.metrics import mean_absolute_error
from skimage.transform import resize
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import statistics as st
import lpips
loss_fn_alex = lpips.LPIPS(net='alex')

import math
import numpy as np

import torch

# This code is running in the local system
# (Not include the image generation process)

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = skm.pairwise.rbf_kernel(X, X, gamma)
    YY = skm.pairwise.rbf_kernel(Y, Y, gamma)
    XY = skm.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # NumPy 배열을 PIL 이미지로 변환
            # transforms.Resize((3, 128, 128)),  # InceptionV3 입력 크기에 맞춤
            transforms.ToTensor(),  # 텐서로 변환
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # return self.transform(img)
        return img


def load_numpy_as_dataloader(numpy_data, batch_size=128):
    dataset = NumpyDataset(numpy_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def compute_statistics_of_dataloader(dataloader, model, device):
    model.eval()
    act = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)[0]  # Inception 모델의 출력 특징
            act.append(pred.cpu().numpy())

    act = np.concatenate(act, axis=0)
    mu = np.mean(act, axis=0)
    act = act.squeeze()
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# NumPy 데이터를 입력받아 FID를 계산하는 함수
def calculate_fid_from_numpy(numpy_data1, numpy_data2, batch_size=128, device='cuda'):
    # GPU 또는 CPU 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Inception 모델 로드 (pool3 레이어 사용)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    # 데이터 로드
    dataloader1 = load_numpy_as_dataloader(numpy_data1, batch_size=batch_size)
    dataloader2 = load_numpy_as_dataloader(numpy_data2, batch_size=batch_size)

    # 특징 벡터 계산
    mu1, sigma1 = compute_statistics_of_dataloader(dataloader1, model, device)
    mu2, sigma2 = compute_statistics_of_dataloader(dataloader2, model, device)

    # FID 계산
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

metrics = dict()
metrics['MAE'] = []
metrics['RMSE'] = []
metrics['PSNR'] = []
metrics['SSIM'] = []
metrics['MS-SSIM'] = []
metrics['MMD'] = []
metrics['FID'] = []
metrics['LPIPS'] = []

# set the directory that fake & real data is included
fake_list = sorted(glob.glob('/home/jionkim/workspace/_eval/3BD_T2/fake/*.nii'))
real_list = sorted(glob.glob('/home/jionkim/workspace/_eval/3BD_T2/real/*.nii.gz'))

for idx in range(0, fake_list.__len__()):
# for idx in range(0, 2):

    mae_value = 100000.
    psnr_value = 0.
    ssim_value = 0.

    fake = nib.load(fake_list[idx]).get_fdata() * 255.
    real = nib.load(real_list[idx]).get_fdata() * 255.
    # fake[fake < 0.] = 0.
    # fake_psnr = 0.4 * (fake - fake.min()) / (fake.max() - fake.min()) + 0.6
    # real_psnr = 0.4 * (real - real.min()) / (real.max() - real.min()) + 0.6
    fake_psnr = (fake - fake.min()) / (fake.max() - fake.min())
    real_psnr = (real - real.min()) / (real.max() - real.min())
    # fake_psnr = fake_psnr.astype(int)
    # 4real_psnr = real_psnr.astype(int)

    '''
    fake = np.flip(fake, axis=0)
    fake_nii = nib.Nifti1Image(fake, None)
    real_nii = nib.Nifti1Image(real, None)
    nib.save(fake_nii,'/home/jionkim/workspace/_eval/test_fake.nii.gz')
    nib.save(real_nii,'/home/jionkim/workspace/_eval/test_real.nii.gz')
    # exit(0)

    fake_sort = sorted(fake.reshape(-1))
    fake_min = fake_sort[int(2097152*0.45)]

    fake = fake - fake_min
    fake[fake < 0.] = 0.
    fake *= 255. / fake.max()
    '''
    fake = fake.astype(np.uint8)
    real = real.astype(np.uint8)

    fake_psnr = np.squeeze(fake_psnr)
    real_psnr = np.squeeze(real_psnr)

    fake = np.squeeze(fake)
    real = np.squeeze(real)

    mae_value = math.sqrt(mean_squared_error(fake, real))
    psnr_value = peak_signal_noise_ratio(fake_psnr, real_psnr)
    ssim_value = structural_similarity(fake, real)

    metrics['MAE'].append(mae_value)
    metrics['PSNR'].append(psnr_value)
    metrics['SSIM'].append(ssim_value)

    # Calculate MMD
    mmd_value = 0.
    rmse_values = []
    for i in range (0, 128):
        mmd_value += mmd_rbf(fake[:, :, i], real[:, :, i])
        rmse_values.append(skm.mean_squared_error(fake[:, :, i], real[:, :, i], squared=False))

    metrics['RMSE'].append(np.average(rmse_values))
    metrics['MMD'].append(mmd_value)

    fake_fid = fake / 255.
    real_fid = real / 255.

    # Calculate FID
    fake_fid = np.expand_dims(fake, axis=1)
    real_fid = np.expand_dims(real, axis=1)
    fake_fid = np.hstack([fake_fid, fake_fid, fake_fid]).astype(np.float32)
    real_fid = np.hstack([real_fid, real_fid, real_fid]).astype(np.float32)
    fid_value = calculate_fid_from_numpy(fake_fid, real_fid)
    metrics['FID'].append(np.average(fid_value / 128.).astype(float))
    '''
    fake_ssim = resize(fake_fid, (128, 3, 256, 256))
    real_ssim = resize(real_fid, (128, 3, 256, 256))
    fake_ssim = torch.Tensor(255. * fake_ssim)
    real_ssim = torch.Tensor(255. * real_ssim)
    ms_ssim_value = ms_ssim(fake_ssim, real_ssim, data_range=255, size_average=False)
    metrics['MS-SSIM'].append(np.average(ms_ssim_value).astype(float))
    '''
    # Calculate LPIPS
    fake_lpips = 2. * fake_fid - 1.
    real_lpips = 2. * real_fid - 1.
    lpips_value = 0.
    for i in range (0, 128):
        flps = torch.Tensor(fake_lpips[i])
        rlps = torch.Tensor(real_lpips[i])
        lpips_value += loss_fn_alex(flps, rlps).item()

    metrics['LPIPS'].append(lpips_value)
    print(str(idx) + ' finished')


# mae_sort_idx = sorted(range(len(metrics['MAE'])),key=metrics['MAE'].__getitem__)[0:8]
mae_final = sorted(metrics['MAE'])[0:8]
rmse_final = sorted(metrics['RMSE'])[0:8]
# psnr_sort_idx = sorted(range(len(metrics['PSNR'])),key=metrics['PSNR'].__getitem__, reverse=True)[0:8]
psnr_final = sorted(metrics['PSNR'], reverse=True)[0:8]
# ssim_sort_idx = sorted(range(len(metrics['SSIM'])),key=metrics['SSIM'].__getitem__, reverse=True)[0:8]
ssim_final = sorted(metrics['SSIM'], reverse=True)[0:8]
# ms_ssim_final = sorted(metrics['MS-SSIM'], reverse=True)[0:8]
# mmd_sort_idx = sorted(range(len(metrics['MMD'])),key=metrics['MMD'].__getitem__, reverse=True)[0:8]
mmd_final = sorted(metrics['MMD'])[0:8]
# fid_sort_idx = sorted(range(len(metrics['FID'])),key=metrics['FID'].__getitem__, reverse=True)[0:8]
fid_final = sorted(metrics['FID'])[0:8]
# lpips_sort_idx = sorted(range(len(metrics['LPIPS'])),key=metrics['LPIPS'].__getitem__, reverse=True)[0:8]
lpips_final = sorted(metrics['LPIPS'])[0:8]

# print(mae_sort_idx)
# print(psnr_sort_idx)
# print(ssim_sort_idx)

'''
# print AVERAGE
print('[EVALUATION (AVG)] [MAE %f] [PSNR %f] [SSIM %f]' % (st.mean(metrics['MAE']), st.mean(metrics['PSNR']),
                                                    st.mean(metrics['SSIM'])))

# print STD
print('[EVALUATION (STD)] [MAE %f] [PSNR %f] [SSIM %f]' % (st.pstdev(metrics['MAE']), st.pstdev(metrics['PSNR']),
                                                    st.pstdev(metrics['SSIM'])))
'''


print('[EVALUATION (AVG)] [MAE %f] [RMSE %f] [PSNR %f] [SSIM %f] [MMD %f] [FID %f] [LPIPS %f]' % (st.mean(mae_final), st.mean(rmse_final), st.mean(psnr_final),
                                                    st.mean(ssim_final), st.mean(mmd_final), st.mean(fid_final), st.mean(lpips_final)))

# print STD
print('[EVALUATION (STD)] [MAE %f] [RMSE %f] [PSNR %f] [SSIM %f] [MMD %f] [FID %f] [LPIPS %f]' % (st.pstdev(mae_final), st.pstdev(rmse_final), st.pstdev(psnr_final),
                                                    st.pstdev(ssim_final), st.pstdev(mmd_final), st.pstdev(fid_final), st.pstdev(lpips_final)))

print('Evaluation finished')
