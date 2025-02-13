import os
import glob
import numpy as np
import nibabel as nib
from skimage.transform import resize
'''
image_data = nib.load('J:/Dataset/HCP_1200_norm_res_128/rb_00001.nii.gz').get_fdata()
print(image_data.shape)

grad_image_data = np.gradient(image_data)
gidp = grad_image_data[0]
gidp[gidp < 0] = 0
grad_nib_img = nib.Nifti1Image(gidp, None)
nib.save(grad_nib_img, 'J:/test/rb_00001_grad_0_proc.nii.gz')
'''

ds_path = '/data/jayeon/dataset/QTAB_proc'
#ds_path = "/data/jayeon/dataset/SynthRAD2023_brain_res_128"

in_path = os.path.join(ds_path, 't1w') #'/data/jayeon/dataset/SynthRAD2023_brain_res_128/mr/' #'J:/Dataset/QTAB_proc/T2w/T2w_'
out_path = os.path.join(ds_path, 't1w_grad') #'J:/Dataset/QTAB_proc/T2w_grad/T2wg_'

os.makedirs(out_path, exist_ok=True)
files_list = sorted(glob.glob(os.path.join(in_path,'*')))
#180: synthrad2023
for i, file in enumerate(files_list):
    #str_i = str(i) #str(i).zfill(4)
    data_idx = os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0]
    image_data = nib.load(os.path.join(in_path, data_idx+'.nii.gz')).get_fdata()
    # image_data = resize(image_data, [128, 128, 128])

    grad_image_data = np.gradient(image_data)
    gidp_1 = grad_image_data[1]
    gidp_2 = grad_image_data[2]
    gidp_1[gidp_1 < 0] = 0
    gidp_2[gidp_2 < 0] = 0
    image_mean = (gidp_1 + gidp_2) / 2.

    # image_mean = np.sqrt(image_mean)
    image_mean = (image_mean - image_mean.min()) / (image_mean.max() - image_mean.min())
    # image_mean[image_mean > 0.05] = 1.
    # image_mean[image_mean < 0.05] = 0.

    image_mean_nib_img = nib.Nifti1Image(image_mean, None)
    nib.save(image_mean_nib_img, os.path.join(out_path, data_idx+'.nii.gz'))

    print(str(i) + ' finished')