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

for i in range (1, 201):

    image_data = nib.load('J:/Dataset/QTAB_proc/T2w/T2w_' + str(i).zfill(4) + '.nii.gz').get_fdata()
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
    nib.save(image_mean_nib_img, 'J:/Dataset/QTAB_proc/T2w_grad/T2wg_' + str(i).zfill(4) + '.nii.gz')

    print(str(i) + ' finished')