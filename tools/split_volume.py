import pydicom as pdc
import glob
import numpy as np
import nibabel as nib
import os
from skimage.transform import resize

'''
path = "J:/Dataset/CHAOS_Train_Sets/Train_Sets/CT/*"
folder_list = glob.glob(path)

for folder_path in folder_list:
    files_list = glob.glob(folder_path + '/DICOM_anon/*.dcm')

    print(folder_path)
    print(len(files_list))

    imgs = []
    for file_path in files_list:
        dcm = pdc.dcmread(file_path)
        img = dcm.pixel_array
        imgs.append(img)

    imgs_arr = np.array(imgs)
    print(imgs_arr.shape)
    imgs_arr = imgs_arr.transpose(2, 1, 0)
    print(imgs_arr.shape)

    print(imgs_arr.min())
    print(imgs_arr.max())

    imgs_arr = 255. * (imgs_arr - imgs_arr.min()) / (imgs_arr.max() - imgs_arr.min())
    imgs_arr = imgs_arr.astype(np.uint8)

    print(imgs_arr.min())
    print(imgs_arr.max())

    imgs_arr = resize(imgs_arr, (1, 128, 128, 128))
    print(imgs_arr.shape)
    folder_path_name = os.path.basename(folder_path)

    nii_save = nib.Nifti1Image(imgs_arr, None)
    nib.save(nii_save, 'J:/Dataset/CHAOS_nibabel_norm_res_128/test/' + folder_path_name + '.nii.gz')
'''
# ds_path = "/data/jayeon/dataset/QTAB_proc"

def main():
    synth = {
        "ds_path" : "/data/jayeon/dataset/SynthRAD2023_brain_res_128",
        "cases": ['mr', 'mr_grad', 'ct', 'ct_grad'],
    }
    
    qtab = {
        "ds_path" : "/data/jayeon/dataset/QTAB_proc",
        "cases": ['t1w', 't1w_grad', 't2w', 't2w_grad'],
    }
    #############################################################
    # target = synth
    target = qtab
    
    for cases in target['cases']:
        split(
            ds_path=target['ds_path']
            ,cases=cases
        )
        
    target = synth
    
    for cases in target['cases']:
        split(
            ds_path=target['ds_path']
            ,cases=cases
        )
        
        
    pass

def split(ds_path = "/data/jayeon/dataset/SynthRAD2023_brain_res_128", cases = 'mr', *args, **kwargs):
    #ds_path = "/data/jayeon/dataset/SynthRAD2023_brain_res_128"

    s_z = 16 #블록이 가지는 slice의 개수
    overlap = 2
    direction = 1 # 0,1,2

    src_path = os.path.join(ds_path,f"{cases}/*") #"/data/jayeon/dataset/SynthRAD2023_brain_res_128/ct_grad/*" #"J:/Dataset/CHAOS_nibabel_norm/*"
    tgt_path = os.path.join(ds_path+f"_s_{s_z}_pd_{overlap}_drt_{['x','y','z'][direction]}", cases)

    files_list = glob.glob(src_path)

    print(len(files_list))


    for file_path in files_list:
        data_idx = int(os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0])
        # data_idx = int(os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0].split('_')[1])

        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_data = np.swapaxes(img_data, direction, img_data.ndim-1)

        # print(img_data.shape)
        #[128,128,128][2] = 128-overlap
        img_z = img_data.shape[2]
        z_num = int(img_z / s_z)
        z_num = z_num + 1 # 2 padding
        
       # img_data = resize(img_data, [128, 128, 126])
        # print(img_data.shape)
        
        # print(img_z)
        # print(z_num)

        prev_idx = 0
        for i in range (0, z_num):

            if data_idx > 200:
                break

            output_path = tgt_path + '/' + str(i)
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)

            cur_idx = prev_idx + s_z

            # img_part = img_data[:, :, (s_z*i):(s_z*(i+1))]
            # img_part = img_data[:, :, int(s_z*i/2):int(s_z*(i+2)/2)]
            img_part = img_data[:, :, prev_idx:cur_idx]
            # print(img_part.shape)

            img_part_nib = nib.Nifti1Image(img_part, None)
            nib.save(img_part_nib, tgt_path + '/' + str(i) + '/' + str(data_idx) + '_' + str(i).zfill(4) + f'_s_{s_z}.nii.gz')

            # print(i)
            # print(int(s_z*i/2))
            # print(int(s_z*(i+2)/2))
            # print(img_part.shape)

            print(prev_idx)
            print(cur_idx)
            
            #겹치는 slice
            
            prev_idx = cur_idx - overlap

        # exit(0)
        print(str(data_idx) + ' finished')

'''
path = "J:/Dataset/CHAOS_nibabel_norm_res_128/*"
after_path = "J:/Dataset/CHAOS_nibabel_norm_res_128_s_16_c"
files_path = glob.glob(path)

for file in files_path:
    print(file)
    head, file_name = os.path.split(file)
    _, idx = os.path.split(head)
    print(file_name)
    print(idx)

    output_path = after_path + '/' + str(idx)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    img = nib.load(file)
    img_data = img.get_fdata()

    img_resize = resize(img_data, [128, 128, 16])
    img_resize_nib = nib.Nifti1Image(img_resize, None)
    nib.save(img_resize_nib, after_path + '/' + idx + '/' + file_name)
'''
'''
path = "J:/Dataset/HCP_1200_norm_res_128_s_16/*/*"
files_list = glob.glob(path)
for file in files_list:

    img = nib.load(file)
    img_data = img.get_fdata()
    # print(img_data.shape)

    head, file_name = os.path.split(file)
    _, idx = os.path.split(head)

    # print(idx + '/' + file_name)
    print(idx + '/' + file_name + ' ' + str(int(idx) + 1))
    '''
'''
path = "J:/Program/PVDM_modify/output/first_stage_main_CHAOS_42/generated_120000.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()
print(img_data.shape)
'''

if __name__ == '__main__':
    main()