# Test of 2D-U-NET

# Headers :
from __future__ import print_function
import os, os.path
import cv2
import numpy as np

from model_past_version import *

# Parameters :
tests = 1
image_rows = 256
image_cols = 256
n_ch = 4
smooth = np.finfo(float).eps
data_path = ''


# Functions :
def dice_coef_test(y_true, y_pred):
    y_true_f = y_true.flatten('F')
    y_pred_f = y_pred.flatten('F')
    intersection = sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

def dice_coef_test_3D(prediction, ground_truth):
    intersection = 0
    slice_pred_sum = 0
    slice_true_sum =0
    for i in range(np.shape(prediction)[0]):
        slice_pred = np.squeeze(prediction[i,:,:,:])
        slice_true = np.squeeze(ground_truth[i,:,:,:])
        slice_pred_f = slice_pred.flatten('F')
        slice_true_f = slice_true.flatten('F')               
        slice_pred_sum = slice_pred_sum + sum(slice_pred_f)
        slice_true_sum = slice_true_sum + sum(slice_true_f)
        intersection = intersection + sum(slice_true_f*slice_pred_f)  
    return (2. * intersection + smooth) / (slice_true_sum + slice_pred_sum + smooth)


# test :
for test_num in range(1, tests+1):
    model = get_unet(image_rows, image_cols, n_ch)
    model.load_weights(os.path.join(data_path,'learned_model_'+str(test_num)+'.hdf5'))
    results_home = 'results_'+str(test_num)

    # load testing data      
    test_data_dir_path = 'C:\\Users\\skazemin\\Documents\\Codes\\Segmentation_Spine\\dataset_slices\\test\\'
    test_dir_list = os.listdir(test_data_dir_path)
    dice_coeff_test = dict()
    for folder in range(len(test_dir_list)):
        test_data_path=os.path.join(test_data_dir_path, test_dir_list[folder])
        test_files = os.listdir(test_data_path)
        total = len(test_files) // 2
    
        imgs_test = np.ndarray((total, image_rows, image_cols, n_ch), dtype=np.uint8)
        imgs_test_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    	
        """ here I tried different input templates having different number of channels:
        		n_ch = 1, when the input only contains only one modality of the image slice
        		n_ch = 4, when the input contains all the 4 modalities in the MRI image for the current slice
        		n_ch = 12, when  the input contains all modalities of the current slice in addition to the previous and next slice
        """

        i = 0 # the monitoring param
        print('-'*30)
        print('Creating testing images...')
        print('-'*30)
        valid_image_name = []
        for file_name in test_files:
            if n_ch == 1:
                if 'mask' in file_name:
                    continue
                image_mask_name = file_name.split('.')[0] + '_mask.tif'
                valid_image_name.append(file_name)
    
                tmp_img = cv2.imread(os.path.join(test_data_path, file_name),0)           
                tmp_img_mask = cv2.imread(os.path.join(test_data_path, image_mask_name),0)
    
                tmp_img = cv2.resize(tmp_img,(image_cols, image_rows))
                tmp_img = np.reshape(tmp_img,(image_rows, image_cols, n_ch))
                tmp_img_mask = cv2.resize(tmp_img_mask,(image_cols, image_rows))
                tmp_img_mask = np.reshape(tmp_img_mask,(image_rows, image_cols, 1))
    
                tmp_img = np.array([tmp_img])
                tmp_img_mask = np.array([img_mask])
                imgs_test[i] = tmp_img
                imgs_test_mask[i] = tmp_img_mask
    
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            if n_ch == 4:
                tmp_img = np.ndarray((image_cols, image_cols, n_ch), dtype=np.uint8)
                channels = ['wat', 'fat', 'inn', 'opp']
                tmp_img_slice_name = '_'.join(file_name.split('.')[0].split('_')[:3])
                if 'wat' in file_name:
                    valid_image_name.append(tmp_img_slice_name)
                    img_name = tmp_img_slice_name+'_'+channels[0]+'.'+file_name.split('.')[1]
                    tmp_img[:,:,0] = cv2.imread(os.path.join(test_data_path, img_name),0)
                else:
                    continue
                for j in range(1,len(channels)):
                    img_name = tmp_img_slice_name+'_'+channels[j]+'.'+file_name.split('.')[1]
                    tmp_img[:,:,j] = cv2.imread(os.path.join(test_data_path, img_name),0)
                image_mask_name = tmp_img_slice_name+'_mask.'+file_name.split('.')[1]
    
                tmp_img = cv2.resize(tmp_img,(image_cols, image_rows))
                tmp_img = np.reshape(tmp_img,(image_rows, image_cols,n_ch))
                tmp_img_mask = cv2.imread(os.path.join(test_data_path, image_mask_name),0)
                tmp_img_mask = cv2.resize(tmp_img_mask,(image_cols, image_rows))
                tmp_img_mask = np.reshape(tmp_img_mask,(image_rows, image_cols,1))
                tmp_img = np.array([tmp_img])
                tmp_img_mask = np.array([tmp_img_mask])
    
                imgs_test[i] = tmp_img
                imgs_test_mask[i] = tmp_img_mask
    
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            if n_ch == 12: #to add previous and next images to channels
                tmp_img = np.ndarray((image_cols, image_rows, n_ch), dtype=np.uint8)
                channels = ['wat', 'fat', 'inn', 'opp']
                tmp_img_slice_name = '_'.join(file_name.split('.')[0].split('_')[:3])
                tmp_img_name = '_'.join(file_name.split('.')[0].split('_')[:2])
                slice_num = tmp_img_slice_name.split('_')[2]  
                if int(slice_num) == 0:
                    continue
                if int(slice_num) == 35:
                    continue
                if 'wat' in file_name:
                    valid_image_name.append(tmp_img_slice_name)
                    for x in range (0, 3): #to have the set (-1,0,1) regarding to previous, current and next slice of image
                        per_cur_next = x-1
                        ch_num = (per_cur_next*4)+4
                        img_name = tmp_img_name+'_'+str(int(slice_num)+per_cur_next)+'_'+channels[0]+'.'+file_name.split('.')[1]
                        tmp_img[:,:,ch_num] = cv2.imread(os.path.join(test_data_path, img_name),0)
                else:
                    continue
                for j in range(1,len(channels)):
                    for x in range (0, 3):
                        per_cur_next = x-1
                        ch_num = (per_cur_next*4)+4+j
                        img_name = tmp_img_name+'_'+str(int(slice_num)+per_cur_next)+'_'+channels[j]+'.'+file_name.split('.')[1]
                        tmp_img[:,:,(ch_num)] = cv2.imread(os.path.join(test_data_path, img_name),0)
                image_mask_name = tmp_img_slice_name+'_mask.'+file_name.split('.')[1]
    
                tmp_img = cv2.resize(tmp_img,(image_cols, image_rows))
                tmp_img = np.reshape(tmp_img,(image_rows, image_cols,n_ch))
                tmp_img_mask = cv2.imread(os.path.join(test_data_path, image_mask_name),0)
                tmp_img_mask = cv2.resize(tmp_img_mask,(image_cols, image_rows))
                tmp_img_mask = np.reshape(tmp_img_mask,(image_rows, image_cols,1))
                tmp_img = np.array([tmp_img])
                tmp_img_mask = np.array([tmp_img_mask])
    
                imgs_test[i] = tmp_img
                imgs_test_mask[i] = tmp_img_mask
    
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
        print('Loading done for test data.')
    
        imgs_test = imgs_test.astype('float32')
        imgs_test_mask = imgs_test_mask.astype('float32')
        imgs_test_mask /= 255.  # scale masks to [0, 1]
        imgs_test_mask = np.round(imgs_test_mask)
    
        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)
        imgs_mask_predict = model.predict(imgs_test, batch_size=4, verbose=1)
        imgs_mask_predict[np.where(imgs_mask_predict<1)]=0
        
        # 3D_image Dice coefficiency calculation
        dice_coeff_test[test_dir_list[folder]] = dice_coef_test_3D(imgs_mask_predict, imgs_test_mask)
        print('Dice of 3D-image '+test_dir_list[folder]+':')
        print(dice_coeff_test[test_dir_list[folder]])
        
        # result path
        result_path = os.path.join(data_path, results_home)
        result_dir_path = os.path.join(result_path, test_dir_list[folder])
        if not os.path.exists(result_dir_path):
            os.makedirs(result_dir_path)
    
        # to save predicted masks (testing results) 
        lp=0
        for im_name in valid_image_name:
            img_pred = (np.reshape(imgs_mask_predict[lp], (image_rows, image_cols))*255).astype(np.uint8)
            image_pred_name = im_name + '_pred.png'
            cv2.imwrite(os.path.join(result_dir_path, image_pred_name), img_pred)
            lp += 1
    
    # save coeff file
    if not os.path.exists(results_home):
        os.makedirs(results_home)
    np.save(results_home+'/test_dice_coeff.npy',dice_coeff_test)
    test_Dice_mean = sum(dice_coeff_test.values())/len(dice_coeff_test)
    print('Dice_Mean = ')
    print(test_Dice_mean)