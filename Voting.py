import os
import numpy as np
import cv2

smooth = np.finfo(float).eps

def dice_coef_test_3D(prediction, ground_truth):
    intersection = 0
    slice_pred_sum = 0
    slice_true_sum =0
    for i in range(np.shape(prediction)[0]): #for each slice
        slice_pred = np.squeeze(prediction[i,:,:])
        slice_true = np.squeeze(ground_truth[i,:,:])
        slice_pred_f = slice_pred.flatten('F')
        slice_true_f = slice_true.flatten('F')               
        slice_pred_sum = slice_pred_sum + sum(slice_pred_f)
        slice_true_sum = slice_true_sum + sum(slice_true_f)
        intersection = intersection + sum(slice_true_f*slice_pred_f)  
    return (2. * intersection + smooth) / (slice_true_sum + slice_pred_sum + smooth)


image_cols = 256
image_rows = 256
num_of_slices = 36
num_of_models = 6

# load testing data      
test_dir_list = os.listdir('test')
dice_coeff_test = dict()

for folder in range(len(test_dir_list)):
    test_data_path = os.path.join('test', test_dir_list[folder])
    test_files = os.listdir(test_data_path)
    
    final_slice_pred = np.ndarray((image_rows,image_cols), dtype=np.uint8)
    final_img_pred = np.ndarray((num_of_slices,image_rows,image_cols), dtype=np.uint8)
    mask_img = np.ndarray((num_of_slices,image_rows,image_cols), dtype=np.uint8) #GT
    
    slice_num=0
    for file_name in test_files:
        if 'mask' in file_name:
            mask_img[slice_num] = cv2.imread(os.path.join(test_data_path, file_name),0)
            img_name  = '_'.join(file_name.split('.')[0].split('_')[:3])

            
            pred_slice_in_model = np.ndarray((num_of_models,image_rows,image_cols), dtype=np.uint8) 
            # matrix of predicted slice in all trained models 
            sum_of_preds = np.ndarray((image_rows,image_cols), dtype=np.uint8) 
            # sum of predicted values for a slice in all trained models

            for model_num in range (0,num_of_models): 
                tmp_path = os.path.join('results'+str(model_num+1), test_dir_list[folder], img_name+'_pred.png')
                pred_slice_in_model[model_num] = cv2.imread(tmp_path,0)
            sum_of_preds[:,:] = sum(pred_slice_in_model[0:num_of_models,:,:])

            final_slice_pred[np.where(sum_of_preds<3)]=0
            final_slice_pred[np.where(sum_of_preds>=3)]=1
            final_img_pred[slice_num] = final_slice_pred
            slice_num += 1

    # to calculate dice_coef_3D
    dice_coeff_test[test_dir_list[folder]] = dice_coef_test_3D(final_img_pred, (mask_img/255))
    print('Dice of 3D-image '+test_dir_list[folder]+':')
    print(dice_coeff_test[test_dir_list[folder]])
mean_dice = float(sum(dice_coeff_test.values())) / len(dice_coeff_test)
print('Mean Dice = ')
print(mean_dice)

