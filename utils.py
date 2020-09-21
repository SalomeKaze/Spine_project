
import os, os.path
from PIL import Image
import numpy as np
import cv2


img_rows = 256
img_cols = 256


def load_image(path):
    img = cv2.imread(path,0)
    return img

def im2double_tens(im):
    im = im/255
    return im

def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.float) / info.max

def load_images(path, n_ch):
    files = list_image_files(path)
    total = len(files) // (n_ch+1)
    
    imgs = np.ndarray((total, img_rows, img_cols, n_ch), dtype = np.uint8)
    imgs_mask = np.zeros((total, img_rows, img_cols, 1), dtype = np.uint8)
    valid_im_name = []# test
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    i = 0
    for file_name in files:            
        if n_ch == 4: 
            tmp_img = np.ndarray((img_cols, img_rows, n_ch), dtype=np.uint8)
            channels = ['wat', 'fat', 'inn', 'opp']
            tmp_img_slice_name = '_'.join(file_name.split('.')[0].split('_')[:3])
            if 'wat' in file_name:
                valid_im_name.append(tmp_img_slice_name) 
                img_name = tmp_img_slice_name+'_'+channels[0]+'.'+file_name.split('.')[1]
                tmp_img[:,:,0] = load_image(os.path.join(path, img_name))

            else:
                continue
            for j in range(1,len(channels)):
                img_name = tmp_img_slice_name+'_'+channels[j]+'.'+file_name.split('.')[1]
                tmp_img[:,:,j] = load_image(os.path.join(path, img_name))
            tmp_img = cv2.resize(tmp_img,(img_rows, img_cols))
            tmp_img = np.reshape(tmp_img,(img_rows, img_cols,n_ch))
            
            image_mask_name = tmp_img_slice_name+'_mask.'+file_name.split('.')[1]
            tmp_img_mask = load_image(os.path.join(path, image_mask_name))
            tmp_img_mask = cv2.resize(tmp_img_mask,(img_rows,img_cols))
            tmp_img_mask = np.reshape(tmp_img_mask,(img_rows, img_cols,1))
            
            tmp_img = np.array([tmp_img])
            tmp_img_mask = np.array([tmp_img_mask])
            
            imgs[i] = tmp_img
            imgs_mask[i] = tmp_img_mask
            
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1

    return {
        'imgs': np.array(imgs),
        'GTs': np.array(imgs_mask),
        'im_name': valid_im_name
    }