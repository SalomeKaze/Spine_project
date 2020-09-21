# Train the 2D U-Net 

import os
import click
import datetime
import numpy as np
from scipy.ndimage import rotate
from utils import load_images
from model import get_unet
import matplotlib.pyplot as plt
import random
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from loss import dice_coef_loss, dice_coef
import cv2

# Parameters
trains = 1
img_rows = 256
img_cols = 256
data_path = ''
n_ch = 4
BASE_DIR = 'weights/'
n_aug_angs = 4
train_split = 0.85

def split_data(imgs_train, train_split):
    total_train = np.asarray(range(np.shape(imgs_train)[0]))
    rand_train = np.asarray(random.sample(range(np.shape(imgs_train)[0]), int(np.ceil(np.shape(imgs_train)[0]*train_split))))
    rand_val = np.setdiff1d(total_train, rand_train)
    return rand_train,rand_val


def augment_data_ltrb(imgs_train):
    rotation_angles = [0,90,180,270]    
    aug_img_train = np.zeros((np.shape(imgs_train)[0]*4,np.shape(imgs_train)[1],np.shape(imgs_train)[2],np.shape(imgs_train)[3]))
    img_ctr = 0
    for i in range(np.shape(imgs_train)[0]):
        img = imgs_train[i,:,:,:]
        for j in range (len(rotation_angles)):
            aug_img_train[img_ctr,:,:,:] = rotate_image(img, rotation_angles[j])
            img_ctr+=1       
    return aug_img_train

def rotate_image( im, angle):
    if isinstance(im, str):
        im = cv2.imread(im)
    rows,cols,_ = im.shape
    rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    im_rotated = cv2.warpAffine(im,rotmat,(cols,rows))
    if np.size(np.shape(im_rotated))<3:
        im_rotated = im_rotated[:,:,np.newaxis]
    return im_rotated


def save_all_weights(d, g, epoch_num):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}.h5'.format(epoch_num)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_num)), True)


def train_multiple_outputs(epoch_num,batch_size):
    for train_num in range(1, trains+1):
        # load training data
        train_data_path = os.path.join('Segmentation_Spine','dataset_slices','train')
        files = os.listdir(train_data_path)
        total = len(files) // (n_ch+1)
        
        imgs_train = np.ndarray((total, img_rows, img_cols, n_ch), dtype=np.uint8)
        imgs_train_mask = np.ndarray((total, img_rows, img_cols, 1), dtype=np.uint8)
        
        data = load_images(train_data_path, n_ch)
        imgs_train_mask, imgs_train = data['GTs'], data['imgs']
        
        imgs_train = imgs_train.astype('float32')
        imgs_train_mask = imgs_train_mask.astype('float32')
        # split data
        rand_train,rand_val = split_data(imgs_train, train_split)
        imgs_train_sp = imgs_train[rand_train,:,:,:]
        imgs_val_sp = imgs_train[rand_val,:,:,:]
        masks_train_sp = imgs_train_mask[rand_train,:,:,:]
        masks_val_sp = imgs_train_mask[rand_val,:,:,:]
        # augment data
        imgs_train_sp_aug = augment_data_ltrb(imgs_train_sp)
        imgs_val_sp_aug = augment_data_ltrb(imgs_val_sp)
        
        masks_train_sp_aug =( augment_data_ltrb(masks_train_sp))
        masks_val_sp_aug = (augment_data_ltrb(masks_val_sp)) 
        masks_train_sp_aug = np.round(masks_train_sp_aug/255)
        masks_val_sp_aug=np.round(masks_val_sp_aug/255)
        print(np.max(imgs_train_sp_aug))
        print(np.min(imgs_train_sp_aug))
        print(np.max(masks_train_sp_aug))
        print(np.min(masks_train_sp_aug))
        
        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        
        model = get_unet(img_rows, img_cols, n_ch)
        model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])
        model_checkpoint = ModelCheckpoint(os.path.join(data_path,'learned_model_'+str(train_num)+'.hdf5'), monitor='val_loss', save_best_only=True)
        history = model.fit(imgs_train_sp_aug, masks_train_sp_aug, batch_size=batch_size, epochs=epoch_num ,verbose=1, shuffle=True,
                            validation_data=(imgs_val_sp_aug, masks_val_sp_aug),
                            callbacks=[model_checkpoint])
        
        print ("*"*30)
        print("train_dice_coeff "+str(history.history['dice_coef']))
        print("val_dice_coeff "+str(history.history['val_dice_coef']))
        print("val_loss "+str(history.history['val_loss']))
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['val_dice_coef'])
        plt.plot(history.history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('Dice')
        plt.xlabel('epoch')
        plt.legend(['train', 'Val' , 'Loss'], loc='upper left')
        plt.show()
        print ("*"*30)
        print('training is done successfully')
        
@click.command()
@click.option('--batch_size', default=8, help='Size of batch')
@click.option('--epoch_num', default=2, help='Number of epochs for training')

def train_command(epoch_num, batch_size):
    return train_multiple_outputs(epoch_num,batch_size)


if __name__ == '__main__':
    train_command()
