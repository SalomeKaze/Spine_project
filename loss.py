

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np
from utils import im2double, im2double_tens

image_shape = (256, 256)
smooth = np.finfo(float).eps

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
   return -dice_coef(y_true, y_pred)

def dice_coef_test_3D(y_pred, y_true):
    GT = y_true
    img_pred = y_pred
    intersection = 0
    slice_pred_sum = 0
    slice_GT_sum =0
    for i in range(np.shape(img_pred)[0]):
        slice_pred = np.squeeze(img_pred[i,:,:,:])
        slice_GT = np.squeeze(GT[i,:,:,:])
        slice_pred_f = slice_pred.flatten('F')
        slice_GT_f = slice_GT.flatten('F')               
        slice_pred_sum = slice_pred_sum + sum(slice_pred_f)
        slice_GT_sum = slice_GT_sum + sum(slice_GT_f)
        intersection = intersection + sum(slice_GT_f*slice_pred_f)  
    return (2. * intersection + smooth) / (slice_GT_sum + slice_pred_sum + smooth)
