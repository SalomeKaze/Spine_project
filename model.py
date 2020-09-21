# Defenition of the model
from keras.layers import Input, concatenate, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model

#Parameters
ngf = 32
ndf = 32
input_nc = 4
output_nc = 1
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)


# Model :
def get_unet(image_rows, image_cols, n_ch):
    inputs = Input((image_rows, image_cols, n_ch))
    act = 'relu'
    conv1 = Conv2D(32, (3 ,3), activation=act, padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1, momentum=0.99)(conv1)
    conv1 = Conv2D(32, (3, 3), activation=act, padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, momentum=0.99)(conv1)  
    conv1 = Conv2D(32, (3, 3), activation=act, padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, momentum=0.99)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, (3, 3), activation=act, padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation=act, padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99)(conv2)    
    conv2 = Conv2D(64, (3, 3), activation=act, padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99)(conv2)    
    conv2 = Conv2D(64, (3, 3), activation=act, padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99)(conv2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99)(conv3)   
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99)(conv3)   
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99)(conv3)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
       
    conv4 = Conv2D(256, (3, 3), activation=act, padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation=act, padding='same')(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), activation=act, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation=act, padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, momentum=0.99)(conv5)
    conv5 = Conv2D(512, (3, 3), activation=act, padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, momentum=0.99)(conv5)
    conv5 = Conv2D(512, (3, 3), activation=act, padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, momentum=0.99)(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv5), conv4], axis=3)
    
    conv6 = Conv2D(256, (3, 3), activation=act, padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation=act, padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99)(conv6)
    conv6 = Conv2D(256, (3, 3), activation=act, padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99)(conv6)
    conv6 = Conv2D(256, (3, 3), activation=act, padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99)(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=act, padding='same')(up7)
    
    conv7 = Conv2D(128, (3, 3), activation=act, padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99)(conv7)
    conv7 = Conv2D(128, (3, 3), activation=act, padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99)(conv7)
    conv7 = Conv2D(128, (3, 3), activation=act, padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99)(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=act, padding='same')(up8)
    
    conv8 = Conv2D(64, (3, 3), activation=act, padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99)(conv8)
    conv8 = Conv2D(64, (3, 3), activation=act, padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99)(conv8)
    conv8 = Conv2D(64, (3, 3), activation=act, padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99)(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=2, padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same')(up9)
    
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same')(conv9)
    conv9 =  BatchNormalization(axis=-1, momentum=0.99)(conv9)
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same')(conv9)
    conv9 = BatchNormalization(axis=-1, momentum=0.99)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model