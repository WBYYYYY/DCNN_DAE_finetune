##  Unet迁移学习（从几何光学模型到GRASP仿真模型）

import cv2
import os
import numpy
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

train_path_nameH=r'D:\SMC\grasp_batch\grasp_batch\train\mini_input/'
train_path_nameT=r'D:\SMC\grasp_batch\grasp_batch\train\mini_label/'

test_path_nameH=r'D:\SMC\grasp_batch\grasp_batch\test\mini_input/'
test_path_nameT=r'D:\SMC\grasp_batch\grasp_batch\test\mini_label/'

imgsize=(128,128)
image_shape=(128,128,1)


# In[ ]:


def imgprocess(imgpath):
        image = cv2.imread(imgpath,2)
        image = cv2.resize(image, imgsize, interpolation=cv2.INTER_CUBIC)
        x_data = np.array(image)
        x_data = x_data.astype(np.float32)
        x_data = np.multiply(x_data, 1.0 / 65535.)
        x_data = x_data.reshape(image_shape)
        return x_data


def imgprocess_L(imgpath):
    image = cv2.imread(imgpath, 2)
    image = cv2.resize(image, imgsize, interpolation=cv2.INTER_CUBIC)
    x_data = np.array(image)
    x_data = x_data.astype(np.float32)
    x_data = np.multiply(x_data, 1.0 / 65535.)
    x_data = np.transpose(x_data)
    x_data = x_data.reshape(image_shape)
    return x_data

# In[ ]:


def reader(datadir_H,datadir_T,batchsize):
    farfield_filenames = sorted(os.listdir(datadir_H))
    phase_filenames = sorted(os.listdir(datadir_T))
    idx = 0
    while 1:
        if idx+batchsize>len(farfield_filenames):
            idx=0
        start = idx
        img_farfields=[]
        img_phases=[]
        for i in range(idx,idx+batchsize):
            farfield_filepath = os.path.join(datadir_H,farfield_filenames[i])
            img_farfield = imgprocess(farfield_filepath)
            img_farfields.append(img_farfield)
            phase_filepath = os.path.join(datadir_T,phase_filenames[i])
            img_phase = imgprocess_L(phase_filepath)
            img_phases.append(img_phase)
        idx=idx+batchsize
        img_farfields = np.array(img_farfields).astype('float32')
        img_phases = np.array(img_phases).astype('float32')
        yield (img_farfields,img_phases)


# In[ ]:


import tensorflow as tf
tf.test.gpu_device_name()


# In[ ]:


from tensorflow.keras.layers import Input, Activation, Add,MaxPooling2D,UpSampling2D,LeakyReLU,Conv2DTranspose,Conv2D,BatchNormalization,Dense,Flatten,LSTM
from tensorflow.keras.layers import Dropout,concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model

def Generator():
    inputs=Input(shape=image_shape)

    x=Conv2D(512,3,padding='same',strides=1,activation='selu')(inputs)
    xj1=BatchNormalization(name='xj1')(x)
    #256,256,3->128,128,3
    x=Conv2D(256,3,padding='same',strides=2,activation='selu')(xj1)
    x=BatchNormalization()(x)

    x=Conv2D(256,3,padding='same',strides=1,activation='selu')(x)
    xj2=BatchNormalization(name='xj2')(x)
    #128,128,3->64,64,3
    x=Conv2D(256,3,padding='same',strides=2,activation='selu')(xj2)
    x=BatchNormalization()(x)

    x=Conv2D(256,3,padding='same',strides=1,activation='selu')(x)
    xj3=BatchNormalization(name='xj3')(x)
    #64,64,3->32,32,3
    x=Conv2D(256,3,padding='same',strides=2,activation='selu')(xj3)
    x=BatchNormalization()(x)

    x=Conv2D(256,3,padding='same',strides=1,activation='selu')(x)
    xj4=BatchNormalization(name='xj4')(x)
    #32,32,3->16,16,3
    x=Conv2D(256,3,padding='same',strides=2,activation='selu')(xj4)
    x=BatchNormalization()(x)

    x=Conv2D(256,3,padding='same',strides=1,activation='selu')(x)
    xj5=BatchNormalization(name='xj5')(x)
    #16,16,3->8,8,3
    x=Conv2D(512,3,padding='same',strides=2,activation='selu')(xj5)
    x=BatchNormalization()(x)
    x=Conv2D(512,3,padding='same',strides=1,activation='selu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(512,3,padding='same',strides=1,activation='selu')(x)
    x=BatchNormalization()(x)

    # ->16,16
    x1 = UpSampling2D(name='us16')(x)
    x1=Conv2D(256,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)
    x1=Conv2D(256,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)

    # ->32,32
    x1 = UpSampling2D(name='us32')(x1)
    x1=Conv2D(256,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)
    x1=Conv2D(256,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)

    # ->64
    x1 = UpSampling2D(name='us64')(x1)
    x1=Conv2D(256,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)
    x1=Conv2D(128,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)

    # ->128
    x1 = UpSampling2D(name='us128')(x1)
    x1=Conv2D(128,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)
    x1=Conv2D(64,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)

    # ->256
    x1 = UpSampling2D(name='us256')(x1)
    x1=Conv2D(64,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)
    x1=Conv2D(32,3,padding='same',strides=1,activation='selu')(x1)
    x1=BatchNormalization()(x1)

    x1=Conv2D(1,3,padding='same',strides=1,activation='selu')(x1)
    print(x1.shape)
    model = Model(inputs=inputs, outputs=x1, name='Generator')
    return model
# model.summary


# In[ ]:


def myssimcost(y_true, y_pred):
    im1 = tf.image.convert_image_dtype(y_true, tf.float32)
    im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
    ssim2 = 1-tf.image.ssim(im1, im2, max_val=1.0)
    aaa=tf.reduce_mean(tf.keras.losses.MSE(im1, im2),axis=1)
    aaa=tf.reduce_mean(aaa,axis=1)
    output=ssim2+10*aaa
    #output = aaa
    return output

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')
train_len = len(os.listdir(train_path_nameH))
test_len = len(os.listdir(test_path_nameH))
batch_size = 10
print(np.ceil(train_len/batch_size))
print(np.ceil(test_len/batch_size))

base_model = Generator()
base_model.load_weights(r'D:\SMC\U-net\test\unet\u_8_9.h5')
for layer in base_model.layers:
    layer.trainable = False

for k in range(0, 20):
    base_model.layers[-k].trainable = True
for k in range(42,52):
    base_model.layers[-k].trainable = False
base_model.summary()
for x in base_model.trainable_weights:
    print(x.name)
print('\n')

#y = Dense(128, activation='selu', kernel_initializer='he_normal')(base_model.output)
#y = Dense(128, activation='selu', kernel_initializer='he_normal')(y)
#y = Dense(1, activation='selu', kernel_initializer='he_normal')(y)

# Full Model: Pre-train Conv + Customized Classifier
#model = Model(inputs=base_model.input, outputs=y, name='Transfer_Learning')
lrate = 0.0001
adam = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=True)
base_model.compile(optimizer=adam, loss=myssimcost, metrics=['mae'])
#model.compile(optimizer=adam, loss=myssimcost, metrics=['mae'])

train_reader = reader(train_path_nameH, train_path_nameT, batch_size)
test_reader = reader(test_path_nameH, test_path_nameT, batch_size)

tensorboard = TensorBoard(log_dir='../data_set/logs', write_images=True, update_freq=1)
model_checkpoint = ModelCheckpoint(r'D:\SMC\U-net\test\unet\u_sym_2.h5', monitor='loss', verbose=1, save_best_only=True)

#hist=model.fit(train_reader,steps_per_epoch=np.ceil(train_len/batch_size),epochs=20,validation_data=test_reader,validation_steps=np.ceil(test_len/batch_size),callbacks=[reduce_lr])
hist = base_model.fit(train_reader,steps_per_epoch=np.ceil(train_len/batch_size),epochs=200,validation_data=test_reader,validation_steps=np.ceil(test_len/batch_size), callbacks=[model_checkpoint, tensorboard])

# In[ ]:


def normal2(x):
    x=x-numpy.amin(x)
    y=x/(numpy.amax(x)-numpy.amin(x))
    return y


# In[ ]:


def getssim(im1,im2):
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1)
    return ssim2


# In[ ]:


#modelpath = r'D:\SMC\U-net\test\unet\tz128_33.h5'
#model.save(modelpath)

