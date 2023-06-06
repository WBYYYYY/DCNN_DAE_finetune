import cv2
import os
import numpy
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import time

imgpath = r'D:\grasp_batch\train_input\NNinput00001.png'
image = cv2.imread(imgpath,2)
image = cv2.resize(image, (128,128), interpolation=cv2.INTER_CUBIC)
x_data = np.array(image)
x_data = x_data.astype(np.float32)
x_data = np.multiply(x_data, 1.0 / 65535.)

img_test = []
img_test.append(x_data)
img_test = np.array(img_test).astype('float32')

def myssimcost(y_true, y_pred):
        im1 = tf.image.convert_image_dtype(y_true, tf.float16)
        im2 = tf.image.convert_image_dtype(y_pred, tf.float16)
        ssim2 = 1-tf.image.ssim(im1, im2, max_val=1.0)
        aaa=tf.reduce_mean(tf.keras.losses.MSE(im1, im2),axis=1)
        aaa=tf.reduce_mean(aaa,axis=1)
        output=ssim2+10*aaa
        return output
time_start = time.time()
model = load_model(r'D:\SMC\U-net\model\DANetV2.h5', custom_objects={'myssimcost': myssimcost})
preds = model.predict(img_test)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img_pre = np.reshape(preds, newshape=(128, 128, 1))
print(img_pre.dtype)
predd = array_to_img(img_pre*65535)
predd.save('test101.png')
time_end=time.time()
print('time cost',time_end-time_start,'s')
