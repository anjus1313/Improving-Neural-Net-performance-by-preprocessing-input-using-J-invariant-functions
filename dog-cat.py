#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:49:34 2019

@author: anjus
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import os
import imageio
import skimage
from skimage import data, img_as_float, img_as_ubyte
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet, denoise_bilateral
from skimage.restoration import denoise_nl_means, estimate_sigma
def imread(path):
    img = imageio.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img

from scipy.signal import convolve2d
def interpolate_image(x, conv_filter=None):
    if conv_filter is None:
        conv_filter = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    return convolve2d(x, conv_filter, mode = 'same')
def generate_mask(shape, idx, width=3):
    m = np.zeros(shape)
    
    phasex = idx % width
    phasey = (idx // width) % width
    
    m[phasex::width, phasey::width] = 1
    return m
def invariant_denoise(img, width, denoiser):
    
    n_masks = width*width
    
    interp = interpolate_image(img)
    
    output = np.zeros(img.shape)
    
    for i in range(n_masks):
        m = generate_mask(img.shape, i, width=width)
        input_image = m*interp + (1 - m)*img
        input_image = input_image.astype(img.dtype)
        output += m*denoiser(input_image)
    return output

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


from mask import Masker
masker = Masker()

# Define custom loss
def custom_loss(layer):
    

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):        
        y_true = y_true*mask
        y_pred = y_pred*mask
        return y_true*log(y_pred)+(1-y_true)*log(1-y_pred)
   
    # Return a function
    return loss

# Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = custom_loss(layer), metrics = ['accuracy'])
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (224,224 ),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')
from keras.callbacks import ModelCheckpoint
filepath ="dog-cat.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 3,
                         validation_data = test_set,
                         validation_steps = 2000,callbacks=callbacks_list,verbose=2)


'''
'''

test_image = image.load_img('dataset/test_set/cats/cat.4001.jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)    

test_image1 = image.load_img('dataset/test_set/dogs/dog.4003.jpg', target_size = (224, 224))
test_image2 = image.img_to_array(test_image1)

noisy = img_as_ubyte(random_noise(test_image2/255, mode = 'gaussian', var=0.01))
test_image3 = np.expand_dims(noisy, axis = 0)
result = classifier.predict(test_image3)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)   
'''
############################################################################
path = '/home/anju/Anju/Internship/test_set'
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
test = []
labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    for f in os.listdir(path + "/" + category):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_exts:
            continue
        fullpath = os.path.join(path + "/" + category, f)
        img = skimage.transform.resize(imread(fullpath), [224,224, 3])
        img = img.astype('float32')
        img[:,:,0] -= 123.68
        img[:,:,1] -= 116.78
        img[:,:,2] -= 103.94
        test.append(img) # NORMALIZE IMAGE 
        label_curr = i
        labels.append(label_curr)
print ("Num imgs: %d" % (len(test)))
print ("Num labels: %d" % (len(labels)) )
print (ncategories) 

x_test = np.stack(test, axis=0)
y_test = np.stack(labels, axis=0)
'''
classifier.load_weights('dog-cat.h5')
print(classifier.evaluate(x_test,y_test))

noisy=np.zeros(((2023,224,224,3)))
reconstructions_bilateral=np.zeros(((2023,224,224,3)))
invariant_reconstructions_bilateral=np.zeros(((2023,224,224,3)))
for i in range(0,2023):
    #noisy[i] = img_as_ubyte(random_noise(np.transpose(x_test[i],(2,0,1)).astype(np.uint8), seed=42, mode='s&p', amount=0.01,salt_vs_pepper=0.25))
    noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), mode = 'gaussian', var=0.1))

for i in range(0,2023): 
    for j in range(0,3):
        reconstructions_bilateral[i,:,:,j]= denoise_bilateral(noisy[i,:,:,j], sigma_color=0.02, sigma_spatial=15,multichannel=False)
        invariant_reconstructions_bilateral[i,:,:,j] = invariant_denoise(noisy[i,:,:,j], 4, lambda x: 
                                    denoise_bilateral(x, sigma_color = 0.02, mode='wrap', multichannel = False))
print(classifier.evaluate(noisy,y_test))        
print(classifier.evaluate(reconstructions_bilateral,y_test))
print(classifier.evaluate(invariant_reconstructions_bilateral,y_test))    



reconstructions_wavelet=np.zeros(((2023,224,224,3)))
invariant_reconstructions_wavelet=np.zeros(((2023,224,224,3)))

for i in range(0,2023):
    noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), seed=42, mode='s&p', amount=0.01,salt_vs_pepper=0.25))
    #noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), mode = 'gaussian', var=0.1))
for i in range(0,2023): 
    for j in range(0,3):
            reconstructions_wavelet[i,:,:,j] = denoise_wavelet(noisy[i,:,:,j], sigma = 0.12, mode='hard', multichannel = False)
                   
            invariant_reconstructions_wavelet[i,:,:,j] = invariant_denoise(noisy[i,:,:,j], 4, lambda x: 
                                    denoise_wavelet(x, sigma = 0.12, mode='hard', multichannel = False))
print(classifier.evaluate(noisy,y_test))
print(classifier.evaluate(reconstructions_wavelet,y_test))
print(classifier.evaluate(invariant_reconstructions_wavelet,y_test)) 

for i in range(0,2023):
    noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), seed=42, mode='s&p', amount=0.1,salt_vs_pepper=0.25))
    #noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), mode = 'gaussian', var=0.1))
for i in range(0,2023): 
    for j in range(0,3):
            reconstructions_wavelet[i,:,:,j] = denoise_wavelet(noisy[i,:,:,j], sigma = 0.12, mode='hard', multichannel = False)

            invariant_reconstructions_wavelet[i,:,:,j] = invariant_denoise(noisy[i,:,:,j], 4, lambda x: 
                                    denoise_wavelet(x, sigma = 0.12, mode='hard', multichannel = False))
print(classifier.evaluate(noisy,y_test))
print(classifier.evaluate(reconstructions_wavelet,y_test))
print(classifier.evaluate(invariant_reconstructions_wavelet,y_test))   

for i in range(0,2023):
    noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), seed=42, mode='s&p', amount=0.5,salt_vs_pepper=0.25))
    #noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), mode = 'gaussian', var=0.1))
for i in range(0,2023): 
    for j in range(0,3):
            reconstructions_wavelet[i,:,:,j] = denoise_wavelet(noisy[i,:,:,j], sigma = 0.12, mode='hard', multichannel = False)

            invariant_reconstructions_wavelet[i,:,:,j] = invariant_denoise(noisy[i,:,:,j], 4, lambda x: 
                                    denoise_wavelet(x, sigma = 0.12, mode='hard', multichannel = False))
print(classifier.evaluate(noisy,y_test))
print(classifier.evaluate(reconstructions_wavelet,y_test))
print(classifier.evaluate(invariant_reconstructions_wavelet,y_test))            

reconstructions_nl=np.zeros(((2023,224,224,3)))
invariant_reconstructions_nl=np.zeros(((2023,224,224,3)))
for i in range(0,2023):
    noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), seed=42, mode='s&p', amount=0.5,salt_vs_pepper=0.25))
    #noisy[i] = img_as_ubyte(random_noise(x_test[i].astype(np.uint8), mode = 'gaussian', var=0.1))
for i in range(0,2023): 
    for j in range(0,3):
        reconstructions_nl[i,:,:,j] = denoise_nl_means(noisy[i,:,:,j], h=21, fast_mode=True,
                                **patch_kw)/255 

        invariant_reconstructions_nl[i,:,:,j] = invariant_denoise(noisy[i,:,:,j], 4, lambda x: denoise_nl_means(x, h=21, fast_mode=True,
                                **patch_kw))/255 
    
  '''                
'''
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(224,224,3))
gauss = gauss.reshape(224,224,3)
for i in range(0,2023):
    noisy[i]=x_test[i]+gauss*255


'''
