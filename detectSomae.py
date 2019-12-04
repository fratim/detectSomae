## Imports
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from dataIO import*

## Seeding
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, seg_data, somae_data, batch_size=8, image_size=128):
        self.ids = ids
        self.batch_size = batch_size
        self.image_size = image_size
        self.seg_data = seg_data
        self.somae_data = somae_data
        self.on_epoch_end()

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []

        for id_name in files_batch:

            _img_2d = self.seg_data[id_name,:,:]
            _img = np.expand_dims(_img_2d,axis=2)
            # _img = np.concatenate((_img_2d,_img_2d,_img_2d),axis=2)
            # print(min(_img))
            # print(max(_img))

            _mask = self.somae_data[id_name,:,:]
            _mask = np.expand_dims(_mask,axis=2)

            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask  = np.array(mask)

        image = image/1.0
        mask = mask/1.0

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

image_size = 352
train_path = "/home/frtim/Desktop/data/stage1_train"
epochs = 5
batch_size = 8

## Training Ids
seg_filepath = "/home/frtim/Documents/Code/SomaeDetection/Mouse/seg_Mouse_773x832x832.h5"
somae_filepath = "/home/frtim/Documents/Code/SomaeDetection/Mouse/somae_reduced_Mouse_773x832x832.h5"
seg_data = ReadH5File(seg_filepath, [1])
somae_raw = ReadH5File(somae_filepath, [1])

z_max = min(seg_data.shape[0],somae_raw.shape[0])
somae_data = np.zeros((z_max,seg_data.shape[1],seg_data.shape[2]),dtype=np.uint64)
somae_data[:,:somae_raw.shape[1],:somae_raw.shape[2]]=somae_raw[:z_max,:,:]

seg_data = seg_data[:,:,:z_max]

# downsample by 4, take 128x128 and binarize
# seg_data = seg_data[:,:128,:128]
# somae_data = somae_data[:,:128,:128]

seg_data[seg_data>0]=1
somae_data[somae_data>0]=1

seg_data = seg_data[:,::2,::2]
somae_data = somae_data[:,::2,::2]

seg_data = seg_data[:,:352,:352]
somae_data = somae_data[:,:352,:352]

# find maximum z coordinate
train_ids = np.arange(0,z_max)
print("total IDs: " + str(len(train_ids)))

## Validation Data Size
val_data_size = 64

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

print("train IDs: " + str(len(train_ids)))
print("valid IDs: " + str(len(valid_ids)))

gen = DataGen(train_ids, seg_data, somae_data, batch_size=batch_size, image_size=image_size)

# while True:
#     fig = plt.figure(figsize=(20, 12))
#     fig.subplots_adjust(hspace=0.4, wspace=0.4)
#     k = random.randint(0, int((len(train_ids)-1)/batch_size))
#     x, y = gen.__getitem__(k)
#     r = random.randint(0, len(x)-1)
#     ax = fig.add_subplot(1, 2, 1)
#     ax.imshow(np.reshape(x[r], (image_size, image_size)), cmap="gray")
#     ax = fig.add_subplot(1, 2, 2)
#     ax.imshow(np.reshape(y[r], (image_size, image_size)), cmap="gray")
#     plt.show()


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 1))

    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #704 -> 352
    c2, p2 = down_block(p1, f[1]) #352 -> 176
    c3, p3 = down_block(p2, f[2]) #176 -> 88
    c4, p4 = down_block(p3, f[3]) #88  -> 44

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3]) #44 -> 88
    u2 = up_block(u1, c3, f[2]) #88 -> 176
    u3 = up_block(u2, c2, f[1]) #175 -> 352
    u4 = up_block(u3, c1, f[0]) #352 -> 704

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

train_gen = DataGen(train_ids, seg_data, somae_data, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, seg_data, somae_data, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                    epochs=epochs)

## Save the Weights
model.save_weights("UNetW_Mouse.h5")
# model.load_weights("UNetW_Mouse.h5")

## Dataset for prediction

print ("Batch, Image")
while True:
    k = random.randint(0, int((len(valid_ids)-1)/batch_size))
    x, y = valid_gen.__getitem__(k)
    result = model.predict(x)
    result = result > 0.5
    r = random.randint(0, len(x)-1)
    print(str(k) +", " + str(r))

    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(np.reshape(x[r]*255, (image_size, image_size)), cmap="gray")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(np.reshape(y[r]*255, (image_size, image_size)), cmap="gray")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(np.reshape(result[r]*255, (image_size, image_size)), cmap="gray")
    plt.show()
