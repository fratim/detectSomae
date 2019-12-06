## Imports
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow import keras
from dataIO import*

## Seeding
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, depth, seg_data, somae_data, batch_size, image_size):
        self.ids = ids
        self.batch_size = batch_size
        self.image_size = image_size
        self.seg_data = seg_data
        self.somae_data = somae_data
        self.depth = depth
        self.on_epoch_end()

    def __getitem__(self, index):

        if(index+1)*self.batch_size > len(self.ids):
            batch_size_dyn = len(self.ids) - index*self.batch_size
        else:
            batch_size_dyn = self.batch_size

        files_batch = self.ids[index*batch_size_dyn : (index+1)*batch_size_dyn]

        image = []
        mask  = []

        for id_name in files_batch:

            _img = self.seg_data[id_name-self.depth:id_name+self.depth+1,:,:]
            _img = np.moveaxis(_img, 0, -1)
            _mask = self.somae_data[id_name,:,:]
            _mask = np.expand_dims(_mask,axis=2)

            # data augmentation: flip 50% of the time
            if (np.random.uniform()>0.5):
                _img = np.swapaxes(_img,0,1)
                _mask = np.swapaxes(_mask,0,1)

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

class PredictDataGen(keras.utils.Sequence):
    def __init__(self, ids, depth, seg_data, batch_size, image_size):
        self.ids = ids
        self.batch_size = batch_size
        self.image_size = image_size
        self.seg_data = seg_data
        self.depth = depth
        self.on_epoch_end()

    def __getitem__(self, index):

        if(index+1)*self.batch_size > len(self.ids):
            batch_size_dyn = len(self.ids) - index*self.batch_size
        else:
            batch_size_dyn = self.batch_size

        files_batch = self.ids[index*batch_size_dyn : (index+1)*batch_size_dyn]
        image = []
        for id_name in files_batch:
            _img = self.seg_data[id_name-self.depth:id_name+self.depth+1,:,:]
            _img = np.moveaxis(_img, 0, -1)
            image.append(_img)

        image = np.array(image)
        image = image/1.0
        return image

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

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

def UNet(image_size, depth):
    f = [16, 32, 64, 128, 256, 512, 1024]
    # f = [16, 32, 64, 64,  128, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, depth*2+1))

    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #704 -> 352
    c2, p2 = down_block(p1, f[1]) #352 -> 176
    c3, p3 = down_block(p2, f[2]) #176 -> 88
    c4, p4 = down_block(p3, f[3]) #88  -> 44
    c5, p5 = down_block(p4, f[4]) #44  -> 22
    c6, p6 = down_block(p5, f[5]) #22  -> 11

    bn = bottleneck(p6, f[6])

    u0 = up_block(bn, c6, f[5]) #22 -> 44
    u1 = up_block(u0, c5, f[4]) #22 -> 44
    u2 = up_block(u1, c4, f[3]) #44 -> 88
    u3 = up_block(u2, c3, f[2]) #88 -> 176
    u4 = up_block(u3, c2, f[1]) #175 -> 352
    u5 = up_block(u4, c1, f[0]) #352 -> 704

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)
    model = keras.models.Model(inputs, outputs)
    return model

def TrainOnMouse():
    image_size = 704
    epochs = 2
    batch_size = 8
    depth = 5

    #Mouse
    seg_filepath = "/home/frtim/Documents/Code/SomaeDetection/Mouse/seg_Mouse_773x832x832.h5"
    somae_filepath = "/home/frtim/Documents/Code/SomaeDetection/Mouse/somae_reduced_Mouse_773x832x832.h5"

    seg_data = ReadH5File(seg_filepath, [1])
    somae_raw = ReadH5File(somae_filepath, [1])

    z_max = min(seg_data.shape[0],somae_raw.shape[0])

    somae_data = np.zeros((z_max,seg_data.shape[1],seg_data.shape[2]),dtype=np.uint64)
    somae_data[:,:somae_raw.shape[1],:somae_raw.shape[2]]=somae_raw[:z_max,:,:]

    seg_data = seg_data[:,:,:z_max]

    seg_data[seg_data>0]=1
    somae_data[somae_data>0]=1

    seg_data = seg_data[:,:image_size,:image_size]
    somae_data = somae_data[:,:image_size,:image_size]

    # find maximum z coordinate
    all_ids = np.arange(0,z_max)## Validation Data Size
    val_data_size = 64

    valid_ids = all_ids[:val_data_size]
    train_ids = all_ids[val_data_size:]
    train_ids = train_ids[depth:-depth]
    valid_ids = valid_ids[depth:-depth]

    model = UNet(image_size, depth)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    train_gen = DataGen(train_ids, depth, seg_data, somae_data, image_size=image_size, batch_size=batch_size)
    valid_gen = DataGen(valid_ids, depth, seg_data, somae_data, image_size=image_size, batch_size=batch_size)

    train_steps = len(train_ids)//batch_size
    valid_steps = len(valid_ids)//batch_size

    model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
    epochs=epochs)

    # Save the Weights
    model.save_weights("UNetW_Mouse.h5")
    # model.load_weights("UNetW_Mouse.h5")

    ## Dataset for prediction
    print ("Batch, Image")
    for _ in range(12):
        k = random.randint(0, int((len(valid_ids)-1)/batch_size))
        x, y = valid_gen.__getitem__(k)
        result = model.predict(x)
        # result = result > 0.5
        r = random.randint(0, len(x)-1)
        print(str(k) +", " + str(r))

        fig = plt.figure(figsize=(20, 12))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(np.reshape(x[r,:,:,depth], (image_size, image_size)), cmap="gray")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(np.reshape(y[r]*255, (image_size, image_size)), cmap="gray")

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(np.reshape(result[r]*255, (image_size, image_size)), cmap="gray")
        plt.show()

def predictZebrafinch():

    image_size = 704
    batch_size = 8
    depth = 5

    #Mouse
    seg_filepath = "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/Zebrafinch-seg-dsp_8.h5"
    output_folder = "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/"
    seg_data = ReadH5File(seg_filepath, [1])
    somae_out = seg_data.copy()
    
    z_max = seg_data.shape[0]

    seg_data[seg_data>0]=1

    somae_mask = np.zeros((seg_data.shape),dtype=np.uint64)

    if seg_data.shape[1]!=image_size or seg_data.shape[2]!=image_size:
        raise ValueError("Image Size not correct")

    # find maximum z coordinate
    predict_ids = np.arange(0,z_max)## Validation Data Size
    predict_ids = predict_ids[depth:-depth]

    model = UNet(image_size, depth)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    predict_gen = PredictDataGen(predict_ids, depth, seg_data, image_size=image_size, batch_size=batch_size)
    predict_steps = math.ceil(len(predict_ids)/batch_size)

    # Load the Weights
    model.load_weights("UNetW_Mouse.h5")

    # ## Dataset for prediction
    # print ("Batch, Image")
    # for _ in range(12):
    #     k = random.randint(0, int((len(predict_ids)-1)/batch_size))
    #     x = predict_gen.__getitem__(k)
    #     result = model.predict(x)
    #     result = result > 0.5
    #     r = random.randint(0, len(x)-1)
    #     print(str(k) +", " + str(r))
    #     fig = plt.figure(figsize=(20, 12))
    #     fig.subplots_adjust(hspace=0.4, wspace=0.4)
    #
    #     ax = fig.add_subplot(1, 2, 1)
    #     ax.imshow(np.reshape(x[r,:,:,depth], (image_size, image_size)), cmap="gray")
    #
    #     ax = fig.add_subplot(1, 2, 2)
    #     ax.imshow(np.reshape(result[r]*255, (image_size, image_size)), cmap="gray")
    #     plt.show()

    print("steps total: " + str(predict_steps))
    idx_start = depth
    for k in range(0,predict_steps):
        print("step " + str(k) + "...")
        x = predict_gen.__getitem__(k)
        result = model.predict(x)
        result[result<=0.5]=0
        result[result>0.5]=1
        somae_mask[idx_start:idx_start+len(x),:,:]=np.squeeze(result)
        idx_start = idx_start + len(x)

    print(np.min(somae_mask))
    print(np.max(somae_mask))
    somae_out[somae_mask==0]=0

    WriteH5File(somae_out,output_folder+"Zebrafinch-somae-dsp_8.h5","main")



def main():
    # TrainOnMouse()
    predictZebrafinch()

if True == 1:
    main()
