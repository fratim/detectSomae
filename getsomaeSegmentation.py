import numpy as np
from dataIO import *
import time
import h5py
import os
import sys

filepath_seg = "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch-somae-44-dsp8.h5"
filepath_somae = "/home/frtim/Documents/Code/SomaeDetection/yl_cb_160nm_ffn_v2.h5"
output_folder = "/home/frtim/Documents/Code/SomaeDetection/"

size_x = int(5632/8)
size_y = int(6632/8)
size_z = int(6144/8)
color = 12

data_out = np.zeros((size_z, size_y, size_x), dtype=np.uint64)

block_dsp = ReadH5File(filename=filepath_seg,box=[1])
print("seg dsp shape: " + str(block_dsp.shape))

somae = ReadH5File(filename=filepath_somae,box=[1])
print("somae shape: " + str(somae.shape))

data_out[0:somae_onecolor.shape[0],0:somae_onecolor.shape[1],0:somae_onecolor.shape[2]]=somae_onecolor
data_stack = np.add(data_out,block_dsp[somae_onecolor>0])

WriteH5File(data_stack,output_folder+"Zebrafinch-somae-added.h5","main")
