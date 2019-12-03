import numpy as np
from dataIO import *
import time
import h5py
import os
import sys

folder_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/"
output_folder = "/home/frtim/Documents/Code/SomaeDetection/"

size_x = int(5632/8)
size_y = int(6632/8)
size_z = int(6144/8)
slice_z = int(128/8)

data_out = np.zeros((size_z, size_y, size_x), dtype=np.uint64)

for bz in range(0,45):
  filename = folder_path+"/"+str(bz*128).zfill(4)+".h5"
  block = ReadH5File(filename=filename,box=[1])
  block_dsp = block[::8,::8,::8]
  print(block.dtype)
  print(block.shape)
  print(block_dsp.shape)

  data_out[(bz*slice_z):(bz*slice_z)+block_dsp.shape[0],0:block_dsp.shape[1],0:block_dsp.shape[2]]=block_dsp
  del block
  del block_dsp

WriteH5File(data_out,output_folder+"Zebrafinch-somae-44-dsp8.h5","main")

del data_out
