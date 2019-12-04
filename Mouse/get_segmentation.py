import numpy as np
import os
import sys
from dataIO import *


folder_path = "/home/frtim/Documents/Code/SomaeDetection/Mouse/neurons/"
output_folder = "/home/frtim/Documents/Code/SomaeDetection/Mouse/segmentations/"

somae_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135]
segmentation_ids = []

files = os.listdir(folder_path)
filename_0 = folder_path+str(files[0])
print(filename_0)
block_0 = ReadH5File(filename=filename_0,box=[1])

size_z = block_0.shape[0]
size_y = block_0.shape[1]
size_x = block_0.shape[2]

del block_0

data_out = np.zeros((size_z, size_y, size_x), dtype=np.uint64)

for file in files:
    print("Processing file: " + str(file))
    filename = folder_path+"/"+str(file)
    block = ReadH5File(filename=filename,box=[1])
    neuronID = int(''.join(c for c in file[:-2] if c.isdigit()))
    print("ID is: " + str(neuronID))
    if (block.shape[0]!=size_z or block.shape[1]!=size_y or block.shape[2]!=size_x):
        print("Shape not equal, skipping!")
        print(file)
        print(block.shape)
    elif (neuronID in somae_ids):
        data_out = data_out+(block*neuronID)
        print("Max is: " + str(np.max(data_out)))
        print("Min is: " + str(np.min(data_out)))
        segmentation_ids.append(neuronID)
        del block
    elif (neuronID not in somae_ids):
        print("Id not in Soame IDs, ID: " + str(neuronID))
    else:
        print("UNKNOWN ERROR")

WriteH5File(data_out,output_folder+"Mouse-all.h5","main")

del data_out

print ("--------------------------")
for Id in somae_ids:
    if Id not in segmentation_ids:
        print("This ID is in somae but not in segmentation: " + str(Id))
# IDs in somae but not in segmentation:
# 1,3,4,5,8,11,14,15,18,25,26,28,38,40,49,50,51,52,57,58,60,61,62,66,73,77,79,80,89,95,102,106,107,109,110,112,115,116,117,118,119,120,123,125,129,133
