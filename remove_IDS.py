import numpy as np
import os
import sys
from dataIO import *


folder_path = "/home/frtim/Documents/Code/SomaeDetection/Mouse/"
output_folder = "/home/frtim/Documents/Code/SomaeDetection/Mouse/"

IDs_remove = [1,3,4,5,8,11,14,15,18,25,26,28,38,40,49,50,51,52,57,58,60,61,62,66,73,77,79,80,89,95,102,106,107,109,110,112,115,116,117,118,119,120,123,125,129,133]

filename_0 = folder_path+"somae_Mouse_773x832x832.h5"
block = ReadH5File(filename=filename_0,box=[1])

for ID in IDs_remove:
    print("Removing " + str(ID))
    block[block==ID]=0

WriteH5File(block,output_folder+"somae_reduced_Mouse_773x832x832.h5","main")
