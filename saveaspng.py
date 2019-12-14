from PIL import Image
import numpy as np
from dataIO import *

folder_path = "/home/frtim/Documents/Code/SomaeDetection/Mouse/"
output_folder = "/home/frtim/Documents/Code/SomaeDetection/Mouse/images/"

filename = folder_path+"somae_reduced_Mouse_773x832x832.h5"
somae_in = ReadH5File(filename=filename,box=[1])

print(somae_in.dtype)

for k in range(somae_in.shape[0]):

    data = somae_in[k,:,:]

    im = Image.fromarray(data)
    im.save(output_folder+"img_"+str(k).zfill(4)+".png")
