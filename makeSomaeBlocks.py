import numpy as np
import math
import os
import h5py

def ReadH5File(filename):
    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        data = np.array(hf[keys[0]])
    return data

def WriteH5File(data, filename, dataset):
    with h5py.File(filename, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset(dataset, data=data, compression='gzip')

dsp = 8
volumesz = (5700,5456,5332)
all_blocksizes = [(128,1024,1024), (192,1536,1536), (256,2048,2048), (512,512,512), (768,768,768), (1024,1024,1024), (1536,1536,1536), (2048,2048,2048)]

input_folder = "/Users/Tim/Documents/Code/detectSomae/somae_in/"
output_folder = "/Users/Tim/Documents/Code/detectSomae/somae_blocks_out/"
somae_dsp = ReadH5File(input_folder+"somae_filled.h5")
os.mkdir(output_folder)

for blocksz in all_blocksizes:

    blocksz_dsp =  [int(blocksz[0]/dsp),  int(blocksz[1]/dsp),  int(blocksz[2]/dsp)]

    dir_name = output_folder + "somaeblocks-dsp{}-{}x{}x{}/".format(dsp,blocksz[2],blocksz[1],blocksz[0])
    os.mkdir(dir_name)

    nb_z = math.ceil(volumesz[0]/blocksz[0])
    nb_x = math.ceil(volumesz[1]/blocksz[1])
    nb_y = math.ceil(volumesz[2]/blocksz[2])

    print("blocksize: {},{},{}".format(blocksz[2],blocksz[1],blocksz[0]))
    print("nbz:{}, nby:{}, nbx:{}".format(nb_z,nb_y,nb_x))

    for bz in range(nb_z):
        for by in range(nb_y):
            for bx in range(nb_x):

                labels_out = np.zeros((blocksz_dsp[0],blocksz_dsp[1],blocksz_dsp[2]),dtype=np.uint64)

                somae_block_dsp = somae_dsp[bz*blocksz_dsp[0]:(bz+1)*blocksz_dsp[0],
                                                 by*blocksz_dsp[1]:(by+1)*blocksz_dsp[1],
                                                 bx*blocksz_dsp[2]:(bx+1)*blocksz_dsp[2]]

                labels_out[:somae_block_dsp.shape[0],:somae_block_dsp.shape[1],:somae_block_dsp.shape[2]] = somae_block_dsp

                filename_dsp = dir_name+'Zebrafinch-somae_filled_refined_dsp{}-{:04d}z-{:04d}y-{:04d}x.h5'.format(dsp,bz,by,bx)
                WriteH5File(labels_out,filename_dsp,   "main")
