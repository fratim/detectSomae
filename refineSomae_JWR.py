import cc3d
import numpy as np
from dataIO import *
from numba import njit, types
from numba.typed import Dict

#compute the connected Com ponent labels
def computeConnectedComp26(labels):
    connectivity = 26 # only 26, 18, and 6 are allowed
    cc_labels = cc3d.connected_components(labels, connectivity=connectivity)

    n_comp = np.max(cc_labels) + 1

    return cc_labels, n_comp

float_array = types.int64[:]

@njit
def evaluateLabels(cc_labels, somae_raw, n_comp):

    items_of_component = Dict.empty(key_type=types.int64,value_type=types.int64)
    label_to_cclabel = Dict.empty(key_type=types.int64,value_type=float_array)
    cc_labels_known = set()
    label_to_cclabel_keys = set()

    for j in range(n_comp):
        items_of_component[j]=0

    for iz in range(cc_labels.shape[0]):
        for iy in range(cc_labels.shape[1]):
            for ix in range(cc_labels.shape[2]):

                curr_comp = cc_labels[iz,iy,ix]
                if curr_comp!=0:
                    items_of_component[curr_comp]+=1
                    if curr_comp not in cc_labels_known:
                        cc_labels_known.add(curr_comp)
                        if somae_raw[iz,iy,ix] in label_to_cclabel_keys:
                            add = np.array([curr_comp]).astype(np.int64)
                            label_to_cclabel[somae_raw[iz,iy,ix]] = np.concatenate((label_to_cclabel[somae_raw[iz,iy,ix]].ravel(), add))
                        else:
                            label_to_cclabel[somae_raw[iz,iy,ix]] = np.array([curr_comp],dtype=np.int64).astype(np.int64)
                            label_to_cclabel_keys.add(somae_raw[iz,iy,ix])

    return items_of_component, label_to_cclabel

@njit
def update_labels(keep_labels, labels_in, cc_labels):
    for iz in range(labels_in.shape[0]):
        for iy in range(labels_in.shape[1]):
            for ix in range(labels_in.shape[2]):

                if cc_labels[iz,iy,ix] not in keep_labels:
                    labels_in[iz,iy,ix]=0

    return labels_in

dsp = 4
block_size =        [1024,1024,1024]
block_size_dsp =    [block_size[0]//dsp,block_size[1]//dsp,block_size[2]//dsp]
n_blocks =          [4,4,4]

# # network size used for prediction (x-y), can be smaller due to padding
# network_size = 704
#
input_folder = "/home/frtim/Documents/Code/SomaeDetection/Mouse/gt_data/"
seg_input_fname = input_folder+"seg_JWR_762x832x832.h5"
somae_input_fname = input_folder+"somae_JWR_773x832x832.h5"
somae_refined_output_fname = input_folder+"somae_JWR_refined_773x832x832.h5"
output_folder_blocks = input_folder+"somae_dsp4_{}x{}x{}/".format(block_size[2],block_size[1],block_size[0])

somae_binary_mask = ReadH5File(somae_input_fname,[1])
seg = ReadH5File(seg_input_fname,[1])
# seg = seg[384:,:,:]

if seg.shape[0]!=somae_binary_mask.shape[0]:
    print(somae_binary_mask.shape)
    raise ValueError("Unknown Error")

somae_raw = seg.copy()
somae_raw[somae_binary_mask==0]=0
somae_raw = somae_raw.astype(np.uint64)

cc_labels, n_comp = computeConnectedComp26(somae_raw)

print("Components found: " + str(n_comp))
items_of_component, label_to_cclabel = evaluateLabels(cc_labels, somae_raw, n_comp)

keep_labels = set()

for entry in label_to_cclabel.keys():
    most_points = -1
    largest_comp = -1
    for comp in label_to_cclabel[entry]:
        if items_of_component[comp]>most_points:
            largest_comp = comp
            most_points = items_of_component[comp]

    keep_labels.add(largest_comp)

somae_refined = update_labels(keep_labels, somae_raw, cc_labels)

WriteH5File(somae_refined,somae_refined_output_fname,"main")

# process somae - write somae points and surface points for every block

somae_refined = ReadH5File(somae_refined_output_fname,[1])
for bz in range(n_blocks[0]):
    for by in range(n_blocks[1]):
        for bx in range(n_blocks[2]):

            # labels_out = np.zeros((block_size_dsp[0],block_size_dsp[1],block_size_dsp[2]),dtype=np.uint64)

            somae_block_dsp = somae_refined[bz*block_size_dsp[0]:(bz+1)*block_size_dsp[0],
                                            by*block_size_dsp[1]:(by+1)*block_size_dsp[1],
                                            bx*block_size_dsp[2]:(bx+1)*block_size_dsp[2]]

            labels_out[:somae_block_dsp.shape[0],:somae_block_dsp.shape[1],:somae_block_dsp.shape[2]] = somae_block_dsp

            print(labels_out.shape)

            filename_dsp = output_folder_blocks+'JWR-somae_filled_refined_dsp{}-{:04d}z-{:04d}y-{:04d}x.h5'.format(dsp,bz,by,bx)
            WriteH5File(labels_out,filename_dsp,   "main")
