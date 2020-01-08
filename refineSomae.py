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

input_folder = "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/"
somae_binary_mask = ReadH5File(input_folder+"Zebrafinch-somae_new-dsp_8.h5",[1])

seg = ReadH5File(input_folder+"Zebrafinch-seg-dsp_8.h5",[1])

somae_raw = seg.copy()
somae_raw[somae_binary_mask==0]=0

cc_labels, n_comp = computeConnectedComp26(somae_raw)
print("Components found: " + str(n_comp))
items_of_component, label_to_cclabel = evaluateLabels(cc_labels, somae_raw, n_comp)

keep_labels = set()

for entry in label_to_cclabel.keys():

    print(entry)
    # print(label_to_cclabel[entry])

    most_points = -1
    largest_comp = -1
    for comp in label_to_cclabel[entry]:
        # print(items_of_component[comp])
        if items_of_component[comp]>most_points:
            largest_comp = comp
            most_points = items_of_component[comp]

    keep_labels.add(largest_comp)

somae_refined = update_labels(keep_labels, somae_raw, cc_labels)

WriteH5File(somae_refined,input_folder+"Zebrafinch-somae_new_refined-dsp_8.h5","main")

# process somae - write somae points and surface points for every block

dsp = 8
# volume_size =       [6144,5632,5632]
block_size =        [512,512,512]
# volume_size_dsp =   [6144/dsp,5632/dsp,5632/dsp]
block_size_dsp =    [int(512/dsp),int(512/dsp),int(512/dsp)]
n_blocks =          [12,11,11]


somae_refined = ReadH5File(input_folder+"Zebrafinch-somae_new_refined-dsp_8.h5",[1])
output_folder = "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/somae_blocks_dsp8_new/"
for bz in range(n_blocks[0]):
    for by in range(n_blocks[1]):
        for bx in range(n_blocks[2]):

            somae_block_dsp = somae_refined[bz*block_size_dsp[0]:(bz+1)*block_size_dsp[0],
                                            by*block_size_dsp[1]:(by+1)*block_size_dsp[1],
                                            bx*block_size_dsp[2]:(bx+1)*block_size_dsp[2]]

            print(somae_block_dsp.shape)

            # somae_block = np.kron(somae_block_dsp, np.ones((dsp,dsp,dsp),dtype=np.uint64))
            # if somae_block.shape[0]!=block_size[0] or somae_block.shape[1]!=block_size[1] or somae_block.shape[2]!=block_size[2]:
            #     raise ValueError("Unknown Error")

            filename_dsp = output_folder+'Zebrafinch-somae_refined_dsp8-{:04d}z-{:04d}y-{:04d}x.h5'.format(bz,by,bx)
            # filename_org = output_folder+'Zebrafinch-somae_refined-{:04d}z-{:04d}y-{:04d}x.h5'.format(bz,by,bx)
            WriteH5File(somae_block_dsp,filename_dsp,   "main")
            # WriteH5File(somae_block,    filename_org,  "main")
