from PIL import Image
import numpy as np
from dataIO import *
import cc3d
from numba import njit

#compute the connected Com ponent labels
def computeConnectedComp26(labels):
    connectivity = 26 # only 26, 18, and 6 are allowed
    cc_labels = cc3d.connected_components(labels, connectivity=connectivity)

    n_comp = np.max(cc_labels) + 1

    return cc_labels, n_comp

# @njit
def evaluateLabels(cc_labels, somae_raw, n_comp):

    items_of_component = dict()
    label_to_cclabel = dict()
    label_to_cclabel_keys = set()
    cc_labels_known = set()

    for j in range(n_comp):
        items_of_component[j]=0

    for iz in range(cc_labels.shape[0]):
        print("iz is: " + str(iz))
        for iy in range(cc_labels.shape[1]):
            for ix in range(cc_labels.shape[2]):

                curr_comp = cc_labels[iz,iy,ix]
                if curr_comp!=0:
                    items_of_component[curr_comp]+=1
                    if curr_comp not in cc_labels_known:
                        cc_labels_known.add(curr_comp)
                        if somae_raw[iz,iy,ix] in label_to_cclabel_keys:
                            label_to_cclabel[somae_raw[iz,iy,ix]].append(curr_comp)
                        else:
                            label_to_cclabel[somae_raw[iz,iy,ix]] = [curr_comp]
                            label_to_cclabel_keys.add(somae_raw[iz,iy,ix])

    return items_of_component, label_to_cclabel

folder_path = "/home/frtim/Documents/Code/SomaeDetection/Mouse/images_cut/"
output_folder = "/home/frtim/Documents/Code/SomaeDetection/Mouse/"

filename_out = output_folder+"somae_reduced_cut_Mouse_773x832x832.h5"

# 773
somae_raw = np.zeros((773,832,832),dtype=np.uint8)

for k in range(773):

    filename_in = folder_path+"_s"+str(k).zfill(3)+".png"
    img = Image.open(filename_in)
    data = np.array(img)

    somae_raw[k,:,:]=data

somae_raw[579,600:602,642]=0
somae_raw[438,307:314,268]=0
somae_raw[668,460,433:441]=0
somae_raw[222:256,301,195:243]=0
somae_raw[76:105,411,317:354]=0
somae_raw[564,422,526:535]=0
somae_raw[708:730,768:792,324]=0

print(np.unique(somae_raw))

cc_labels, n_comp = computeConnectedComp26(somae_raw)
print("Components found: " + str(n_comp))
items_of_component, label_to_cclabel = evaluateLabels(cc_labels, somae_raw, n_comp)

somae_refined = np.zeros((somae_raw.shape),dtype=np.uint16)

print(label_to_cclabel)

for entry in label_to_cclabel.keys():

    print(entry)

    most_points = -1
    largest_comp = -1
    for comp in label_to_cclabel[entry]:
        if items_of_component[comp]>most_points:
            largest_comp = comp
            most_points = items_of_component[comp]

    # print(largest_comp)
    somae_refined[cc_labels==largest_comp]=entry

WriteH5File(somae_refined, filename_out,   "main")
