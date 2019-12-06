import cc3d
import numpy as np
from dataIO import *
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
    cc_labels_known = set()

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
                        if somae_raw[iz,iy,ix] in label_to_cclabel.keys():
                            label_to_cclabel[somae_raw[iz,iy,ix]].append(curr_comp)
                        else:
                            label_to_cclabel[somae_raw[iz,iy,ix]] = [curr_comp]

    return items_of_component, label_to_cclabel

output_folder = "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/"
somae_raw = ReadH5File(output_folder+"Zebrafinch-somae-dsp_8.h5",[1])

cc_labels, n_comp = computeConnectedComp26(somae_raw)
print("Components found: " + str(n_comp))
items_of_component, label_to_cclabel = evaluateLabels(cc_labels, somae_raw, n_comp)

somae_refined = np.zeros((somae_raw.shape),dtype=np.uint16)

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

    # print(largest_comp)
    somae_refined[cc_labels==largest_comp]=entry

WriteH5File(somae_refined,output_folder+"Zebrafinch-somae_refined-dsp_8.h5","main")
