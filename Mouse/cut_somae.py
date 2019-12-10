import numpy as np
import os
import sys
from dataIO import *
from numba import njit, types
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import Dict


# calculate mean
@njit
def get_mean(somae_in, avg_z, avg_y, avg_x, n_points):
    for iz in range(somae_in.shape[0]):
        for iy in range(somae_in.shape[1]):
            for ix in range(somae_in.shape[2]):
                if somae_in[iz,iy,ix]!=0:

                    avg_z[somae_in[iz,iy,ix]]+=iz
                    avg_y[somae_in[iz,iy,ix]]+=iy
                    avg_x[somae_in[iz,iy,ix]]+=ix

                    n_points[somae_in[iz,iy,ix]]+=1

    return avg_z, avg_y, avg_x, n_points

# calculate std
@njit
def get_std(somae_in, std, avg_z, avg_y, avg_x):
    for iz in range(somae_in.shape[0]):
        for iy in range(somae_in.shape[1]):
            for ix in range(somae_in.shape[2]):
                if somae_in[iz,iy,ix]!=0:

                    # std_z[somae_in[iz,iy,ix]]+=((iz-avg_z[somae_in[iz,iy,ix]])/avg_z[somae_in[iz,iy,ix]])**2
                    # std_y[somae_in[iz,iy,ix]]+=((iy-avg_y[somae_in[iz,iy,ix]])/avg_y[somae_in[iz,iy,ix]])**2
                    # std_x[somae_in[iz,iy,ix]]+=((ix-avg_x[somae_in[iz,iy,ix]])/avg_x[somae_in[iz,iy,ix]])**2
                    dist=np.sqrt(((iz-avg_z[somae_in[iz,iy,ix]])/avg_z[somae_in[iz,iy,ix]])**2+((iy-avg_y[somae_in[iz,iy,ix]])/avg_y[somae_in[iz,iy,ix]])**2+((ix-avg_x[somae_in[iz,iy,ix]])/avg_x[somae_in[iz,iy,ix]])**2)
                    std[somae_in[iz,iy,ix]]+=dist**2

    return std

@njit
def calculate_new_colors(somae_in, avg_z, avg_y, avg_x, std):
    # color everything further than one std in other color
    for iz in range(somae_in.shape[0]):
        for iy in range(somae_in.shape[1]):
            for ix in range(somae_in.shape[2]):
                if somae_in[iz,iy,ix]!=0:
                    curr_Id = somae_in[iz,iy,ix]
                    # dst = np.sqrt(((iz - avg_z[curr_Id])/avg_z[curr_Id])**2 + ((iy - avg_y[curr_Id])/avg_y[curr_Id])**2 + ((ix - avg_x[curr_Id])/avg_x[curr_Id])**2)
                    dst = np.sqrt(((iz - avg_z[curr_Id])/avg_z[curr_Id])**2 + ((iy - avg_y[curr_Id])/avg_y[curr_Id])**2 + ((ix - avg_x[curr_Id])/avg_x[curr_Id])**2)
                    if dst>1*std[curr_Id]:
                        somae_in[iz,iy,ix]=68

    return somae_in


def main():
    folder_path = "/home/frtim/Documents/Code/SomaeDetection/Mouse/"
    output_folder = "/home/frtim/Documents/Code/SomaeDetection/Mouse/"

    filename = folder_path+"somae_reduced_Mouse_773x832x832.h5"
    somae_in = ReadH5File(filename=filename,box=[1])

    seg_IDS = np.unique(somae_in)
    seg_IDS = np.delete(seg_IDS, 0)
    print(seg_IDS)

    avg_z = Dict.empty(key_type=types.float64,value_type=types.float64)
    avg_y = Dict.empty(key_type=types.float64,value_type=types.float64)
    avg_x = Dict.empty(key_type=types.float64,value_type=types.float64)
    std = Dict.empty(key_type=types.float64,value_type=types.float64)
    n_points = Dict.empty(key_type=types.float64,value_type=types.float64)

    for ID in seg_IDS:
        avg_z[ID] = 0
        avg_y[ID] = 0
        avg_x[ID] = 0
        std[ID] = 0
        n_points[ID] = 0

    avg_z, avg_y, avg_x, n_points = get_mean(somae_in, avg_z, avg_y, avg_x, n_points)

    for ID in seg_IDS:
        avg_z[ID] = avg_z[ID]/n_points[ID]
        avg_y[ID] = avg_y[ID]/n_points[ID]
        avg_x[ID] = avg_x[ID]/n_points[ID]

    std = get_std(somae_in, std, avg_z, avg_y, avg_x)

    for ID in seg_IDS:
        std[ID] = np.sqrt(std[ID]/(n_points[ID]-1))

    print("Calculating new colors")

    somae_in = calculate_new_colors(somae_in, avg_z, avg_y, avg_x, std)

    filename = output_folder+"somae_reduced_colored_Mouse_773x832x832.h5"
    WriteH5File(somae_in, filename,   "main")

if 1==True:
    main()
