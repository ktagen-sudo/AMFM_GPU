#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:28:52 2023

@author: storm
"""

import nibabel as nib
import numpy as np
import os
from gaborkern3D import setFilterBanks, calculateAMFM, dca
import pandas as pd
from skimage import data, util, measure, morphology
from scipy.ndimage.morphology import distance_transform_edt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from scipy.signal import hilbert
from scipy.fft import fftn, ifftn
from scipy import misc
import imageio
import cv2
from scipy.fftpack import fftshift 
from scipy.fft import fft2
from scipy.fft import fftn
from scipy import ndimage
from scipy.signal import fftconvolve

WORK_DIR="/home/storm/PhD_open_dataset/open_ms_data/cross_sectional/coregistered_resampled/"
listID = sorted(os.listdir(WORK_DIR))

##########################################################################

key_mag_IF_high = "3dt1_mag_IF_high"
key_mag_IF_medium = "3dt1_mag_IF_medium"
key_mag_IF_low = "3dt1_mag_IF_low"

key_ang_IF_high = "3dt1_ang_IF_high"
key_ang_IF_medium = "3dt1_ang_IF_medium"
key_ang_IF_low = "3dt1_ang_IF_low"

key_IA_high = "3dt1_IA_high"
key_IA_medium = "3dt1_IA_medium"
key_IA_low = "3dt1_IA_low"


print("{0}".format(80 * "-"))
print("..............Advanced AMFM Gabor processing begins.............")
print("{0}".format(80 * "-"))

for patient_num, DIR in enumerate(listID):
    
    print("####### Working on patient number {} #########".format(patient_num ))
    patient = os.path.join(WORK_DIR, DIR)

    # Identifying the FLAIR dataset
    flair_path = os.path.join(patient, "FLAIR.nii.gz")
    lesion_path = os.path.join(patient, "consensus_gt.nii.gz") 
    
    # Naming the files to save
    dirSav, NameSav = os.path.split(flair_path)
    
    # Ground truth filename
    NameSav_h = NameSav.split('.')[0] + '_advanced_gabor_amfm_high.nii.gz'
    NameSav_m = NameSav.split('.')[0] + '_advanced_gabor_amfm_medium.nii.gz'
    NameSav_l = NameSav.split('.')[0] + '_advanced_gabor_amfm_low.nii.gz'
    
    NameSav_ang_h = NameSav.split('.')[0] + '_advanced_gabor_amfm_ang_high.nii.gz'
    NameSav_ang_m = NameSav.split('.')[0] + '_advanced_gabor_amfm_ang_medium.nii.gz'
    NameSav_ang_l = NameSav.split('.')[0] + '_advanced_gabor_amfm_ang_low.nii.gz'
    
    NameSav_ia_h = NameSav.split('.')[0] + '_advanced_gabor_amfm_ia_high.nii.gz'
    NameSav_ia_m = NameSav.split('.')[0] + '_advanced_gabor_amfm_ia_medium.nii.gz'
    NameSav_ia_l = NameSav.split('.')[0] + '_advanced_gabor_amfm_ia_low.nii.gz'
    
    high_scale_Sav = os.path.join(dirSav, NameSav_h)
    medium_scale_Sav = os.path.join(dirSav, NameSav_m)
    low_scale_Sav = os.path.join(dirSav, NameSav_l)
    
    high_scale_ang_Sav = os.path.join(dirSav, NameSav_ang_h)
    medium_scale_ang_Sav = os.path.join(dirSav, NameSav_ang_m)
    low_scale_ang_Sav = os.path.join(dirSav, NameSav_ang_l)
    
    high_scale_ia_Sav = os.path.join(dirSav, NameSav_ia_h)
    medium_scale_ia_Sav = os.path.join(dirSav, NameSav_ia_m)
    low_scale_ia_Sav = os.path.join(dirSav, NameSav_ia_l)
    
    # Loading
    flair_nii = nib.load(flair_path)
    affine = flair_nii.affine
    hdr = flair_nii.header
    gt_nii = nib.load(lesion_path)
    
    # Transforming data to numpy array
    flair_arr = flair_nii.get_fdata()
    gt_arr = gt_nii.get_fdata().astype(int)
    
    # Getting shape
    sl, x_flair, y_flair = flair_arr.shape

    gt_arr = gt_arr > 0
    
    # Setting filterbanks
    filters = setFilterBanks()  

    #AMFM part 
    AMFM = {}
    hImg = hilbert(flair_arr)
    
    # Convolution part
    for key in list(filters.keys()):
        AMFM[key] = {}
        AMFM[key]['IA'] = {}  
        AMFM[key]['IP'] = {}
        AMFM[key]['IFx'] = {}
        AMFM[key]['IFy'] = {}
        AMFM[key]['IFz'] = {}
        filter_ = filters[key]
        filterImg = fftconvolve(hImg, filter_, mode='same')

        print("{0}".format(80 * "-"))
        print(".................... Calculating AMFM .....................")
        print("{0}".format(80 * "-"))
        IA, IP, IFx, IFy, IFz = calculateAMFM(filterImg, filter_)
        AMFM[key]['IA'] = IA
        AMFM[key]['IP'] = IP
        AMFM[key]['IFx'] = IFx
        AMFM[key]['IFy'] = IFy
        AMFM[key]['IFz'] = IFz 
        
    #Getting the different scales
    scale_dict = {}
    scale_dict['high'] = {}
    scale_dict['med'] = {}
    scale_dict['low'] = {}
    scale_dict['dc'] = {}

    for key in filters.keys():
        if key <= 8:
            scale_dict['high'][key] = {}
            scale_dict['high'][key] = AMFM[key]

        elif key > 8 and key <= 24:
            scale_dict['med'][key] = {}
            scale_dict['med'][key] = AMFM[key]   
        elif key > 24 and key <= 40:
            scale_dict['low'][key] = {}
            scale_dict['low'][key] = AMFM[key] 
        else:
            scale_dict['dc'][key] = {}
            scale_dict['dc'][key] = AMFM[key]


    IAl, IPl, IFxl, IFyl, IFzl = dca(scale_dict['low'])
    IAm, IPm, IFxm, IFym, IFzm = dca(scale_dict['med'])
    IAh, IPh, IFxh, IFyh, IFzh = dca(scale_dict['high'])
    IAdc, IPdc, IFxdc, IFydc, IFzdc = dca(scale_dict['dc'])
    
    
    # Calculating the mag IF for medium and high scale
    print("{0}".format(80 * "-"))
    print("..............AMFM calculation high, medium & low scale.............")
    print("{0}".format(80 * "-"))
 
    mag_IF_m = np.sqrt(IFxm**2 + IFym**2 + IFzm**2)
    mag_IF_h = np.sqrt(IFxh**2 + IFyh**2 + IFzh**2)
    mag_IF_l = np.sqrt(IFxl**2 + IFyl**2 + IFzl**2)
    
    print("{0}".format(80 * "-"))
    print("..............Compute angle IF .........................")
    print("{0}".format(80 * "-"))
                                         
    # Reshaping the arrays for the dot product
   
    tmp_dot_ang_m = np.arctan2(IFxm, IFym)
    tmp_dot_ang_h = np.arctan2(IFxh, IFyh)
    tmp_dot_ang_l = np.arctan2(IFxl, IFyl)
    print("shape phase is", tmp_dot_ang_l.shape)
    
    ang_IF_m = np.arctan2(tmp_dot_ang_m, IFzm)
    ang_IF_l = np.arctan2(tmp_dot_ang_l, IFzl)
    ang_IF_h = np.arctan2(tmp_dot_ang_h, IFzh)
    
    # Reshaping arrays
    #ang_IF_m = ang_IF_m.reshape( (x_flair,-1,y_flair))
    #ang_IF_h = ang_IF_h.reshape( (x_flair,-1,y_flair))
    #ang_IF_l = ang_IF_l.reshape( (193,-1,193))
    
    medium_scale_lesions = mag_IF_m * gt_arr
    high_scale_lesions = mag_IF_h * gt_arr
    low_scale_lesions = mag_IF_l * gt_arr
     
    medium_scale_ang_lesions = ang_IF_m * gt_arr
    high_scale_ang_lesions = ang_IF_h * gt_arr
    low_scale_ang_lesions = ang_IF_l * gt_arr
     
    medium_scale_ia_lesions = IAm * gt_arr
    high_scale_ia_lesions = IAh * gt_arr
    low_scale_ia_lesions = IAl * gt_arr
   
   
    # Converting the zeros to nan
    medium_scale_lesions[medium_scale_lesions == 0] = np.nan
    high_scale_lesions[high_scale_lesions == 0] = np.nan
    low_scale_lesions[low_scale_lesions == 0] = np.nan
     
    medium_scale_ang_lesions[medium_scale_ang_lesions == 0] = np.nan
    high_scale_ang_lesions[high_scale_ang_lesions == 0] = np.nan
    low_scale_ang_lesions[low_scale_ang_lesions == 0] = np.nan
     
    medium_scale_ia_lesions[medium_scale_ia_lesions == 0] = np.nan
    high_scale_ia_lesions[high_scale_ia_lesions == 0] = np.nan
    low_scale_ia_lesions[low_scale_ia_lesions == 0] = np.nan
    
    # Saving AMFM
    print("{0}".format(80 * "-"))
    print("..............SAVIng HIgh, medium & low scale.............")
    print("{0}".format(80 * "-"))
    
    nib_mag_IF_h = nib.Nifti1Image(high_scale_lesions, affine, header=hdr)
    nib.save(nib_mag_IF_h, high_scale_Sav)
                                           
    nib_mag_IF_m = nib.Nifti1Image(medium_scale_lesions, affine, header=hdr)
    nib.save(nib_mag_IF_m, medium_scale_Sav)
     
    nib_mag_IF_l = nib.Nifti1Image(low_scale_lesions, affine, header=hdr)
    nib.save(nib_mag_IF_l, low_scale_Sav)
     
    # Saving phase IF
    nib_ang_IF_h = nib.Nifti1Image(high_scale_ang_lesions, affine, header=hdr)
    nib.save(nib_ang_IF_h, high_scale_ang_Sav)
                                           
    nib_ang_IF_m = nib.Nifti1Image(medium_scale_ang_lesions, affine, header=hdr)
    nib.save(nib_ang_IF_m, medium_scale_ang_Sav)
     
    nib_ang_IF_l = nib.Nifti1Image(low_scale_ang_lesions, affine, header=hdr)
    nib.save(nib_ang_IF_l, low_scale_ang_Sav)
     
    # Saving IA
    nib_IA_h = nib.Nifti1Image(high_scale_ia_lesions, affine, header=hdr)
    nib.save(nib_IA_h, high_scale_ia_Sav)
                                           
    nib_IA_m = nib.Nifti1Image(medium_scale_ia_lesions, affine, header=hdr)
    nib.save(nib_IA_m, medium_scale_ia_Sav)
     
    nib_IA_l = nib.Nifti1Image(low_scale_ia_lesions, affine, header=hdr)
    nib.save(nib_IA_l, low_scale_ia_Sav)