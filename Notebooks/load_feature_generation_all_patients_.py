# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nibabel as nib
import numpy as np
import os
from skimage import data, util, measure, morphology
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KernelDensity
import pandas as pd

WORK_DIR="/home/storm/PhD_open_dataset/open_ms_data/cross_sectional/coregistered_resampled/"

# Defining the total bins and the hist feature array
total_bins = 5
features = np.zeros([1, total_bins])
PATIENTS = sorted(os.listdir(WORK_DIR))
IND = 0 

# Create dictionary to retain coordinates of lesions:
total_patients = 30
patient_dict = dict(zip(PATIENTS, [{"indexes" : [], 
                                    "cluster_types" : [],
                                    "total_cluster_types" : [] } for _ in range(total_patients) ]))

for patient_num, DIR in enumerate(PATIENTS):
    
    print("####### Working on patient number {} #########".format(patient_num ))
    patient = os.path.join(WORK_DIR, DIR)

    # Loading the FLAIR dataset
    flair_path = os.path.join(patient, "FLAIR.nii.gz")
    lesion_path = os.path.join(patient, "consensus_gt.nii.gz")
    
    flair_nii = nib.load(flair_path)
    gt_nii = nib.load(lesion_path)
    
    # Transforming data to numpy array
    flair_arr = flair_nii.get_fdata()
    gt_arr = gt_nii.get_fdata().astype(int)
    
    # Preprocessing step:
    #We clean the lesion volume of a very small lesions
    thresh_mask = morphology.remove_small_objects(gt_arr, min_size=15) 
    label_image = measure.label(thresh_mask, connectivity=gt_arr.ndim)
       
    #Connected components
    props = measure.regionprops(label_image)
    properties = ['label', 'area', 'centroid']
    props_table = measure.regionprops_table(label_image,
                           properties=properties)
    
    data = pd.DataFrame(props_table)


    # Get only the lesions based on the ground truth.
    # lesion_only = flair_arr * gt_arr
    
    #
    sl, x, y = flair_arr.shape
    
    #features = np.zeros([props_table['label'].shape[0], total_bins])
    
    if patient_num == 0:
        patient_dict[DIR]["indexes"].append(IND)
    else:
        patient_dict[DIR]["indexes"].append(IND + 1)
        
    for ind in np.nditer(props_table['label']): 
        region = label_image==ind
        lesion_only = flair_arr * region
        
        
        # histogram
        les = np.where(lesion_only > 0)
        h, edges = np.histogram(lesion_only[les], bins=total_bins)
        
        # Applying kernel density
        #h = h[:, np.newaxis]
        #kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(h)
        #h_kde = kde.score_samples(kde)
        
        if patient_num == 0 and ind ==0:
            features[0, :] = h
        else:        
            features = np.vstack((features, h))
            #features[ind -1] = h
            
            
    #
    IND += np.ndarray.item(ind)
    patient_dict[DIR]["indexes"].append(IND)


# Clustering for all the patients    
total_clusters = 3
clustering = AgglomerativeClustering(n_clusters=total_clusters, 
                                     ).fit(features)

# Print Clustering #
print("Total clusters are {}".format(clustering.n_clusters))

for patient_ in patient_dict.keys():
    start_ind = patient_dict[patient_]["indexes"][0]
    end_ind = patient_dict[patient_]["indexes"][1]
    
    cluster_types = np.unique(clustering.labels_[start_ind:end_ind])
    patient_dict[patient_]["cluster_types"] = cluster_types.tolist()
    patient_dict[patient_]["total_cluster_types"] = len(cluster_types.tolist())

# Converting into a dataframe and saving as csv:
SOFTWARE_DIR = "/home/storm/software"
pd_clusters = pd.DataFrame.from_dict(patient_dict)
pd_clusters.to_csv(os.path.join(SOFTWARE_DIR, "cluster_types.csv"))
