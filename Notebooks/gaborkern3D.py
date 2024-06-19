#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mar 01, 2023
@author: Kevin Fotso
 
"""
__author__ = "Kevin Fotso"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Kevin Fotso"
__email__ = "kevin.fotsotagne@austin.utexas.edu"
__status__ = "released"


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fftpack import fftshift 
from scipy.fft import fftn, ifftn
from scipy import misc
import imageio
import cv2
import matplotlib.pyplot as plt

from numpy.lib import scimath as SM

from scipy import ndimage
from scipy.signal import fftconvolve
from numba import jit

def rotation(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def setFilterBanks():
           
    #This function allows to set multiple filterbanks
    #The filterbanks parameters
    lamb0 = 2
    orientations = 8
    scales = 5
    gamma = 1
    phase = 0
    bandwidth = 1
    overlapIndex = 0.5
    offset = 0
    stop = np.pi - (np.pi / orientations) + offset
    step = np.pi / orientations
    theta = np.linspace(offset, stop, orientations)
    lamb = lamb0
    filters = {}
    numb = 0
    for scale in np.arange(scales, 0, -1):
        lamb0 = lamb
        for th in theta:
            
            result, sig = gaborKern3D(th, lamb, gamma,
                                     bandwidth, phase, 1 / overlapIndex)
            
            #filters[scale,th] = result
            numb +=1
            filters[numb] = result
            
        lamb = lamb0 * (2**bandwidth)
                        
            
            
    # Time to add the DC filter
    f1 =  2 * np.pi / lamb 
    result = gaussKern(f1, sig, overlapIndex)
    filters[numb+1] = result
    
    return filters

def gaussKern(f0, s0, overlapIndex):
    # Double check on this
    
    overlap = SM.sqrt(2* np.log(1 / overlapIndex))
    sigma = s0 * overlap/(s0*f0 - overlap)
    two_sig = 2 * sigma
    n = np.vectorize(complex) (np.ceil(two_sig.real), np.ceil(two_sig.imag)) 
    X, Y, Z = np.mgrid[-n:n+2, -n:n+2, -n:n+2]
    X = X.T
    Y = Y.T
    Z = Z.T
    
    kern = np.exp(-1/2 *(X**2 + Y**2 + Z**2) / (sigma**2))    
    kern /= np.sum(kern)
    
    return kern

def gaborKern3D(theta, lamb, gamma, 
               bandwidth, phase, overlapInd):
    
    #We have to call a special library because npsqrt(negative number )
    #returns Nan    
    qFac = (1/ np.pi) * SM.sqrt((np.log(overlapInd)/2)) * ((2**bandwidth + 1) / (2**bandwidth - 1))
    
    
    sigma = lamb * qFac
    
    #N = np.around(4*sigma, 0) 
    four_sig = 4 * sigma
    N = np.vectorize(complex) (np.ceil(four_sig.real), np.ceil(four_sig.imag)) 
    
    X, Y, Z = np.mgrid[-N:N+2, -N:N+2, -N:N+2]
    X = X.T
    Y = Y.T
    Z = Z.T
    
    #xTheta = X * np.cos(theta) + Y * np.sin(theta)
    #yTheta = -X * np.sin(theta) + Y * np.cos(theta)
    theta_ = [theta, theta, theta]
    R = rotation(theta_) 
    zTheta = Z * R[0,0] + Y * R[0,1] + X * R[0,2]
    yTheta = Z * R[1,0] + Y * R[1,1] + X * R[1,2]
    xTheta = Z * R[2,0] + Y * R[2,1] + X * R[2,2]
    
    gaussian = np.exp(-((xTheta**2) + gamma**2 * (yTheta**2) + gamma**2 * (zTheta**2)) / (2*sigma**2))
    
    gabor = gaussian * np.cos(2*np.pi*xTheta / lamb  + phase)
    
    maxfft = np.max(np.abs(fftn(gabor)))
    gaborfft = fftn(gabor / maxfft)
    final_gabor = np.real(ifftn(gaborfft))

    return final_gabor, sigma

@jit(nopython=True, parallel=True)
def calculateAMFM(img, gabor_):
    #AMFM dictionary
    rows, cols,slices = img.shape  #Assuming that the rows and the columns are the same?
    #Calculate the IA
    #and then normalizing it
    IA = np.absolute(img)             #/ np.sum(fftn(gabor_))
    #Calculating the Phase
    IP = np.angle(img)
       
    IAnorm = img / IA
    IFx = np.zeros(IA.shape)
    IFy = IFx.copy()
    IFz = IFx.copy()
    
   
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            for s in range(1, slices-1):
                IFx[i, j, s] = np.absolute(np.arccos(np.real((IAnorm[i+1,j,s] + IAnorm[i-1,j, s]) /
                                                   (2 * IAnorm[i,j, s]))))

                IFy[i, j, s] = np.absolute(np.arccos(np.real((IAnorm[i,j+1, s] + IAnorm[i,j-1, s]) /
                                                   (2 * IAnorm[i,j, s]))))
                
                IFz[i, j, s] = np.absolute(np.arccos(np.real((IAnorm[i,j, s+1] + IAnorm[i,j, s-1]) /
                                                   (2 * IAnorm[i,j, s]))))
            
    return IA, IP, IFx, IFy, IFz

def dca(scale):
    
    all_keys = list(scale.keys()) 
    N,M,Z = scale[all_keys[0]]['IA'].shape
    #
    IA_ = scale[all_keys[0]]['IA']
    IP_ = scale[all_keys[0]]['IP']
    IFx_ = scale[all_keys[0]]['IFx']
    IFy_ = scale[all_keys[0]]['IFy']
    IFz_ = scale[all_keys[0]]['IFz']
    
    for row in range(0, N):
        for col in range(0,M):
            for slice_ in range(0,Z):
                pos = all_keys[0]
                tmp = scale[pos]['IA'][row,col,slice_]

                for inc_pos in all_keys:

                    if tmp < scale[inc_pos]['IA'][row,col,slice_]:
                        pos = inc_pos
                        tmp = scale[inc_pos]['IA'][row,col,slice_]

                IA_[row, col, slice_] = tmp
                IP_[row, col, slice_] = scale[pos]['IP'][row,col, slice_]
                IFx_[row, col, slice_] = scale[pos]['IFx'][row,col, slice_]
                IFy_[row,col, slice_] = scale[pos]['IFy'][row,col, slice_]
                IFz_[row,col, slice_] = scale[pos]['IFz'][row,col, slice_]
            
                        
    return IA_, IP_, IFx_, IFy_, IFz_

    
if __name__ == "__main__":
    print("voila!!!!")

