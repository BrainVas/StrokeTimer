#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:22:53 2017

@author: m131199
"""
import os
import nibabel as nb
import numpy as np

#def arrangeData(images, labels, fold, dtype):
#
#    [img_rows,img_cols,numImgs] = images.shape
##    a=np.array([16,32,64,128,256])
##    m = numImgs/(a*1.0)
##    fold = int(a[1+np.where((m>=1) & (m<2))[0]])
#    pad = np.zeros((img_rows,img_cols,fold-numImgs)) 
#    images = np.concatenate((images,pad-1024), axis=2)
#    labels = np.concatenate((labels,pad), axis=2)
#    numI = (img_rows/(2*fold))*(img_cols/(2*fold))
#    images= images.reshape(numI,2*fold,2*fold,fold,1).astype(dtype)
#    labels= labels.reshape(numI,2*fold,2*fold,fold,1).astype(dtype)
#    return images,labels,numImgs

def arrangeData(images, labels, fold, dtype):

    [img_rows,img_cols,numImgs] = images.shape
#    a=np.array([16,32,64,128,256])
#    m = numImgs/(a*1.0)
#    fold = int(a[1+np.where((m>=1) & (m<2))[0]])
    pad = np.zeros((img_rows,img_cols,fold-numImgs)) 
    images = np.concatenate((images,pad-1024), axis=2)
    labels = np.concatenate((labels,pad), axis=2)
    images= images.reshape(1,img_rows,img_cols,fold,1).astype('float32')
    labels= labels.reshape(1,img_rows,img_cols,fold,1).astype('uint8')
    return images,labels

def arrange3DtestLabel(labels, fold, dtype):

    [img_rows,img_cols,numImgs] = labels.shape
    pad = np.zeros((img_rows,img_cols,fold-numImgs)) 
    labels = np.concatenate((labels,pad), axis=2)
    labels= labels.reshape(1,img_rows,img_cols,fold,1).astype('uint8')
    return labels
def arrange3DtestImage(vol, fold=48, dtype='float32', pad_value=-1024):
    """
    Split a 3D volume into fixed-depth blocks of shape (Nblocks, H, W, fold, 1).
    
    Parameters
    ----------
    vol : np.ndarray
        Input 3D volume with shape (H, W, D).
    fold : int, optional
        Number of slices per block (default=48).
    dtype : str or np.dtype, optional
        Data type for output array (default='float32').
    pad_value : scalar, optional
        Value to use for padding if the depth D is not divisible by `fold`
        (default=-1024, typical CT background in HU).

    Returns
    -------
    out : np.ndarray
        Output blocks with shape (Nblocks, H, W, fold, 1).
        Nblocks = ceil(D / fold).
    """
    # Ensure input is 3D
    assert vol.ndim == 3, f"Expected 3D volume (H, W, D), got shape {vol.shape}"
    H, W, D = vol.shape

    # Compute how many slices to pad so depth is a multiple of `fold`
    remainder = D % fold
    pad_slices = (fold - remainder) % fold  # =0 if already divisible

    # Pad along depth if necessary
    if pad_slices > 0:
        pad = np.full((H, W, pad_slices), pad_value, dtype=vol.dtype)
        vol = np.concatenate([vol, pad], axis=2)

    # After padding, new depth
    D_new = vol.shape[2]
    nblocks = D_new // fold

    # Allocate output array
    out = np.zeros((nblocks, H, W, fold, 1), dtype=dtype)

    # Slice the volume into consecutive blocks of depth `fold`
    for b in range(nblocks):
        sl = vol[:, :, b * fold:(b + 1) * fold]  # (H, W, fold)
        out[b, :, :, :, 0] = sl.astype(dtype, copy=False)

    return out
