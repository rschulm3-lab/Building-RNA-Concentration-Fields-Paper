# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:43:04 2022

@author: rmish
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import cmcrameri.cm as cmc
from skimage import io, exposure, data
from skimage.transform import warp

from skimage.transform import rescale, resize, downscale_local_mean

def normalizer(arr):
    return (arr - np.min(arr))/(np.max(arr)-np.min(arr))
s = 700
p = int((1024-s)/2)

def channelnorm(im, channel, vmin, vmax):
    c = (im[:,:,channel]-vmin) / (vmax-vmin)
    c[c<0.] = 0
    c[c>1.] = 1
    im[:,:,channel] = c
    return im

#%%Load everything

model_im = np.load('Fig6_model.npy')
model_im1 = resize(model_im, (s,s),preserve_range=True)#,anti_aliasing=True)
model_im2 = np.pad(model_im1,p)
model_im3 = normalizer(model_im2)
plt.figure()
plt.imshow(model_im3,cmap = cmc.oslo)#,vmax=np.max(data_im)*0.5,vmin=np.max(data_im)*0.001)
plt.axis('off')
plt.tight_layout()
plt.savefig('model_padded.svg')
plt.savefig('model_padded.png')

# exp_im1 = np.load('im_adj_difim16_v1.npy')
exp_im1 = np.load('Fig6_data.npy')
# exp_im22 = normalizer(exp_im1)
percentiles = np.percentile(exp_im1, (60, 100))
exp_im2 = exposure.rescale_intensity(exp_im1,out_range=(0,1),in_range=tuple(percentiles))#.astype(np.uint8)


#%%
descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(model_im3)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(exp_im2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors


matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.8,#max_distance = 100,
                              cross_check=True)

#%%
from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.measure import ransac
from skimage.feature import plot_matches

# Select keypoints from the source (image to be registered)
# and target (reference image)
src = keypoints2[matches12[:, 1]][:, ::-1]
dst = keypoints1[matches12[:, 0]][:, ::-1]

model_robust, inliers = ransac((src, dst), SimilarityTransform,
                               min_samples=10, residual_threshold=5, max_trials=500)

#%%warping time
output_shape = exp_im2.shape
warp_exp_im1 = warp(exp_im2, model_robust.inverse, 
               output_shape=output_shape, cval=-1)

warp_exp_im2 = normalizer(warp_exp_im1)
warp_exp_im2 = warp_exp_im2 - 0.5
warp_exp_im2[warp_exp_im2<0] = 0
warp_exp_im2= normalizer(warp_exp_im2)

plt.figure()
plt.imshow(warp_exp_im2,cmap = cmc.oslo)
plt.axis('off')
plt.savefig('exp.svg',pad_inches = 0)
plt.savefig('exp.png',pad_inches = 0)

#%%error cacluations and all
se = ((warp_exp_im2 - model_im3)**2)
err = np.abs(warp_exp_im2-model_im3)
meanerr = np.mean(err)
mse = ((warp_exp_im2 - model_im3)**2).mean(axis=None)
print('mean error aligned: ',meanerr)
print('mse: ',mse)
plt.figure()
plt.imshow(se,cmap='Greys_r')
plt.axis('off')
plt.savefig('spatial_error_aligned.svg',pad_inches = 0)
plt.savefig('spatial_error_aligned.png',pad_inches = 0)
