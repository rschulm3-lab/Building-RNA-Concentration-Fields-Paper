# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:49:35 2022

@author: rmish
"""

#%%initialization 
import Fig5_modeling_convolution_v1 as mc
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.optimize import minimize_scalar, minimize, differential_evolution
import cmcrameri.cm as cm
import matplotlib.animation as animation
import time
import cmcrameri.cm as cmc
import skimage.draw as skdraw
plt.style.use('dark_background')
#%%Functions
def dkernel(s,b):
    # b=5
    t = np.linspace(-10,10, s)/(b*np.sqrt(2)) # Misha
    # t = np.linspace(-5,5, s)/(b*np.sqrt(2)) # DongWoo
    bump = np.exp(-b*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1
    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    return kernel

def sumf2(ab,mat,indmeans,s,toplot = False):
    a = ab[0]
    # a=1
    b = ab[1]
    # print(a,b)

    kernel = a*dkernel(s,b)
    cmat = signal.convolve2d(mat, kernel, boundary='fill', mode='same')
    mask = 1-mat
    minus_mask = cmat*mask   
    # print("a", mask)
    # print("b", minus_mask)
# mmasknorm = (minus_mask-np.min(minus_mask))/(np.max(minus_mask)-np.min(minus_mask))
    sumdiff = np.sum(indmeans-minus_mask)**2
    # print(sumdiff)
    if toplot == False:       
        return sumdiff
    else:
        return minus_mask
    
    
def sumf2b(ab,mat,indmeans,s,xlist,toplot = False):
    a = ab[0]
    # a=1
    b = ab[1]
    # print(a,b)
    mask1 = np.zeros([h,w])#,dtype='uint8')
    for ind,row in xlist.iterrows():
        # print(row)
        rr,cc = skdraw.disk((row[0],row[1]),2,shape=[h,w])
        # print(rr,cc)
        mask1[rr,cc] = 1
    kernel = a*dkernel(s,b)
    cmat = signal.convolve2d(mat, kernel, boundary='fill', mode='same')
    mask = 1-mat

    # mmasknorm = (minus_mask-np.min(minus_mask))/(np.max(minus_mask)-np.min(minus_mask))
    # print(sumdiff)
    if toplot == False:      
        minus_mask = cmat*mask*mask1

        sumdiff = np.sum(indmeans*mask*mask1-minus_mask)**2

        return sumdiff
    else:
        minus_mask = cmat*mask

        return minus_mask
    
def remove_labels(ax):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    
def coord2mat(coords,s):
    mat = np.zeros((s,s))
    for idx, coord in coords.iterrows():
        x, y, z = tuple(coord)
        # print(x,y)
        mat[int(x), int(y)] = 1
    return mat


#%% load sender configs
# mats = np.load('four_mats.npy')
s=11

allmats = [pd.read_pickle('Fig5_model_X.pkl'),pd.read_pickle('Fig5_model_random.pkl'),pd.read_pickle('Fig5_model_O.pkl'),
            pd.read_pickle('Fig5_model_fitted_pat.pkl'),pd.read_pickle('Fig5_model_single.pkl'),pd.read_pickle('Fig5_model_clov.pkl')]
indmeansv2 = pd.read_pickle('Fig5_data.pkl')
# indmeansv3 = pd.read_pickle('indmeansv3.pkl')
# indmeansv1 = pd.read_pickle('indmeans.pkl')

#indmeans_V1: with rescale intensity
#indmeans v2: with normalization
#indmeansv3: as close to raw data as their is


# # %% new loss F
x0 = [1,7]
alphas = np.zeros(np.shape(indmeansv2))
betas = np.zeros(np.shape(indmeansv2))
bounds = [(0,15),(0,15)]
a1 = []
b1 = []
a2=[]
b2=[]
h=s
w=s
for ind in [0,1,2,5]:
# for ind in [0,1,2,5]:
# for ind in [5]:
    print('ind: '+str(ind))
    xlist = allmats[ind]*11
    mat = coord2mat(xlist,s)
    cyc = 40
    indm = indmeansv2.at[cyc,ind]        
    res1 = minimize(sumf2,x0 = x0,args = (mat,indm,s),bounds = bounds)#,options={'disp':True})#,bounds = bounds)#,method='Newton-CG')    
    a1.append(res1.x[0])
    b1.append(res1.x[1])
    print('res1: '+str(res1.x[0])+', '+str(res1.x[1]))

    # res2 = differential_evolution(sumf2,x0=x0,args = (mat,indm,s),bounds = bounds,mutation = (0.5,1.9),init='sobol',popsize = 50,recombination=0.1)#,disp=True)#,atol=0.0001)#,disp=True)#,workers=5)#,disp=True)
    # print('res2: '+str(res2.x[0])+', '+str(res2.x[1]))
    # a2.append(res2.x[0])
    # b2.append(res2.x[1])
    
    
    
    
a1m = np.mean(a1)
a1sd = np.std(a1)
b1m = np.mean(b1)
b1sd = np.std(b1)


a2m = np.mean(a2)
a2sd = np.std(a2)
b2m = np.mean(b2)
b2sd = np.std(b2)
#%%Plot updated plots
ab1 = [a1m,b1m]
ab2 = [a1sd, b1sd]
print("a and b____",ab1[0],ab1[1])
print("a and b of std_____",ab2[0],ab2[1])
ab2 = [a2m,b2m]
X1,Y1 =  np.mgrid[0:s, 0:s]
s_size = 200/3

rcolor = '#b60000ff'
scolor = '#4a7dca'
for ind in [0,1,2,5]:
# for ind in [5]:
    fig, ax = plt.subplot_mosaic("AB",dpi=200)

    print('ind: '+str(ind))
    xlist = allmats[ind]*11
    mat = coord2mat(xlist,s)
    [xg,yg] = np.where(mat)

    # cyc = 24

    cyc = 24
    indm = indmeansv2.at[cyc,ind]
    cmat1 = sumf2(ab1,mat,indm,s,toplot = True)
    cmat2 = sumf2(ab2,mat,indm,s,toplot = True)
    # ax["A"].imshow(indm, cmap = cm.oslo)
    cmat1n = (cmat1-np.min(cmat1))/(np.max(cmat1)-np.min(cmat1))
    
    # cmat2n = (cmat2-np.min(cmat2))/(np.max(cmat2)-np.min(cmat2))
    ax["A"].scatter(X1,s-Y1-1,c=indm,s=s_size,cmap = cmc.oslo,linewidth=0,vmin=0.3)#,vmax=0.9)
    ax["A"].scatter(xg,s-yg-1,c=rcolor,s=s_size)
    ax["B"].scatter(X1,s-Y1-1,c=cmat1n,s=s_size,cmap = cmc.oslo,linewidth=0,vmin=0.3)#,vmax=0.9)
    ax["B"].scatter(xg,s-yg-1,c=rcolor,s=s_size)
    ax["A"].set_xlim([-3,s+2])
    ax["A"].set_ylim([-3,s+2])
    ax["B"].set_xlim([-3,s+2])
    ax["B"].set_ylim([-3,s+2])
    # ax["C"].scatter(X1,s-Y1-1,c=np.abs(cmat1n-indm),s=s_size,cmap = 'Greys_r',linewidth=0)#,vmin=0.3)


    # ax["B"].imshow(cmat1, cmap = cm.oslo)
    plt.tight_layout()
    ax["A"].set_axis_off()
    ax["B"].set_axis_off()
    ax["A"].set_aspect('equal', 'box')
    ax["B"].set_aspect('equal', 'box')
    # ax["C"].set_aspect('equal', 'box')
    remove_labels(ax["A"])
    remove_labels(ax["B"])
    plt.savefig('fit_v_model_ind_'+str(ind)+'_v1.svg',bbox_inches='tight')
    plt.savefig('fit_v_model_ind_'+str(ind)+'_v1.png',bbox_inches='tight')
    # remove_labels(ax["C"])

    mse = ((cmat1n - indm)**2).mean(axis=None)
    print("mse is_____",mse)

