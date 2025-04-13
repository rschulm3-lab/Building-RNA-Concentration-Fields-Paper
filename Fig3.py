# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:29:37 2022

@author: rmish
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# import image_processor_v6 as IP
# from cmcrameri.cm import batlow
import cmcrameri.cm as cmc
import matplotlib.cm as cm2
import matplotlib.colors as mcolors
import os
from scipy.interpolate import interp1d
from scipy import stats
from scipy import interpolate


def s_profile_plotter(profx,t1,foldername='',tot=10, logged = False):
    length = np.size(profx,axis=1)
    ums = np.arange(0,length)/length*1250
    ys=cmc.batlow(np.linspace(0,1,tot+1))
    t1isolate = np.around(t1[range(tot)])
    # tpandas = pd.DataFrame(data=t1isolate).round(2).astype('str') + ' hrs'
    # for l in range(np.size(profx,axis=1)):
    plt.figure(dpi=150)
    plt.tight_layout()
    c = 0
    l=0
    for k in range(tot):
        # if not logged:
        plt.plot(ums,profx[k,:],color=ys[c])
        # else:
        #     if k == 0:
        #         plt.plot(ums,profx[k,:],color=ys[c])
        #     else:
        #         proflogged = [math.log(profx[k,y]) for y in range(503)]
        #         plt.plot(ums,proflogged,color=ys[c])
        c+=1
    fs = 13
    # plt.title('Gel '+str(l)+'X-Profile')
    # if not logged:
    plt.ylabel('Concentration (nM)',fontsize=fs,weight='bold')
    # else:
    #     plt.ylabel('ln(Concentration)',fontsize=fs,weight='bold')
    plt.xlabel('Distance (um)',fontsize=fs,weight='bold')
    plt.xticks(fontsize=fs,weight='bold')
    plt.yticks(fontsize=fs,weight='bold')
    ax1 = plt.gca()
    ax1.fontsize = fs
    normalize = mcolors.Normalize(vmin=t1isolate.T.min(), vmax=t1isolate.T.max())
    scalarmappaple = cm2.ScalarMappable(norm=normalize, cmap=cmc.batlow)
    scalarmappaple.set_array(t1isolate[l])
    cb = plt.colorbar(scalarmappaple)
    cb.set_label('time (hours)',rotation=270,fontsize=fs, labelpad=15)
    cb.ax.tick_params(labelsize=fs)
    # plt.legend(tpandas[l],loc='lower left')
    plt.savefig(foldername+'_gel_ '+str(l)+' x-profile.svg',format='svg',dpi=150)
    plt.savefig(foldername+'_gel_ '+str(l)+' x-profile.png',format='png',dpi=150)
    plt.close()

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx#array[idx]

def integral_width(array, maxi = []): #Characteristic length and normalized integral
    cl_conc = np.max(array)/2

    dists = np.where(np.logical_and(array>=0.95*cl_conc, array<=1.10*cl_conc))[0]
    # print(dists)
    if dists.size == 0:
        clength = 0
    else:
        clength = (dists[-1]-dists[0])/2
    
    if maxi != []:
        maxsum = np.sum(maxi)
    else:
        maxsum = 1.0
    summed = 0
    summed = np.sum(array)/maxsum   #/len(array)     
    
    # integral = summed/len(array)
    return summed, clength

def HWHM(xumtot,ctot):
    z1 = ctot
    yToFind = np.max(z1)/2    
    yreduced = np.array(z1) - yToFind
    freduced = interpolate.UnivariateSpline(xumtot, yreduced, s=0)
    # print(freduced.roots()[-1])
    return freduced.roots()[-1]

#%%Model analysis
tot=10
htp = 8
bm = pd.read_table('Fig3_data.txt',sep=',',header=None)
z = bm.shape[1]
mdist = bm[0]-(bm[0].max()/2)
sigcutoff = 600
n1 = np.where(mdist>-sigcutoff)[0][0]
n2 = np.where(mdist<sigcutoff)[0][-1]
bm2 = bm.iloc[n1:n2]
t0 = np.arange(z-1)*10/60
integrals =  np.zeros(z-1)
widths = np.zeros(z-1)
ww = np.zeros(z-1)
#%%% Model FWHM and integral
for i in range(1,z):
    j=i-1
    integrals[j], widths[j] = integral_width(bm2[i])#,maxi = maxsig[1])
    ww[j] = HWHM(mdist[n1:n2],bm2[i])

half = len(mdist)//2
fs=15

halfmdist = mdist[half:].to_numpy()
w2 = np.array([halfmdist[int(x)] for x in widths])

#%%% model profile plots
plt.figure(dpi=300)
tot=18
ys=cmc.batlow(np.linspace(0,1,tot))
t1isolate = np.around(t0[range(tot)])

for i in range(1,tot):
    plt.plot(mdist[n1:n2],bm2[i],linewidth=2,color=ys[i])
    np.savetxt(str(i)+'model.csv', (mdist[n1:n2],bm2[i]), delimiter=',')

plt.ylabel('Concentration (nM)',fontsize=fs,weight='bold')
# else:
#     plt.ylabel('ln(Concentration)',fontsize=fs,weight='bold')
plt.xlabel(r'x-coord (μm)',fontsize=fs,weight='bold')
plt.xticks(fontsize=fs,weight='bold')
plt.yticks(fontsize=fs,weight='bold')
ax1 = plt.gca()
ax1.fontsize = fs
normalize = mcolors.Normalize(vmin=t1isolate.T.min(), vmax=t1isolate.T.max())
scalarmappaple = cm2.ScalarMappable(norm=normalize, cmap=cmc.batlow)
scalarmappaple.set_array(t1isolate[0])
cb = plt.colorbar(scalarmappaple,ax=ax1)
cb.set_label('time (hours)',rotation=270,fontsize=fs, labelpad=15)
cb.ax.tick_params(labelsize=fs)
plt.savefig('model_profiles_v2.svg',format='svg',dpi=300)
plt.savefig('model_profiles_v2.png',format='png',dpi=300)






