# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:38:01 2022

@author: rmish
"""
import numpy as np
from scipy import signal
import scipy
import cmcrameri.cm as cm


def lininterp(x,y,c):
    if not(x.is_integer() and y.is_integer()):        
        x_u = int(np.ceil(x))
        x_l = int(np.floor(x))
        xudiff = x_u-x
        xldiff = x-x_l
        y_u = int(np.ceil(y))
        y_l = int(np.floor(y))
        yudiff = y_u-y
        yldiff = y-y_l
        
        lowmean = np.mean([xudiff,yudiff])
        midmean1 = np.mean([xudiff,yldiff])
        midmean2 = np.mean([xldiff,yudiff])
        highmean = np.mean([xldiff,yldiff])
    
        c[x_u,y_u] = highmean
        c[x_l,y_l] = lowmean
        c[x_u,y_l] = midmean2
        c[x_l,y_u] = midmean1
    else:
        c[int(x),int(y)] = 1
    return c

def conv_plotter2(locs,s,kernel):
    clsum = np.zeros((s,s))
    locpairs = []
    for i in range(int(len(locs)/2)):
        xi,yi = locs[2*i],locs[2*i+1]
        while xi > s-1:
            xi = xi-s
        while xi < 0:
            xi = xi + s-1
        # xi = np.clip(xi,0,s-1)
        while yi > s-1:
            yi = yi-s
        while yi < 0:
            yi = yi + s-1
        ci = np.zeros((s,s))
        cl = lininterp(xi,yi,ci)
        clsum =np.add(clsum,cl)
        locpairs.append((xi,yi))
    c_conv = signal.convolve2d(clsum, kernel, boundary='fill', mode='same')
    return c_conv,locpairs

def dkernel(s,b):
    # b=5
    t = np.linspace(-10,10, s)/(b*np.sqrt(2))
    bump = np.exp(-b*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1
    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    return kernel

def f6(params,kernel,turing_target,s,plotter = False):
    clsum = np.zeros((s,s))
    for i in range(int(len(params)/2)):
        xi,yi = params[2*i],params[2*i+1]
        while xi > s-1:
            xi = xi-s
        while xi < 0:
            xi = xi + s-1
        while yi > s-1:
            yi = yi-s
        while yi < 0:
            yi = yi + s-1
        ci = np.zeros((s,s))
        if not(xi.is_integer() and yi.is_integer()):        
            x_u = int(np.ceil(xi))
            x_l = int(np.floor(xi))
            xudiff = x_u-xi
            xldiff = xi-x_l
            y_u = int(np.ceil(yi))
            y_l = int(np.floor(yi))
            yudiff = y_u-yi
            yldiff = yi-y_l
            
            lowmean = np.mean([xudiff,yudiff])
            midmean1 = np.mean([xudiff,yldiff])
            midmean2 = np.mean([xldiff,yudiff])
            highmean = np.mean([xldiff,yldiff])
        
            ci[x_u,y_u] = highmean
            ci[x_l,y_l] = lowmean
            ci[x_u,y_l] = midmean2
            ci[x_l,y_u] = midmean1
        else:
            ci[int(xi),int(yi)] = 1
        # cl = lininterp(xi,yi,ci)
        clsum =np.add(clsum,ci)
        
    c_conv = signal.convolve2d(clsum, kernel, boundary='fill', mode='same')
    # cnorm = (c_conv-np.min(c_conv))/(np.max(c_conv)-np.min(c_conv))
    sums = sum(sum((c_conv-turing_target)**2))
    return sums

def spline_interp(mat):
    s = len(mat)
    xnorm = ynorm = np.linspace(0,1,s)
    fit = scipy.interpolate.RectBivariateSpline(xnorm,ynorm,mat)
    dx2, dy2 = 0.01, 0.01
    x2 = np.arange(0, 1, dx2)
    y2 = np.arange(0, 1, dy2)
    X2, Y2 = np.meshgrid(x2,y2)
    z2 = fit(x2, y2)
    return z2

def scatter_return(arr,s):
    for i in range(len(arr)):
        while arr[i] > s:
            arr[i] = arr[i]-s
        while arr[i] < 0:
            arr[i] = arr[i] + s
    return arr
def normalizer(arr, maxed = []):
    if maxed == []:
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    else:
        return (arr-np.min(arr))/(maxed-np.min(arr))
    
    
def update_plot_v1(i,ax,ai,s,kernel):
    ap = ai.x[i]
    lossf = ai.f[i]
    c, locpairs = conv_plotter2(ap,s,kernel)
    cspline = spline_interp(c)
    ax['B'].imshow(cspline, cmap = cm.oslo)#,vmin=0.3*np.max(c),vmax=0.9*np.max(c))
    ax['B'].set_title('Generation: ' + str(i))
    if ai.accept[i] == True:
        # ax['C'].cla()
        ax['D'].scatter(i,lossf,c='w')
        ax['C'].imshow(cspline, cmap = cm.oslo)#,vmin=0.3*np.max(c),vmax=0.9*np.max(c))
        ax['C'].set_title('New Best Attempt')
        # ax['C'].scatter(apy,apx,c='r',s=25)
    else:
        ax['D'].scatter(i,lossf,c='r')