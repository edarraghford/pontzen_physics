import numpy as np 
import scipy
import sys
import math
import radial_data3 as rad 
from matplotlib import cm
import pylab as pl 
import matplotlib.pyplot as plt
from scipy import randn, fft, ifft, real,stats
from scipy.fftpack import fft2, ifft2 
from numpy import arange, pi, cos, sin, zeros, double
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.patches as mpatches 
import pynbody 


def getNyquistInteger(ndata): 
    return int(ndata/2)  

def generateLinearlySpacedFrequencies(nq): 
    x = np.linspace(0,nq-1,nq)
    return x 

def generateLinearlySpacedGridFrequency(nq):  
    x = generateLinearlySpacedFrequencies(nq)
    return x, x, x 

#Transforms 3D function back to frequency domain, averages spatially, 
#and plots result spectrum against the input spectrum
def plt_PSD_avg(fn,grid,ndata, pixsize):
    nq = getNyquistInteger(ndata)
    x, y, z = generateLinearlySpacedGridFrequency(nq)
#    x = np.logspace(0,np.log10((ndata/2)),ndata/2)-1
#    y = np.logspace(0,np.log10((ndata/2)),ndata/2)-1
#    z = np.logspace(0,np.log10((ndata/2)),ndata/2)-1


    xx,yy,zz = np.meshgrid(x,y,z)
    s = pynbody.load("snapshot_999.dms") # load data 
    growth = pynbody.analysis.cosmology.linear_growth_factor(s) 
#compute power spectrum 
    pow = np.absolute(np.fft.fftn(fn,norm="ortho"))**2/ (2*np.pi)**3 /growth**2
    pow1 = np.absolute(np.fft.fftn(grid,norm="ortho"))**2/ (2*np.pi)**3 /growth**2
    fx = rad.radial_data(pow1[:nq,:nq,:nq], x=xx,y=yy,z=zz) 
    fs = rad.radial_data(pow[:nq,:nq,:nq], x=xx,y=yy,z=zz) #collapse to 1D spectrum  
    fx.mean = fx.mean/np.average(fx.mean[1:nq])
    data = np.loadtxt("planck_2015_matterpower.dat") #load planck spectrum 
    fx.mean = fx.mean * 37 
#plot both spectra 
    plt.xscale("log")
    plt.yscale("log") 
    y1 = 2* np.pi / pixsize* np.fft.fftfreq(ndata,d=1) #frequency modes for plot 
#    y1 = np.logspace(np.log10(y[0]),np.log10(y1[ndata/2]),ndata/2 )

    center = (y1[:-1] + y1[1:]) / 2

#    test = np.logspace(0, np.log10(y1[ndata/2-1]+1), ndata/2)
#    test = test-1
#    print test
#    center = (test[:-1] + test[1:]) / 2
#    mean, edges, number = stats.binned_statistic(y1[1:ndata/2], fs.mean[1:ndata/2], 'mean', bins = test)
#    center = (edges[:-1] + edges[1:]) / 2 
#    plt.plot(center, mean)    
    plt.plot(center[1:nq-1],fs.mean[1:nq-1], label='Simulated Power Spectrum') 
    plt.plot(center[1:nq],fx.mean[1:nq], label= 'Simulated Power Spectrum (pre-gridded)')	    
    plt.plot(data[:,0], data[:,1], label = 'Planck Power Spectrum') 
    plt.ylim([0.01,10000])
    plt.xlim([0.1,20])
    plt.xlabel("k(unit of k)") 
    plt.title("Input Power Spectrum and Output Power Spectrum")  
    plt.legend()
    plt.savefig('powerspectrum.png') 
    plt.show() 

#Divide simulated spectrum by physical spectrum, by first binning and averaging data
    bins = 60
    bin1_mean,bin1_edges,bin1_number = stats.binned_statistic(center[1:ndata/2], fs.mean[1:ndata/2], 'mean',bins = bins, range=(min(center[1:ndata/2]),max(center[1:ndata/2]))) #simulation data
    bin2_mean,bin2_edges,bin2_number = stats.binned_statistic(data[:,0], data[:,1], 'mean',bins = bins, range=(min(center[1:ndata/2]),max(center[1:ndata/2]))) #planck data 
    bin3 = bin1_mean/bin2_mean #divide
    x1 = np.linspace(min(center[1:ndata/2]),max(center[1:ndata/2]),bins) 
    plt.title("simulated spectrum/actual spectrum") 
    plt.grid(color = 'b', linestyle = '--') 
    plt.scatter(x1,bin3) 
    plt.savefig('residuals.png') 


def average_spectrum(f):
    a =  np.sum(f)
    b =  a/(ndata)**3
    return (f/b)

filename = sys.argv[1]  
s = pynbody.load(filename) #load simulation 
boxsize = s.properties['boxsize'].in_units("Mpc a h**-1")
ndata = int(np.cbrt(len(s))) 
pixsize = boxsize/ndata 
f = pynbody.sph.to_3d_grid(s, 'rho', ndata, ndata, ndata) 
h = f.in_units("Msol Mpc**-3 a**3 h**3") 
h1 = average_spectrum(h) #divide by average density
grid = np.load("grid-0.npy") 
grid2 = average_spectrum(grid)
p = plt_PSD_avg(h1,grid2,ndata,pixsize)

