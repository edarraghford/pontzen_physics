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

repeat = 1 
ndata = 256 
boxsize = 50.0 
pixsize = boxsize/ndata 
duplicate = 0 
 

# Input power spectrum 
def power_spectrum(x,p):
    return x**p 

# Generates function from input power spectrum 
def function_generator(n,p):
    re=np.zeros((n,n,n))
    im=np.zeros((n,n,n))
    x = np.linspace(1,(n/2)-1,n/2-1)
    y = np.linspace(1,(n/2)-1,n/2-1)
    z = np.linspace(1,(n/2)-1,n/2-1)
    X,Y,Z = np.meshgrid(x,y,z)
    vspectrum = np.vectorize(power_spectrum)
    T = np.sqrt(X**2+Y**2+Z**2)
    mag = np.sqrt(vspectrum(T,p)) 
    pha = 2.0*pi*np.random.randn((n/2-1),(n/2-1),(n/2-1)) 
    re[1:n/2,1:n/2,1:n/2] = mag*cos(pha)*np.random.randn((n/2-1),(n/2-1),(n/2-1))
    im[1:n/2,1:n/2,1:n/2]  = mag*sin(pha)*np.random.randn((n/2-1),(n/2-1),(n/2-1))
    re[n/2+1:,n/2+1:,n/2+1:] = np.flip(np.flip(np.flip((re[1:n/2,1:n/2,1:n/2]),2),1),0)        
    im[n/2+1:,n/2+1:,n/2+1:] = np.negative(np.flip(np.flip(np.flip((im[1:n/2,1:n/2,1:n/2]),2),1),0))
    im[n/2,n/2,n/2] = 0.0
    res = np.fft.ifftn(re+complex(0,1)*im,norm="ortho")
    return real(res) 

# Plot function generated from power spectrum  
def plot_data(fn,d): 
    plt.imshow(fn, cmap=cm.coolwarm, extent = [0,50,50,0])
    plt.title("3D spatial function generated from power spectrum")
    plt.xlabel("x")
    plt.ylabel("y")  
    plt.colorbar() 
    plt.show()

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
def plt_PSD_avg(fn,grid):
    ndata = 256 
    nq = getNyquistInteger(ndata)
    x, y, z = generateLinearlySpacedGridFrequency(nq)
    b = np.logspace(0,np.log10((ndata/2+1)),ndata/2+1)-1
#    y = np.logspace(0,np.log10((ndata/2)),ndata/2)-1
#    z = np.logspace(0,np.log10((ndata/2)),ndata/2)-1
    xx,yy,zz = np.meshgrid(x,y,z)
    s = pynbody.load("snapshot_999.dms") # load data
    growth = pynbody.analysis.cosmology.linear_growth_factor(s)
    pow = np.absolute(np.fft.fftn(fn,norm="ortho"))**2/(2*np.pi)**3/growth**2 	
    grid1 = np.absolute(np.fft.fftn(grid,norm="ortho"))**2/ (2*np.pi)**3 /growth**2
    pow1 = np.zeros(ndata/2) 
    n = np.zeros(ndata/2) 
    for l in range(ndata/2):  
        for i in range(ndata/2):
            for j in range(ndata/2):
                for k in range(ndata/2):
                    if ((np.sqrt(i**2+j**2+k**2) >= l) and (np.sqrt(i**2+j**2+k**2) < (l+1))): 
                        pow1[l] += pow[i][j][k]
                        n[l] = n[l]+1 
    pow1 = pow1/(n) 

    fx = np.zeros(ndata/2)
    p = np.zeros(ndata/2)
    for l in range(ndata/2):
        for i in range(ndata/2):
            for j in range(ndata/2):
                for k in range(ndata/2):
                    if ((np.sqrt(i**2+j**2+k**2) > l) and (np.sqrt(i**2+j**2+k**2) < (l+1))):
                        fx[l] += grid1[i][j][k]
                        p[l] = p[l]+1
    fx = fx/(p)                   
    fs = rad.radial_data(pow[:ndata/2,:ndata/2,:ndata/2], x=xx,y=yy,z=zz) 
    print pow1 

    data = np.loadtxt("planck_2015_matterpower.dat")
    fx = fx/np.average(fx[1:nq]) 
    fx = fx * 37
    plt.xscale("log")
    plt.yscale("log")  
    y1 = 2* np.pi / pixsize* np.fft.fftfreq(256,d=1)
    center = (y1[:-1] + y1[1:]) / 2
 #   test = np.logspace(np.log10(center[0]+1), np.log10(center[ndata/2-2]+1), ndata/2-1)
#    test = test-1
#    center = (y1[:-1] + y1[1:]) / 2
    plt.plot(center[1:nq],pow1[1:nq], label='mine') 	    
    plt.plot(center[1:nq],fx[1:nq], label = 'pre-gridded')
    plt.plot(data[:,0], data[:,1]) 
    plt.ylim([0.01,10000])
    plt.xlim([0.1,20])
#    plt.plot(x,power_spectrum(x,p))#*max(fs.mean[1:ndata/2]))
    plt.title("Input Power Spectrum and Output Power Spectrum") 
#    blue_patch = mpatches.Patch(color='blue', label = 'Output Power Spectrum') 
#    orange_patch = mpatches.Patch(color='orange',label = 'Input Power Spectrum') 
    plt.legend() 
    plt.show()



#    return fs 

def average_spectrum(f):
    a =  np.sum(f)
    b =  a/(ndata)**3
    return (f/b)

s = pynbody.load("snapshot_999.dms")
f = pynbody.sph.to_3d_grid(s, 'rho', 256, 256, 256)
h = f.in_units("Msol Mpc**-3 a**3 h**3") 
h1 = average_spectrum(h)
grid = np.load("grid-0.npy")
grid2 = average_spectrum(grid)
p = plt_PSD_avg(h1,grid2)

