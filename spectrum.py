import numpy as np 
import scipy
import sys
import math
import radial_data3 as rad 
from matplotlib import cm
import pylab as pl 
import matplotlib.pyplot as plt
from scipy import randn, fft, ifft, real
from scipy.fftpack import fft2, ifft2 
from numpy import arange, pi, cos, sin, zeros, double
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.patches as mpatches 
import pynbody 

repeat = 1 
ndata = 256 
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

#Transforms 3D function back to frequency domain, averages spatially, 
#and plots result spectrum against the input spectrum
def plt_PSD_avg(fn,p):
    x = np.linspace(1,(ndata/2),ndata/2)
    y = np.linspace(1,(ndata/2),ndata/2)
    z = np.linspace(1,(ndata/2),ndata/2)
#    print(x)
    xx,yy,zz = np.meshgrid(x,y,z)
    pow = np.absolute(np.fft.fftn(fn,norm="ortho"))**2 
    f = np.zeros(ndata/2) 
    R = np.sqrt(xx**2+yy**2+zz**2) 
    print pow	
    fs = rad.radial_data(pow[1:ndata/2+1,1:ndata/2+1,1:ndata/2+1], x=xx,y=yy,z=zz) 
#    print(fs.mean[1:ndata/2+1])
    plt.plot(x,fs.mean[1:ndata/2+1]) 	    
#    plt.plot(x,power_spectrum(x,p))#*max(fs.mean[1:ndata/2]))
    plt.title("Input Power Spectrum and Output Power Spectrum") 
    blue_patch = mpatches.Patch(color='blue', label = 'Output Power Spectrum') 
    orange_patch = mpatches.Patch(color='orange',label = 'Input Power Spectrum') 
    plt.legend(handles=[blue_patch,orange_patch]) 
    plt.show()
    return fs 

def average_spectrum(f):
    a =  np.sum(f)
    b =  a/(ndata)**3
    return (f/b)

s = pynbody.load("snapshot_104.dms")
f = pynbody.sph.to_3d_grid(s, 'rho', 256, 256, 256)
h = f.in_units("Msol Mpc**-3") 
h1 = average_spectrum(h)
p = plt_PSD_avg(h1, 1)

