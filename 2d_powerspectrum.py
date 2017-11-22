import numpy as np 
import scipy
import sys
import math
import radial_data as rad 
from matplotlib import cm
import pylab as pl 
import matplotlib.pyplot as plt
from scipy import randn, fft, ifft, real
from scipy.fftpack import fft2, ifft2 
from numpy import arange, pi, cos, sin, zeros, double
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.patches as mpatches 

repeat = 1 
ndata = 1000 
duplicate = 0 

# Input power spectrum 
def power_spectrum(x):
	return 1/x**(0.5)

# Generates function from input power spectrum 
def function_generator(n):
	re=np.zeros((n,n))
	im=np.zeros((n,n))
	x = np.linspace(1,(n/2)-1,n/2-1)
	y = np.linspace(1,(n/2)-1,n/2-1)
	X,Y = np.meshgrid(x,y)
	vspectrum = np.vectorize(power_spectrum)
	T = np.sqrt(X**2+Y**2)
	mag = np.sqrt(vspectrum(T)) 
	pha = 2.0*pi*np.random.randn((n/2-1),(n/2-1)) 
	re[1:n/2,1:n/2] = mag*cos(pha)*np.random.randn((n/2-1),(n/2-1))
	im[1:n/2,1:n/2]  = mag*sin(pha)*np.random.randn((n/2-1),(n/2-1))
	re[n/2+1:,n/2+1:] = np.fliplr(np.flipud(re[1:n/2,1:n/2]))        
	im[n/2+1:,n/2+1:] = np.negative(np.fliplr(np.flipud(im[1:n/2,1:n/2])))
	im[n/2,n/2] = 0.0
	res = np.fft.ifft2(re+complex(0,1)*im,norm="ortho")
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
def plt_PSD_avg(fn):
        x = np.linspace(1,(ndata/2)-1,ndata/2-1)
        y = np.linspace(1,(ndata/2)-1,ndata/2-1)
        xx,yy = np.meshgrid(x,y)
	pow = np.absolute(np.fft.fft2(fn,norm="ortho"))**2 
	f = np.zeros(ndata/2) 
	fs = rad.radial_data(pow[1:ndata/2,1:ndata/2], x=xx,y=yy) 
	plt.plot(x,fs.mean[1:ndata/2]) 	    
	plt.plot(x,power_spectrum(x))
	plt.title("Input Power Spectrum and Output Power Spectrum") 
	blue_patch = mpatches.Patch(color='blue', label = 'Output Power Spectrum') 
	orange_patch = mpatches.Patch(color='orange',label = 'Input Power Spectrum') 
	plt.legend(handles=[blue_patch,orange_patch]) 
	plt.show()

fn = function_generator(ndata) 
plot_data(fn,duplicate)
plt_PSD_avg(fn) 

