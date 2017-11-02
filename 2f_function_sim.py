import numpy as np 
import sys
import math
from matplotlib import cm
import pylab as pl 
import matplotlib.pyplot as plt
from scipy import randn, fft, ifft, real
from scipy.fftpack import fft2, ifft2 
from numpy import arange, pi, cos, sin, zeros, double
from mpl_toolkits.mplot3d import Axes3D 

def power_spectrum(x):
	return x**(0.5)

def flicker_generator(n):
	re=np.zeros((n,n))
	im=np.zeros((n,n))
	for i in range(1,(n/2),1):
		for j in range(1,(n/2),1): 
				mag = np.sqrt(power_spectrum(double(np.sqrt(i**2+j**2))))  
				pha = 2.0*pi*randn(1)
				re[i,j] = mag*cos(pha)*randn(1)
				im[i,j] = mag*sin(pha)*randn(1)
				re[n-i,n-j] = re[i,j]
				im[n-i,n-j] = -im[i,j]
	im[n/2,n/2] = 0.0
	res = np.fft.ifft2(re+complex(0,1)*im,norm="ortho")
	return real(res) 
repeat = 1
ndata = 100
t = np.arange(ndata)
fn = flicker_generator(ndata)

def plot_data(fn,d): 
 	plt.style.use('ggplot')
	fig = plt.figure(figsize=(10,10))

	ax = fig.gca( projection='3d')

	X,Y = np.meshgrid(t,t)
	if (d == 1): 
		for i in range(repeat):  
			ax.plot_surface(X,Y, fn[i], cmap=cm.coolwarm) 
	else:
		ax.plot_surface(X,Y, fn, cmap=cm.coolwarm)
	plt.show()


x = np.zeros((repeat,ndata,ndata))

for i in range(repeat):
	x[i] += flicker_generator(ndata)  

# d = 0 for single plot, d = 1 for multiple plots 
d = 0 
plot_data(fn,d) 

def plt_PSD(fn):
        fig, ax = plt.subplots(2,2)
        x = np.linspace(1,ndata/2,ndata/2-1)
        pow = np.zeros((ndata,ndata))
        for i in range(repeat):
                pows = np.fft.fft2(fn[i],norm="ortho")
                pow += np.absolute(pows)**2
        pow = pow/repeat
        pow3 = pow[1:ndata/2, 1:ndata/2]
        X,Y = np.meshgrid(x,x)
        R = np.sqrt(X**2+Y**2)
        fx = np.zeros((ndata/2-1,ndata/2-1))

        for  i in range (0,ndata/2-1):
                for j in range(0,ndata/2-1):
                        if i == 0 and j ==0:
                                fx[i,j] = 0
                        else:
                                fx[i,j] = power_spectrum(R[i,j])
        ax[0,0].imshow(pow3,cmap = cm.coolwarm, extent = [0,50,50,0])

        ax[1,0].imshow(fx, cmap = cm.coolwarm, extent = [0,50,50,0])
	plt.show() 

def plt_PSD_avg(fn):
	x = np.linspace(1,ndata/2,ndata/2-1)
	pow = np.zeros((ndata,ndata))  
	pows = np.fft.fft2(fn,norm="ortho") 
	pow = np.absolute(pows)**2
	pow = pow/repeat 
	pow3 = pow[1:ndata/2, 1:ndata/2]
        X,Y = np.meshgrid(x,x)
	R = np.sqrt(X**2+Y**2) 
	f = np.zeros(ndata/2) 
	for k in range(ndata/2):
		n = 0 
		for l in range(ndata/2):
			for m in range(ndata/2):
				if (np.sqrt(l**2+m**2) <= k) and (np.sqrt((l+1)**2+(m+1)**2) > k):
					f[k] += pow[l,m]
					n = n+1
		f[k] = f[k]/n 
 	fs = f[1:]
	plt.plot(x,fs) 	    
	plt.plot(x,power_spectrum(x))
	plt.show()
plt_PSD_avg(fn) 

