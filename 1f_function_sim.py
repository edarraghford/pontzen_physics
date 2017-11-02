import numpy as np 
import sys
import math
import matplotlib.pyplot as plt
from scipy import randn, fft, ifft, real
from numpy import arange, pi, cos, sin, zeros, double
from mpl_toolkits.mplot3d import Axes3D 

def power_spectrum(x):
	return x**2

def flicker_generator(n):
	re=zeros(n)
	im=zeros(n)
	for i in range(1,(n/2),1):
		mag = np.sqrt(power_spectrum(double(i)))  
		pha = 2.0*pi*randn(1)
		re[i] = mag*cos(pha)*randn(1)
		im[i] = mag*sin(pha)*randn(1)
		re[n-i] = re[i]
		im[n-i] = -im[i]
	im[n/2] = 0.0
	res = ifft(re+complex(0,1)*im,norm="ortho")
	return real(res) 
repeat = 100
ndata = 100
t = np.arange(ndata)
fn = flicker_generator(ndata)

def plot_data(fn): 
 	plt.style.use('ggplot')
	plt.figure(figsize=(12,14))

	ax = plt.subplot(211)

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)

	plt.xlabel('Time')
	plt.ylabel('Signal') 
	for i in range (repeat):
		plt.plot(t,fn[i],'-',color='black')
	plt.show()
x = np.zeros((repeat,ndata))
 

for i in range (repeat):
	x[i] += flicker_generator(ndata) 
	


 
plot_data(x) 

def plt_PSD(fn):
        plt.style.use('ggplot')
        plt.figure(figsize=(12,14))

        ax = plt.subplot(211)

	 
	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
	x = np.linspace(1,ndata/2,ndata/2-1) 
	pow = np.zeros(ndata)
	for i in range (repeat): 
		pows = fft(fn[i],norm="ortho") 
		pow += np.absolute(pows)**2  
 	pow = pow/repeat
	pow3 = pow[1:ndata/2]   

 
	plt.plot(x, pow3)
	plt.plot(x,power_spectrum(x)) 
	plt.yscale('log') 
	plt.xscale('log') 

	plt.show()

plt_PSD(x) 

