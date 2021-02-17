# -*- coding: utf-8 -*-
"""
Created on Sun May 24 08:18:56 2020

@author: Enrique

Simulations of multifreq AFM over a generalized Maxwell model

TEsting two materials and phase spectroscopy, with viscoelastic material properties as in mat_prop_0526.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import time


lib_path = r'C:\Users\Enrique\Documents\Github-repositories\TappingMode_Simulations\Epoxy'
os.chdir(lib_path)
from EAL_AFM_lib import dynamic_spectroscopy_mdr


##############Viscoelastic material parameters#######################
f = 70.0e3
Ge1 = 1.0e8
G1 = np.array([1.5e8,1.1e9,1.0e9])
Gg1 = Ge1 + sum(G1[:])
tau1 = np.array([0.02/f,0.03/f,0.1/f])

omega = np.logspace(4,7,num=100)

Ge2 = 1.0e7  #Equilibrium modulus in Pa
G_arm =( Gg1-Ge2)/2.0
G2 = np.array([G_arm,G_arm])
tau_arm = tau1[-1]
tau2 = np.array([tau_arm,tau_arm])
Gg2=Ge2+sum(G2[:])
tau1 *=0.01  #shifting tau1
##############Viscoelastic material parameters#######################



#################Getting Amp-Phase dynamic spectrocopy curves for material 1 and material 2at 70 kHz#################################
t0 = time.time()
fo1 = 70.0e3
k_m1 = 0.2
A1, A2, A3 = 50.0e-9, 0.0, 0.0
Q1, Q2, Q3 = 150.0, 300.0, 450.0
Hc = 7.1e-20  #Hamaker constant for polistyrene accoding to Garcia, SanPaulo two regimes AFM
R = 5.0e-9
Hc = 7.1e-20  #Hamaker constant for polistyrene accoding to Garcia, SanPaulo two regimes AFM
period1 = 1.0/fo1
dt = period1/1.0e4
startprint = 5.0*Q1*period1
simultime = startprint + 30.0*period1
printstep = period1/1.0e3
amp1,phi1,zeq1,_,_,_ = dynamic_spectroscopy_mdr(G1, Ge1, tau1, R, fo1, k_m1, A1, A2, A3, H=Hc, z_step=0.01*A1)
amp2,phi2,zeq2,_,_,_ = dynamic_spectroscopy_mdr(G2, Ge2, tau2, R, fo1, k_m1, A1, A2, A3, H=Hc, z_step=0.01*A1)
t1 = time.time()
print('Total time to simulate dynamic spectroscopy at 70 kHz: %2.3f'%(t1-t0))


fig,ax = plt.subplots()
ax.plot(zeq1*1.0e9,amp1*1.0e9,'*',label='Amp material 1 at 70 kHz')
ax.plot(zeq2*1.0e9,amp2*1.0e9,label='Amp material 2 at 70 kHz')
ax.legend(loc='best')
ax.set_xlabel('Zeq, nm')
ax.set_ylabel('Amp, nm')
fig.savefig('Amp curve at 70 kHz.png', bbox_inches='tight')

fig,ax = plt.subplots()
ax.plot(zeq1*1.0e9,phi1,'*',label='Phase material 1 at 70 kHz')
ax.plot(zeq2*1.0e9,phi2,label='Phase material 2 at 70 kHz')
ax.set_xlabel('Zeq, nm')
ax.set_ylabel('Phase, deg')
ax.legend(loc='best')
fig.savefig('Phase curve at 70 kHz.png', bbox_inches='tight')

#################Getting Amp-Phase dynamic spectrocopy curves for material 1 and material 2at 70 kHz#################################




