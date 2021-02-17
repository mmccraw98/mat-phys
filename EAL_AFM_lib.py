# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 09:21:54 2017

@author: Enrique Alejandro

Description: this library contains the core algortihms for tapping mode AFM simulations.

Updated May 25th 2020
"""

import numpy as np
from numba import jit
import os
import time

path = r'C:\Users\Enrique\Documents\Github-repositories\TappingMode_Simulations\Epoxy'
path.replace('\\','//')

os.chdir(path)
from AFM_calculations import E_diss, V_ts, Amp_Phase



def verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3):
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = zb + z1 + z2 + z3
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet = jit()(verlet)



def GenMaxwell_parabolic_LR(G, tau, R, dt, startprint, simultime, fo1, k_m1, A1, A2, A3, zb, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19):
    """This function is designed for tapping over a Generalized Maxwel surface"""
    """The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion"""
    """Modified Dec 2nd 2017, making sure tau passed does not contain values lower than timestep which would make numerical integration unstable"""
    G_a = []
    tau_a = []
    for i in range(len(G)): #this for loop is to make sure tau passed does not contain values lower than timestep which would make numerical integration unstable
        if tau[i] > dt*10.0:
            G_a.append(G[i])
            tau_a.append(tau[i])
    G = np.array(G_a)
    tau = np.array(tau_a)
        
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    a = 0.2e-9  #interatomic distance
    eta = tau*G
    Gg = Ge
    for i in range(len(tau)):
        Gg = Gg + G[i]
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    xc, xc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    alfa = 16.0/3.0*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_Gxc = 0.0
    sum_G_pb_pc = 0.0
    
        
    while t < simultime:
        t = t + dt
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
       
        sum_Gxc = 0.0
        sum_G_pb_pc = 0.0  
        if tip > xb: #aparent non contact
            for i in range(len(tau)):
                sum_Gxc = sum_Gxc + G[i]*xc[i]
            if sum_Gxc/Gg > tip:  #contact, the sample surface surpassed the tip in the way up
                xb = tip
                pb = (-xb)**1.5
                for i in range(len(tau)):
                    sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
                Fts = alfa*( Ge*pb + sum_G_pb_pc )
                #get postion of dashpots
                for i in range(len(tau)):
                    pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                    pc[i] = pc[i] + pc_rate[i]*dt
                    xc[i] = -(pc[i])**(2.0/3)
            
            else: #true non-contact
                xb = sum_Gxc/Gg
                Fts = 0.0
                for i in range(len(tau)):
                    xc_rate[i] = G[i]*(xb-xc[i])/eta[i]
                    xc[i] = xc[i] + xc_rate[i]*dt
                    pc[i] = (-xc[i])**(3.0/2)     #debugging
                     
        else:  #contact region
            xb = tip
            pb = (-xb)**1.5
            for i in range(len(tau)):
                sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
            Fts = alfa*( Ge*pb + sum_G_pb_pc )
            #get postion of dashpots
            for i in range(len(tau)):
                pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                pc[i] = pc[i] + pc_rate[i]*dt
                xc[i] = -(pc[i])**(2.0/3)
        
        
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if tip > xb:  #overall non-contact
            Fts = -H*R/( 6.0*( (tip-xb) + a )**2 )
        else:
            Fts = Fts - H*R/(6.0*a**2)
        
      
           
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)


def MDR_GenMaxwell_tapping(G, tau, R, dt, simultime, zb, A1, k_m1, fo1, printstep=1, Ndy = 1000, Ge = 0.0, dmax = 10.0e-9, startprint =1, Q1=100, Q2=200, Q3=300, H=2.0e-19, A2 = 0.0, A3 = 0.0):
    """This function runs a simulation for a parabolic probe in force spectroscopy"""
    """over a generalized Maxwell surface"""
    """Output: time, tip position, tip-sample force, contact radius, and sample position"""
    G_a = []
    tau_a = []
    for i in range(len(G)): #this for loop is to make sure tau passed does not contain values lower than timestep which would make numerical integration unstable
        if tau[i] > dt*10.0:
            G_a.append(G[i])
            tau_a.append(tau[i])
    G = np.array(G_a)
    tau = np.array(tau_a)
    
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2  
    f1 = fo1  #excited at resonance
    f2 = fo2  #excited at resonance
    f3 = fo3  #excited at resonance
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
   
    if printstep == 1:
        printstep = 10.0*dt
    if startprint == 1:
        startprint = 5.0*Q1*1.0/fo1
        
    eta = G*tau #bulk viscosity of the dashpots in the Generalized Maxwell model
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element
    ke = 8*Ge*dy
    k = 8*G*dy
    c = 8*eta*dy
    kg=ke
    for j in range(len(k)):
        kg+=k[j]
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n = np.zeros(len(y_n)), np.zeros(len(y_n))    #position of the base of each SLS (or higher order viscoelastic material) dependent on position
    xc_dot_n = np.zeros(( len(y_n), len(tau)))    #velocity of each dashpot
    xc_n = np.zeros(( len(y_n), len(tau)))    #position of the dashpot of each SLS (or higher order viscoelastic material) dependent on time as function of position and tau
    F_n =  np.zeros(len(y_n))  #force on each SLS (or higher order viscoelastic material) element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []  #sample position, penetration
    probe_a = []   #array that will contain tip's position
    t = 0.0
    F = 0.0
    sum_kxc = 0.0
    sum_k_xb_xc = 0.0
    printcounter = 1
    if printstep == 1:
        printstep = dt
    ar_a = [] #array with contact radius
    ar = 0.0  #contact radius
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    a = 0.2e-9 #interatomic distance
    while t < simultime:
        t = t + dt
        #probe = 5.0e-9-10.0e-9*np.sin(2.0*np.pi*fo1*t)  #
        probe, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, F, dt, fo1,fo2,fo3,f1,f2,f3)
        if probe < 0.0:  #indentation is only meaningful if probe is lower than zero position (original position of viscoelastic foundation)
            d = -1.0*probe  #forcing indentation to be positive  (this is the indentation)
        else:
            d = 0.0
        if t > (startprint + printstep*printcounter):
            F_a.append(F)
            t_a.append(t)   
            d_a.append(x_n[0])
            ar_a.append(ar)        
            probe_a.append(probe)
        
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0  #initializing contact radius to zero
        for n in range(len(y_n)):  #advancing in space
            tip_n[n] =  g_y[n] - d
            if tip_n[n] > 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
                F_n[n] = 0.0  #testing this line 05/25/2020
    
            if tip_n[n] > x_n[n]: #aparent non contact
                for i in range(len(tau)):  #iterating for different characteristic times
                    sum_kxc = sum_kxc + k[i]*xc_n[n,i]
                if sum_kxc/kg > tip_n[n]:  #contact, the sample surface surpassed the tip in the way up
                    x_n[n] = tip_n[n]
                    for i in range(len(tau)): #iterating for different characteristic times
                        sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                    F_n[n] =  - ke*x_n[n] - sum_k_xb_xc
                else:  #true non-contact
                    x_n[n] = sum_kxc/kg
                    F_n[n] = 0.0
                sum_kxc = 0.0
                sum_k_xb_xc = 0.0
            else: #contact region, tip is lower than the sample's surface
                x_n[n] = tip_n[n]
                for i in range(len(tau)): #iterating for different characteristic times
                    sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                F_n[n] = - ke*x_n[n] - sum_k_xb_xc
                sum_k_xb_xc = 0.0
                if F_n[n] < 0.0:  #negative force is unphysical, testing 05/25/2020
                    F_n[n] = 0.0  #testing this line 05/25/2020
            #getting position of dashpots
            for i in range(len(tau)):
                xc_dot_n[n,i] = k[i]*(x_n[n]-xc_n[n,i])/c[i]
                xc_n[n,i] = xc_n[n,i] + xc_dot_n[n,i]*dt
    
            if F_n[n] > 0.0: #only positive forces are added
                F = F + F_n[n] #getting total tip-sample force
                ar = y_n[n]   #getting actual contact radius  
        
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if probe > x_n[0]:  #overall non-contact
            F = -H*R/( 6.0*( (probe-x_n[0]) + a )**2 )
        else: #overall contact
            F = F - H*R/(6.0*a**2) 
        
    
    return np.array(t_a), np.array(probe_a), np.array(F_a), np.array(ar_a), np.array(d_a)


def Hertz_tapping(G, R, dt, startprint, simultime, fo1, k_m1, A1, A2, A3, zb, printstep = 1, Q1=100, Q2=200, Q3=300, H=2.0e-19, nu=0.5):
    """This function is designed for a parabolic tip tapping over an elastic semi-infinite solid"""
    """This simulation assumes no vdW interactions (they are screened due to liquid sorrounding)"""
    """It is assumed that tip modulus is much larger than sample modulus therefore E* is equal to E of sample"""
    """Created Nov 15th 2017, modified May 24th 2020"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
        
    t_a = []
    Fts_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    tip = 0.0
    if nu == 0.5:
        alfa = 16.0/3.0*np.sqrt(R)
    else:
        alfa = 8.0/(3.0*(1-0-nu))*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    
    a = 0.2e-9  #interatomic distance
        
        
    while t < simultime:
        t = t + dt
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            tip_a.append(tip)
            printcounter = printcounter + 1    
                
          
        if tip > a:  #attractive portion
            Fts = -H*R/( 6.0*(tip)**2 )
        else:
            Fts = alfa*G*(a-tip)**(1.5) - H*R/(6.0*a**2)        
        
                                  
    return np.array(t_a), np.array(tip_a), np.array(Fts_a)



GenMaxwell_jit = jit()(GenMaxwell_parabolic_LR)  #this line should stay outside function to allow the numba compilation and simulation acceleration work properly

def dynamic_spectroscopy(G, tau, R, dt, startprint, simultime, fo1, k_m1, A1, A2, A3, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19, z_step = 1):
    """This function is designed for tapping mode spectroscopy to obtain amplitude and phase curves as the cantilever is approached towards the surface.
    
    The contact mechanics are performed over the framework of Lee and Radok for viscoelastic indentation (Lee, E. Ho, and Jens Rainer Maria Radok. "The contact problem for viscoelastic bodies." Journal of Applied Mechanics 27.3 (1960): 438-444.) 
    
    Parameters:
    ---------- 
    G :  numpy.ndarray
        moduli of the springs in the Maxwell arms of a generalized Maxwell model (also called Wiechert model)
    tau: numpy.ndarray
        relaxation times of the Maxwell arms
    R : float
        tip radius
    dt : float
        simulation timestep
    fo1 : float
        1st eigenmode resonance frequency    
    k_m1 : float
        1st eigenmode's stiffness    
    A1 : float
        target oscillating amplitude of 1st cantilever eigenmode
    A2 : float
        target oscillating amplitude of 2nd cantilever eigenmode
    A3 : float
        target oscillating amplitude of 3rd cantilever eigenmode
    printstep : float, optional
        how often the data will be stored, default is timestep
    Ge : float, optional
        rubbery modulus, the default value is zero
    Q1 : float, optional
        first eigenmode's quality factor
    Q2 : float, optional
        second eigenmode's quality factor
    Q3 : float, optional
        third eigenmode's quality factor
    H : float, optional
        Hammaker constant
    z_step : float, optional
        cantilever equilibrium spatial step between runs. The smaller this number, the more runs but slower the simulation
    
    Returns:
    -------  
    np.array(amp) : numpy.ndarray
        array containing the reduced amplitudes at different cantilever equilibrium positions
    np.array(phase) : numpy.ndarray
        array containing the phase shifts obtained at different cantilever equilibrium positions    
    np.array(zeq) : numpy.ndarray
        array containing the approaching cantilever equilibrium positions
    np.array(Ediss) : numpy.ndarray
        array containing the values of dissipated energy
    p.array(Virial) : np.ndarray
        array containing the values of the virial of the interaction
    np.array(peakF) : np.ndarray
        array containing valued of peak force
    np.array(maxdepth) : numpy.ndarray
        array containing the values of maximum indentation
    np.array(t_a) : numpy.ndarray
        time trace
    np.array(tip_a) : numpy.ndarray
        2D array containing the tip trajectory for each run
    np.array(Fts_a) : numpy.ndarray
        2D array containing the tip-sample interacting force for each run
    np.array(xb_a) : numpy.ndarray
        2D array array containing the instant position of the viscoelastic surface for each run   
    """
    if z_step == 1:
        z_step = A1*0.05 #default value is 5% of the free oscillation amplitude
    zeq = []
    peakF = []
    maxdepth = []
    amp = []
    phase = []
    Ediss = []
    Virial = []
    
    tip_a = []
    Fts_a = []
    xb_a = []
    zb = A1*1.1  #initial cantilever static position
    A = A1
                                     
    while A > A1*0.05:              
        t, tip, Fts, xb = GenMaxwell_jit(G, tau, R, dt, startprint, simultime, fo1, k_m1, A1, A2, A3, zb, printstep, Ge, Q1, Q2, Q3, H)
        A,phi = Amp_Phase(t, tip, fo1)
        Ets = E_diss(tip, Fts, dt, fo1)
        fts_peak = Fts[np.argmax(Fts)]
        tip_depth = xb[np.argmax(tip)] -xb[np.argmin(tip)]
        Vts = V_ts(tip-zb, Fts, dt)
        
        #Attaching single values to lists
        zeq.append(zb)
        peakF.append(fts_peak)
        maxdepth.append(tip_depth)
        amp.append(A)
        phase.append(phi)
        Ediss.append(Ets)
        Virial.append(Vts)
        
        #attaching 1D arrays to lists
        tip_a.append(tip)
        Fts_a.append(Fts)
        xb_a.append(xb)
        
        zb -= z_step
        
            
    return np.array(amp), np.array(phase), np.array(zeq), np.array(Ediss), np.array(Virial), np.array(peakF), np.array(maxdepth), t, np.array(tip_a), np.array(Fts_a)#, np.array(xb_a)

mdr_jit = jit()(MDR_GenMaxwell_tapping)

def dynamic_spectroscopy_mdr(G, Ge, tau, R, fo1, k_m1, A1, A2, A3, dmax=10.0e-9, Ndy = 1000, Q1=100, Q2=200, Q3=300, H=2.0e-20, z_step = 1):
    
    
    period1 = 1.0/fo1
    startprint = 5.0*Q1*period1
    printstep = period1/1.0e3    
    simultime = startprint + 30.0*period1  
    dt=period1/1.0e4
    
    if z_step == 1:
        z_step = A1*0.05 #default value is 5% of the free oscillation amplitude
    zeq = []
    peakF = []
    maxdepth = []
    amp = []
    phase = []
    Ediss = []
    Virial = []
    
    tip_a = []
    Fts_a = []
    zb = A1*1.1  #initial cantilever static position
    A = A1
    
    
    while A > A1*0.05:         
        t0 = time.time()
        t, tip, Fts,_,_ = mdr_jit(G, tau, R, dt, simultime, zb, A1, k_m1, fo1, printstep, Ndy, Ge, dmax, startprint, Q1, Q2, Q3, H, A2, A3)
        t1 = time.time()
        print('Time for the loop with Zc at %2.3f nm is %2.3f'%(   (zb*1.0e9),(t1-t0)  )   )
        A,phi = Amp_Phase(t, tip, fo1)
        Ets = E_diss(tip, Fts, dt, fo1)
        fts_peak = Fts[np.argmax(Fts)]
        #tip_depth = xb[np.argmax(tip)] -xb[np.argmin(tip)]
        Vts = V_ts(tip-zb, Fts, dt)
        
        #Attaching single values to lists
        zeq.append(zb)
        peakF.append(fts_peak)
        maxdepth.append(min(tip))
        amp.append(A)
        phase.append(phi)
        Ediss.append(Ets)
        Virial.append(Vts)
        
        #attaching 1D arrays to lists
        tip_a.append(tip)
        Fts_a.append(Fts)
        #xb_a.append(xb)
        
        zb -= z_step
            
    return np.array(amp), np.array(phase), np.array(zeq), np.array(Ediss), np.array(Virial), np.array(peakF)