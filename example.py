#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 03:06:08 2019

@author: robertwi
"""
from tillotson import tillotson
from varpoly import varpoly
from expdata import expdata
import numpy as np
from matplotlib import pyplot as plt
from sympy import *
import csv
import pandas as pd
from matplotlib import rc
col = plt.cm.Set1(np.linspace(0,1,10))

## input values
rho=np.linspace(1,10000,3000)
T=0

## material creation
fors=varpoly();
## (rho0,B0,dBdP0,AA,ZZ,tmin,delta)
fors.creatematerial(3300,95e9,4.2,36,18,0.3,0.001) #fitted to hugoniot
## (a,b=5./3.,c=0.6)
fors.createexpstate(4.445,c=0.89) # Evap 12.5*10**6 fits to Tboil,Svap,Sliq=7408

fors.calculate_P(rho,T)
fors.calculate_B(rho,T)
fors.calculate_dBdP(rho,T)
fors.calculate_grun_cold(rho)
plt.figure(1)
plt.semilogx(fors.eta,fors.P/10**9,color=col[1],linewidth=2.0)
plt.ylabel('Pressure (GPa)')
plt.xlabel('$\\rho / \\rho_0$')
plt.figure(2)
plt.semilogx(fors.eta,fors.B/10**9,color=col[1],linewidth=2.0)
plt.ylabel('Bulk Modulus (GPa)')
plt.xlabel('$\\rho / \\rho_0$')
plt.figure(3)
plt.semilogx(fors.eta,fors.grun,color=col[1])
plt.ylabel('Gruneisen parameter')
plt.xlabel('$\\rho / \\rho_0$')
plt.figure(4)
plt.semilogx(fors.eta,fors.grun_q,color=col[1])
plt.ylabel('q')
plt.xlabel('$\\rho / \\rho_0$')

fors.gethugoniot(rho)
plt.figure(5)
plt.plot(fors.up/1000,fors.us/1000,color=col[1],linewidth=2.0)
plt.ylabel('Shock velocity (km/s)')
plt.xlabel('Particle velocity (km/s)')
plt.figure(6)
plt.plot(fors.eta,fors.Ph/10**9,color=col[1],linewidth=2.0)
plt.ylabel('Pressure (GPa)')
plt.xlabel('$\\rho / \\rho_0$')