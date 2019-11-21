#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 01:47:28 2019

@author: robertwi
"""
import numpy as np
import pandas as pd

class expdata():
    
    def __init__(self):
        self.exprho=self.expP=self.expu=self.exprho0=self.expP0=self.dexpP=self.dexprho=[]
        
    def getdata(self,string):
        if('Fayalite'==string or 'fayalite' == string):
            # Chen, Ahrens 2002
            self.exprho0=np.array(4380)
            self.exprho=np.array([4900,5610,6760,7000,7260,7210,7340,7590,7650,7730])
            self.dexprho=np.array([30,50,50,60,60,100,30,70,60,80])
            self.expP0=np.array(0)
            self.expP=np.array([23.3,46.2,103.3,124.7,141.4,153,169.4,190.5,200.3,211.9])*10**9
            self.dexpP=np.array([0.4,0.9,0.8,0.8,1.7,2.1,0.7,1.5,2.1,1.6])*10**9
            self.expu=-1./self.exprho*((self.expP0-self.expP)/2+self.expP)-1/self.exprho0*((self.expP0-self.expP)/2+self.expP0)
        elif ('Fosterite'==string  or 'fosterite'==string):
            ## Mosenfelder 2007
            self.exprho0=3222
            self.exprho=np.array([3980,4550,4950,5100,5410,5480])
            self.dexprho=np.array([30,40,40,50,130,600])
            self.expP0=0
            self.expP=np.array([38.9,70.6,97.4,130.4,141.7,188.5])*10**9
            self.dexpP=np.array([0.6,0.6,0.8,1.3,3.3,2.0])*10**9
            data = pd.read_csv('Root-2018-Forsterite-QMD-Hugoniot.csv', header = 1,delimiter=';',dtype=np.float64).values
            data2 = pd.read_csv('Mosenfelder-2007-forsterite-hugoniot.csv', header = 0,delimiter=';',dtype=np.float64).values
            self.expP=np.concatenate((data[1:,2]*10**9,data2[:,1]*10**9,self.expP))
            self.exprho=np.concatenate((data[1:,0]*10**3,data2[:,0]*10**3,self.exprho))
            self.expeta=self.exprho/self.exprho0
            self.expu=-1/self.exprho*((self.expP0-self.expP)/2+self.expP)-1/self.exprho0*((self.expP0-self.expP)/2+self.expP0)
        elif ('Iron'==string  or 'iron'==string):
            # Brown 2000
            self.exprho0=7850
            self.exprho=np.array([9550,9560,9610,9720,10190,10220,10480,10550,10860,10900,11250,11600,11630,11710,11730,11730,11690,11900,12010,12070,12110,12170,12330,12320,11580,12140,12380,12470,12580,12580,12700,12880,13050,13150,13100,13380,13660])
            self.dexprho=np.array([10,10, 20,20, 10, 20, 20, 20, 30, 20, 30, 40, 30, 20, 20, 30, 30, 50, 40, 80, 110, 70, 30, 30, 80, 40, 50, 50, 30, 30, 60, 40, 60, 30, 100, 40, 110])
            self.expP0=0
            self.expP=np.array([39.6,40.1,42.7,46.3,69.6,70.5,84.6,89.1,107.8,110.2,135.1,165.2,168.7,175.4,175.7,177.5,179.4,191.8,209.5,218.2,226.2,225.5,244.2,245.7,161.5,222,262.8,272.2,273.6,283.2,300.1,314.3,328.8,354.6,361.8,400.7,442.1])
            self.dexpP=np.array([2, 2, 4, 5, 2, 4, 4, 4, 7, 6, 9, 13, 9, 6, 6, 8, 8, 14, 13, 25, 39, 25, 12, 11, 22, 14, 18, 20, 9, 12, 22, 16, 22, 13, 40, 16, 46])*10**(-1)
            # Al'tshuler 1958,1962, 1981
            self.exprho=np.append(self.exprho,[17317, 12740, 13130, 13190, 13380, 13690, 13790, 15284])
            self.dexprho=np.append(self.dexprho, [566, 112, 125, 127, 133, 144, 148, 535])
            self.expP=np.append(self.expP, [1351, 300, 344, 362, 400, 429, 487, 870])
            self.dexpP=np.append(self.dexpP, [97, 6.2, 7.2, 7.5, 8.3, 8.9, 10.1, 32])
            ## Krupnikov 1963
            self.exprho=np.append(self.exprho, [15622, 15582])
            self.dexprho=np.append(self.dexprho, [547, 545])
            self.expP=np.append(self.expP, [938, 894])
            self.dexpP=np.append(self.dexpP, [33, 32])
            self.expP=self.expP*10**9
            self.dexpP=self.dexpP*10**9
            self.expu=-1/self.exprho*((self.expP0-self.expP)/2+self.expP)-1/self.exprho0*((self.expP0-self.expP)/2+self.expP0)
            self.dexpu=np.sqrt(self.dexprho**2*(0.5*(1/self.exprho0-1/self.exprho))**2+self.dexpP**2*(self.expP/(2*self.exprho**2))**2)
        elif ('Aluminum'==string  or 'aluminum'==string):
            # Knudson 2003
            self.us=np.array([11.08, 11.36, 13.77, 14.01, 14.64, 14.67, 14.91, 15.03, 15.11, 15.25, 15.23, 16.08, 17.83, 17.82, 17.89])*10**3
            self.dus=np.array([28, 28, 45, 22, 23, 47, 24, 24, 24, 50, 50, 27, 59, 20, 20])*10
            self.up=np.array([4.13, 4.37, 6.38, 6.53, 7.09, 7.05, 7.21, 7.21, 7.42, 7.44, 7.50, 8.08, 9.59, 9.66, 9.81])*10**3
            self.dup=np.array([5, 5, 7, 7, 9, 9, 9, 9, 9, 10, 10, 10, 15, 16, 18])*10
            self.expP0=0
            self.exprho0=2700
            self.expP=self.exprho0*self.us*self.up
            self.drus=self.exprho0*self.up
            self.drup=self.exprho0*self.us
            self.dexpP=np.sqrt(self.drus**2*self.dus**2+self.drup**2.*self.dup**2)
            self.exprho=self.us/(self.us-self.up)*self.exprho0
            self.drus=(1/(self.us-self.up)-self.us/(self.us-self.up)**2)*self.exprho0
            self.drup=(self.us/(self.us-self.up)**2)*self.exprho0
            self.dexprho=np.sqrt(self.drus**2*self.dus**2+self.drup**2*self.dup**2)
#            %% Nellis 2003
            self.expP2=np.array([64.94, 102, 162.9, 62.22, 68.16, 162, 194.1, 208.8, 101.6, 150.6])*10**9
            self.expP=np.append(self.expP,self.expP2)
            self.dexpP2=np.array([0.11, 0.2, 0.7, 0.11, 0.20, 0.7, 0.9, 1, 0.2, 0.7])*10**9
            self.dexpP=np.append(self.dexpP,self.dexpP2)
            self.exprho2=self.exprho0*np.array([1/0.7020, 1/0.6444, 1/0.5869, 1/0.7069, 1/0.6952, 1/0.5869, 1/0.5658, 1/0.5575, 1/0.6451, 1/0.5957])
            self.dexprho2=self.exprho0*np.array([1/(0.7020-0.0007), 1/(0.6444-0.0006), 1/(0.5869-0.0009), 1/(0.7069-0.0007), 1/(0.6952-0.0008), 1/(0.5869-0.0009), 1/(0.5658-0.0009), 1/(0.5575-0.0009), 1/(0.6451-0.0006), 1/(0.5957-0.0010)])-self.exprho2
            self.exprho=np.append(self.exprho,self.exprho2)
            self.dexprho=np.append(self.dexprho,self.dexprho2)
            self.expu=-1./self.exprho*((self.expP0-self.expP)/2+self.expP)-1./self.exprho0*((self.expP0-self.expP)/2+self.expP0)
        else:
            print("found no material stored as ",string)