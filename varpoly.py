# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:09:58 2017

@author: Khyzath
"""


from eos import eos
import math
import scipy.integrate as integrate
#from mpmath import *
import mpmath as mp
import numpy as np
import scipy.optimize as optimize
import numba
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize
import gc

class varpoly(eos):
    
    def __init__(self):
        self.P=self.B=self.dBdP=self.grun=self.vp=self.t=[]
        self.rho=self.P_c=self.B_c=self.dBdP_c=self.rho_critical=self.U_c=[]
        self.T=0
        self.B_iso=self.dBdP_iso=self.U=self.grun_int=[]
        self.Pvapstate=[]
        self.rhovapstate=[]
        self.Svapstate=[]
        self.Uvapstate=[]
        self.Tvapstate=[]
        self.S=self.F=self.H=[]
    
    def creatematerial(self,rho0,B0,dBdP0,AA,ZZ,tmin,delta):
        self.rho0=float(rho0)
        self.B0=float(B0)
        self.dBdP0=float(dBdP0)
        self.AA=float(AA)
        self.ZZ=float(ZZ)
        self.tmin=float(tmin)
        self.delta=float(delta)
        self.findA()
        self.etacrit=float(self.rho_critical/self.rho0)
        x2=self.A0/self.A1
        n=(self.A1+self.A2)/self.A1
        b=1./self.A1;
        self.G1_const = x2**(n-1)*mp.gammainc(1-n,a=x2)
        self.G2_const = (1./b*(x2**b*mp.gammainc(1-n,a=x2)-mp.gammainc(1-n+b,a=x2)))
        self.rho=np.array([float(self.rho0)])
        self.grun0,self.grun_q0,self.grun_lamb0 = self.calculate_grun_cold(self.rho0)
     
    def createexpstate(self,a,b=5./3.,c=0.6):
        self.Pexp_const_a=a
        self.Pexp_const_b=b
        self.Bexp0=self.B0/(a-b)
        self.grun_const_c=c
        self.grun_ideal=1./3.
        yk=1+self.grun0-self.grun_ideal
        self.grun_const_d=yk*c+self.grun_q0*self.grun0
        
    def getmaterial(self):
        pass
## Calculates the cold pressure               
    def calculate_P_cold(self,rho):    
        self.checkmat(rho)

        if type(self.P_c) is not np.ndarray:
            self.P_c = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] >= self.rho0):
                    x2=self.A0/self.A1
                    d=self.B0*math.exp(x2)/self.A1
                    x1=x2*self.eta[i]**(-self.A1)
                    n=(self.A1+self.A2)/self.A1
                    g1=x1**(n-1)*mp.gammainc(1-n,a=x1)
                    P1=0
                    P2=0
                    eta=self.eta[i]
                    if(self.rho[i]>self.rho_critical):
                        eta = self.etacrit;
                        P2 = self.C * self.rho[i]**(5./3.) - self.D * self.rho[i]**(4./3.) - self.E * self.rho[i]
                        P1 = self.C * self.rho_critical**(5./3.) - self.D * self.rho_critical**(4./3.) - self.E * self.rho_critical
                    Ppoly=d*(eta**self.A2*g1-self.G1_const)
                    self.P_c[i]=(Ppoly-P1)+P2
                else:
                    self.P_c[i]=self.Bexp0*(self.eta[i]**self.Pexp_const_a-self.eta[i]**self.Pexp_const_b)

# Finds the transition density for when to switch to TFD scheme (Vhigh pressure)
    def findcrit(self,rho, n_dyne):
        n_num = (5./3.)**2*self.C*rho**(5./3.)-(4./3.)**2*self.D*rho**(4./3.)-self.E*rho
        n_denom = (5./3.)*self.C*rho**(5./3.)-(4./3.)*self.D*rho**(4./3.)-self.E*rho
        n_guess = n_num/n_denom
        # subtracting off terms that add to n_dyne
        self.A2 = n_guess-(n_dyne-n_guess)*(self.rho0/rho)**(n_dyne/(n_dyne-n_guess))    
        self.A1 = n_dyne/(n_dyne-self.A2)
        self.A0 = n_dyne - self.A2
        B_1 = n_denom
        B_2 = np.log(self.B0) + (self.A0/self.A1)*(1-(self.rho0/rho)**self.A1)+self.A2*np.log(rho/self.rho0)
        B_2 = np.exp(B_2)
        return B_2-B_1
    
    def findA(self):
        AoZ = self.AA/self.ZZ;
        self.C = 5.16e12/((2690*AoZ)**(5./3.));
        self.D = 5.16e12/((2690*AoZ)**(4./3.))*(0.20732+0.40726*self.ZZ**(2./3.));
        self.E = 5.16e12/((2690*AoZ)**(3./3.))*0.01407;
        self.rho_critical = optimize.newton(self.findcrit, self.rho0*200000000, args=(self.dBdP0,))

# checks if new rho or T is called
    def checkmat(self,rho=None,T=None):

        if T is None:
            T = self.T
        if rho is None:
            rho = self.rho    
        if type(rho) is not np.ndarray:
            rho=np.array([float(rho)])
        if type(T) is not np.ndarray:
            T=np.array([float(T)])
        if np.array_equal(self.rho,rho) and np.array_equal(self.T,T):
            pass            
        else:
            if np.array_equal(self.rho,rho):
                self.P=self.B=self.B_iso=self.dBdP=self.dBdP_iso=[]
                self.S=self.F=self.H=self.U=self.grun=[]
            else:
                self.P=self.B=self.dBdP=self.grun=self.vp=self.t=[]
                self.P_c=self.B_c=self.dBdP_c=self.U_c=[]
                self.B_iso=self.dBdP_iso=self.U=self.grun_int=[]
                self.S=self.F=self.H=[]
                
            self.rho=rho
            self.T=T
            self.eta=self.rho/self.rho0
            
## COLD BULK MODULUS  
    def calculate_B_cold(self,rho):
        self.checkmat(rho)    
        self.calculate_P_cold(self.rho)
        if type(self.B_c) is not np.ndarray:
            self.B_c = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] >= self.rho0):
                    Bc = np.log(self.B0) + (self.A0/self.A1)*(1-(self.rho0/self.rho[i])**self.A1)+self.A2*np.log(self.eta[i])
                    BTFD = (5. / 3.) * self.C * self.rho[i]**(5. / 3.) - (4. / 3.) * self.D*self.rho[i]**(4. / 3.) - self.E * self.rho[i]
                    B = np.zeros_like(Bc)
                    if(self.rho[i]>self.rho_critical):
                        self.B_c[i]= BTFD
                    else:
                        self.B_c[i]= np.exp(Bc)
                else:
                    self.B_c[i]=self.Bexp0*(self.Pexp_const_a*self.eta[i]**self.Pexp_const_a-self.Pexp_const_b*self.eta[i]**self.Pexp_const_b)
            

## DERIVATIVES OF BULK MODULUS COLD    
    def calculate_dBdP_cold(self,rho):

        self.checkmat(rho)
        self.calculate_B_cold(self.rho)
        if type(self.dBdP_c) is not np.ndarray:
            self.dBdP_c = np.zeros_like(self.rho)
            self.d2BdP_c = np.zeros_like(self.rho)
            self.d3BdP_c = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] >= self.rho0):
                    dBdPc=self.A0*self.eta[i]**(-self.A1)+self.A2;
                    d2BdPc=-self.A1*(dBdPc-self.A2)/self.B_c[i];
                    d3BdPc=-(self.A1/self.B_c[i]**2)*(self.A2*dBdPc+self.B_c[i]*d2BdPc-dBdPc**2)
                    ##TFD TERMS
                    n_num = (5. / 3.)**2 * self.C - (4. / 3.)**2 * self.D * self.rho[i]**(-1. / 3.) - self.E * self.rho[i]**(-2. / 3.)
                    n_denom = (5. / 3.) * self.C - (4. / 3.) * self.D * self.rho[i]**(-1. / 3.) - self.E * self.rho[i]**(-2. / 3.)    
                    dBdPTFD=n_num / n_denom
                    
                    dn_num = (1. / 3.) * (4. / 3.)**2 * self.D * self.rho[i]**(-4. / 3.) + self.E * (2. / 3.) * self.rho[i]**(-5. / 3.)
                    dn_denom = (1. / 3.) * (4. / 3.) * self.D * self.rho[i]**(-4. / 3.) + self.E * (2. / 3.) * self.rho[i]**(-5. / 3.)
                    d2BdPTFD=((self.rho[i] * dBdPTFD) / self.B_c) * (dn_num / n_num - dn_denom / n_denom)
                    
                    d2n_num = -(1. / 3.) * (4. / 3.)**3 * self.D * self.rho[i]**(-7. / 3.) - self.E * (2. / 3.) * (5. / 3.) * self.rho[i]**(-8. / 3.)
                    d2n_denom = -(1. / 3.) * (4. / 3.)**2 * self.D * self.rho[i]**(-7. / 3.) - self.E * (2. / 3.) * (5. / 3.) * self.rho[i]**(-8. / 3.)
                    first = d2n_num / n_num - dn_num**2 / n_num**2 - d2n_denom / n_denom + dn_denom**2 / n_denom**2
                    second = dn_num / n_num - dn_denom / n_denom
                    derivrnb = (dBdPTFD * self.rho[i] + self.rho[i] * (self.B_c[i] * d2BdPTFD - dBdPTFD**2)) / self.B_c[i]**2
                    d3BdPTFD=((self.rho[i]**2 * dBdPTFD) / self.B_c[i]**2) * first + derivrnb * second
        
                    if(rho[i]>self.rho_critical):
                        self.dBdP_c[i] = dBdPTFD
                        self.d2BdP_c[i] = d2BdPTFD
                        self.d3BdP_c[i] = d3BdPTFD
                    else:
                        self.dBdP_c[i] = dBdPc
                        self.d2BdP_c[i] = d2BdPc
                        self.d3BdP_c[i] = d3BdPc
                else:
                    self.dBdP_c[i] = (self.Bexp0/self.B_c[i])*(self.Pexp_const_a**2*self.eta[i]**self.Pexp_const_a-self.Pexp_const_b**2*self.eta[i]**self.Pexp_const_b)
                    self.d2BdP_c[i] = (self.Bexp0/self.B_c[i]**2)*(self.Pexp_const_a**3*self.eta[i]**self.Pexp_const_a-self.Pexp_const_b**3*self.eta[i]**self.Pexp_const_b-self.dBdP_c[i])
                    self.d3BdP_c[i] = 0.0
                    
         
## calculate t-parameter and its pressure derivatives
    def calculate_tparam(self,rho):
        self.checkmat(rho)      
        self.calculate_B_cold(self.rho)
        self.calculate_dBdP_cold(self.rho)
        if self.t is not np.ndarray:
            tinf=2.5;
            td=self.tmin-tinf;
            self.t = np.zeros_like(self.rho)
            self.dtdP = np.zeros_like(self.rho)
            self.d2tdP = np.zeros_like(self.rho)
            self.d3tdP = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] >= self.rho0):
                    self.t[i]=td*(self.rho0/self.rho[i])**self.delta+tinf;
                    self.dtdP[i]=-self.delta*(self.t[i]-tinf)/self.B_c[i];
                    self.d2tdP[i]=-(self.delta/self.B_c[i]**2)*(self.dBdP_c[i]*(tinf-self.t[i])+self.B_c[i]*self.dtdP[i]);
                    self.d3tdP[i]=-(self.delta/self.B_c[i]**3)*(self.B_c[i]*(self.d2BdP_c[i]*(tinf-self.t[i])-2*self.dtdP[i]*self.dBdP_c[i])+2*self.dBdP_c[i]**2*(self.t[i]-tinf)+self.B_c[i]**2*self.d2tdP[i])
            

## calculate gruneisen and derivatives
    def calculate_grun_cold(self,rho):
        self.checkmat(rho)
        self.calculate_P_cold(self.rho)
        self.calculate_B_cold(self.rho)
        self.calculate_dBdP_cold(self.rho) 
        self.calculate_tparam(self.rho)
        
        if type(self.grun) is not np.ndarray:
            self.grun = np.zeros_like(self.rho)
            self.grun_q = np.zeros_like(self.rho)
            self.grun_lamb = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] >= self.rho0):
                    F=self.P_c[i]/self.B_c[i];
                    Ng=self.dBdP_c[i]/2-1/6;
                    Dg=1-(2*self.t[i]/3)*F;
                    Tg=(self.t[i]/3)*(1-F/3);
                    taug=(self.P_c[i]/3)*self.dtdP[i];
                    dF=1/self.B_c[i]-(self.P_c[i]*self.dBdP_c[i])/self.B_c[i]**2;
                    dN=self.d2BdP_c[i]/2;
                    dD=(-2/3)*(self.dtdP[i]*F+dF*self.t[i]);
                    dT=(self.dtdP[i]/3)*(1-F/3)-(dF*self.t[i])/9;
                    dtau=(1/3)*(self.dtdP[i]+self.P_c[i]*self.d2tdP[i]);
                    d2F=(1/self.B_c[i]**3)*(2*self.P_c[i]*self.dBdP_c[i]**2-self.B_c[i]*(self.P_c[i]*self.d2BdP_c[i]+2*self.dBdP_c[i]));
                    d2N=self.d3BdP_c[i]/2;
                    d2D=(-2/3)*(self.d2tdP[i]*F+2*self.dtdP[i]*dF+d2F*self.t[i]);
                    d2T=(self.d2tdP[i]/3)*(1-F/3)-2*(self.dtdP[i]/9)*dF-(d2F*self.t[i])/9;
                    d2tau=(1/3)*(2*self.d2tdP[i]+self.P_c[i]*self.d3tdP[i]);
                    self.grun[i]= (Ng-Tg-taug)/Dg;
                    dgamma=(1/Dg)*((dN-dtau-dT)-dD*self.grun[i]);
                    self.grun_q[i]=(-self.B_c[i]/self.grun[i])*dgamma;
                    d2gamma=(1/Dg**2)*(Dg*(d2N-d2tau-self.grun[i]*d2D-dD*dgamma-d2T)-dD*(dN-dtau-self.grun[i]*dD-dT));
                    self.grun_lamb[i]=((-self.B_c[i])*(1/self.grun[i]**2)*(self.B_c[i]*dgamma**2-self.grun[i]*(self.dBdP_c[i]*dgamma+self.B_c[i]*d2gamma)))/self.grun_q[i];
                else:
                    yk=1+self.grun0-self.grun_ideal
                    K=(self.grun_const_d*self.eta[i]**self.grun_const_d-self.grun_const_c*yk*self.eta[i]**self.grun_const_c)
                    dK=(self.grun_const_d**2*self.eta[i]**self.grun_const_d-self.grun_const_c**2*yk*self.eta[i]**self.grun_const_c)
                    self.grun[i]=yk*self.eta[i]**self.grun_const_c-self.eta[i]**self.grun_const_d+self.grun_ideal
                    self.grun_q[i]=K/self.grun[i]
                    self.grun_lamb[i]=-(dK/K - self.grun_q[i])
        return self.grun,self.grun_q,self.grun_lamb
    
    
    
####### Heat capacity ################   
#NOTE the full debye model is not implemented in this version of the code    
    def calculate_Cv(self,rho,T=0):
        self.checkmat(rho)
        Cv0=24943/(self.AA)*1.0
        self.Cv=Cv0*np.ones_like(self.rho);
        self.dlnCvdlnrho=0.0*np.ones_like(self.rho);
        self.d2lnCvdlnrho=0.0*np.ones_like(self.rho);
        self.dlnCvdlnT=0.0*np.ones_like(self.rho);

####### Internal energy ################
    def calculate_U(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_P_cold(self.rho)
        self.calculate_grun_cold(self.rho)
        self.calculate_Cv(self.rho)
        self.calculate_U_cold(self.rho)
        if type(self.U) is not np.ndarray:
            self.U_therm = self.Cv*T
            self.U = self.U_c + self.U_therm
        return self.U
            
    def calculate_U_cold(self,rho):
        self.checkmat(rho)
        self.calculate_P_cold(self.rho)
        self.calculate_grun_cold(self.rho)
        if type(self.U_c) is not np.ndarray:
            self.U_c = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] >= self.rho0):              
                    d=(1.-self.A1)/self.A1;
                    xae=(self.A0/self.A1);
                    K=(self.B0*math.exp(xae)/(self.A1*self.A0*self.rho0))*xae**(-d);
                    n=(self.A1+self.A2)/self.A1;
                    b=1./self.A1;
                    xe=xae*(self.eta[i])**(-self.A1);
                    C = xae**(n-1.)*mp.gammainc(1-n,a=xae);
                    Fc1=xae**(self.A2/self.A1)*(1./b*(xe**b*mp.gammainc(1-n,a=xe)-mp.gammainc(1-n+b,a=xe)));
                    Fc2=C*xe**(d+1.)/(d+1.);
                    Fc10=xae**(self.A2/self.A1)*self.G2_const;
                    Fc20=C*xae**(d+1.)/(d+1.);
                    self.U_c[i]=-K*(Fc1-Fc2-Fc10+Fc20);
                else:
                    K=self.B0/(self.Pexp_const_a-self.Pexp_const_b)
                    a=self.Pexp_const_a
                    b=self.Pexp_const_b
                    self.U_c[i]=K*(self.rho[i]**(a-1.)/(self.rho0**(a)*(a-1.))+(self.rho[i]**(b-1.))/(self.rho0**(b)*(1.-b)))
                    
####### Extract temperature from internal energy ################
    def calculate_TfromU(self,rho,U):
        self.checkmat(rho)
        self.calculate_P_cold(self.rho)
        self.calculate_grun_cold(self.rho)
        self.calculate_Cv(self.rho)
        self.calculate_U_cold(rho)
        T=(U-self.U_c)/self.Cv
        return T

####### FULL PRESSURE ################
## NOTE: This version of the code does not include
## the replacement of the unphysical expanded state pressure
## with a constant vapor pressure. Mainly because this code was used
## for analysis. For hydrocode implementation follow the instructions
## given in the paper.
        
    def calculate_P(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_P_iso(self.rho,self.T)
        
        
    def calculate_P_iso(self,rho,T):
        self.checkmat(rho,T)    
        self.calculate_P_cold(self.rho)    
        self.calculate_grun_cold(self.rho)
        self.calculate_Cv(self.rho)
        if type(self.P) is not np.ndarray:
            self.P_vib = np.zeros_like(self.rho)
            self.P = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                if(self.rho[i] > self.rho0):
                    self.P_vib[i] = self.Cv[i]*T*self.grun[i]*self.rho[i]
                    self.P[i] = self.P_c[i]+self.P_vib[i]
                else:
                    self.P_vib[i] = self.Cv[i]*T*self.grun[i]*self.rho[i]
                    self.P[i] = self.P_c[i]+self.P_vib[i]
            
####### ISOTHERMAL BULK MODULUS ################      
    def calculate_B_iso(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_B_cold(self.rho)
        self.calculate_grun_cold(self.rho)
        self.calculate_Cv(self.rho)
        self.calculate_P_iso(self.rho,self.T)
        if type(self.B_iso) is not np.ndarray:
            self.B_iso = self.B_c+self.P_vib*(1-self.grun_q+self.dlnCvdlnrho)
    
####### PRESSURE DERIVATIVES OF ISOTHERMAL BULK MODULUS ################
    def calculate_dBdP_iso(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_B_iso(self.rho,self.T)
        self.calculate_dBdP_cold(self.rho)
        if type(self.dBdP_iso) is not np.ndarray:
            self.dBdP_iso=self.dBdP_c*self.B_c/self.B_iso+(self.P_vib/self.B_iso)*( (1-self.grun_q+self.dlnCvdlnrho)**2+self.grun_lamb*self.grun_q+self.d2lnCvdlnrho)

####### ADIABATIC BULK MODULUS ################    
    def calculate_B(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_B_cold(self.rho)
        self.calculate_P(self.rho,self.T)
        self.calculate_grun_cold(self.rho)
        self.calculate_Cv(self.rho)
        if type(self.B) is not np.ndarray:
            self.B = self.B_c+self.P_vib*(1-self.grun_q+self.grun+self.dlnCvdlnrho)
####### PRESSURE DERIVATIVES OF ADIABATIC BULK MODULUS ################
    def calculate_dBdP(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_dBdP_iso(rho,self.T)
        if type(self.dBdP) is not np.ndarray:
            yaT=self.grun*(self.P_vib/self.B_iso)
            self.dBdP=(self.dBdP_iso+yaT*(2+self.grun-3*self.grun_q-self.grun*self.dlnCvdlnT))/(1+yaT)
###### THERM EXP ####
    def calculate_thermexp(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_P(self.rho,self.T)
        self.calculate_B_iso(self.rho,self.T)
        if self.thermexp is not np.ndarray:
            self.thermexp=(self.grun*self.rho*self.Cv/self.B_iso)
###### HEAT CAPACITY RATIO ####        
    def calculate_heatcapratio(self,rho,T):
        self.checkmat(rho,T)
        self.thermexp(self.rho,self.T)
        if self.heatcapratio is not np.ndarray:
            self.heatcapratio=1+self.grun*self.thermexp*T
###### Cp ####        
    def calculate_Cp(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_heatcapratio(self.rho,self.T)
        if self.Cp is not np.ndarray:
            self.Cp=self.heatcapratio*self.Cv
        
### Entalphy
    def calculate_H(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_U(self.rho,self.T)
        self.calculate_P(self.rho,self.T)
        if type(self.H) is not np.ndarray:
            self.H=self.U+self.P/self.rho
## Entropy      
    def calculate_S(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_grun_integral(self.rho)
        self.calculate_Cv(self.rho)
        if type(self.S) is not np.ndarray:
            self.S=self.Cv*(np.log(self.T)-self.grun_int)
## Gibbs free energy        
    def calculate_G(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_U(self.rho,self.T)
        self.calculate_S(self.rho,self.T)
        self.calculate_P(self.rho,self.T)
        if type(self.G) is not np.ndarray:
            self.G=self.U+self.P/self.rho-self.T*self.S
## Helmholtz free energy
    def calculate_F(self,rho,T):
        self.checkmat(rho,T)
        self.calculate_U(self.rho,self.T)
        self.calculate_S(self.rho,self.T)
        if type(self.F) is not np.ndarray:
            self.F=self.U-T*self.S
## Speed of sound            
    def calculate_cs(self,rho,u):
        self.calculate_B(rho,u)
        if self.cs is not np.ndarray:
            self.cs=math.sqrt(self.B/self.rho)
## dudrho
    def func_dudrho(self,u,rho):
        T=self.calculate_TfromU(rho,u)
        print('myu and T and rho!!',u,T,rho)
        self.calculate_P(rho,T)
        print(rho,u,self.P)
        du=self.P/rho**2
        return du
    
    def func_lindemann(self,Tm,rho,u):
        self.calculate_grun_cold(rho)
        dTm=2*Tm/self.rho*(self.grun-1/3);  
        return dTm

    def calculate_all(self,rho,u):
        pass

## Get isentrope from du=P/rho**2 drho
    def getisentrope_U(self,rho,T):
        self.calculate_U(rho,T)
        ustart=self.U[0]
        Ttest=self.calculate_TfromU(rho,ustart)
        us=integrate.odeint(self.func_dudrho, ustart, rho, rtol=1e-11, mxhnil=0)
        return us
    
    def getisotrope(self,rho,T):
        self.calculate_P(rho,T)
        us=np.trapz(self.P/rho**2,rho)
        return us

## Get isentrope from gruneisen    
    def getisentrope_grun(self,rho,T,compress=1):
        self.calculate_grun_integral(rho)
        self.calculate_Cv(rho)
        self.calculate_P_cold(rho)
        if(compress == 1):
            self.T_adi=np.exp(self.grun_int+np.log(T))
        else:
            self.T_adi=np.exp(np.log(T)-self.grun_int2) 
        self.P_adi=self.P_c+self.grun*self.Cv*self.rho*self.T_adi
        self.B_adi=self.B_c+self.grun*self.Cv*self.rho*self.T_adi*(1-self.grun_q+self.grun+self.dlnCvdlnrho)
        self.cs_adi=np.sqrt(self.B_adi/self.rho)
        return self.T_adi
    
    def grun_integrand(self,rho):
        self.calculate_grun_cold(rho)
        us=self.grun/rho
        return us
    
    
    def setup_integrator(self):
        self.test=varpoly()
        self.test.creatematerial(self.rho0,self.B0,self.dBdP0,self.AA,self.ZZ,self.tmin,self.delta)
        self.test.createexpstate(a=self.Pexp_const_a,b=self.Pexp_const_b,c=self.grun_const_c)
    
    
    
    def calculate_grun_integral(self,rho):
        self.checkmat(rho)
        self.calculate_grun_cold(rho)
        self.setup_integrator()
        if type(self.grun_int) is not np.ndarray:
            self.grun_int = np.zeros_like(self.rho)
            self.grun_int2 = np.zeros_like(self.rho)
            for i in range(len(self.rho)):
                    #self.grun_int[i]=np.trapz(self.grun[:(i+1)]/self.rho[:(i+1)],self.rho[:(i+1)])
                    integ=integrate.quad(self.test.grun_integrand, 0.0000001 ,self.rho[i])
                    #integ2=integrate.quad(self.test.grun_integrand, self.rho[-1],self.rho[i])
                    self.grun_int[i]=integ[0]
                    #self.grun_int2[i]=-integ2[0]

## calculate Hugoniot curve
    def gethugoniot(self,rho):
        rho=np.linspace(self.rho0,rho[-1],len(rho))
        self.checkmat(rho);
        self.calculate_U_cold(rho)
        self.calculate_grun_cold(rho)
        self.calculate_P_cold(rho)
        self.calculate_Cv(rho)
        D=(1./self.rho0+1./self.rho)/2.
        C=(1-self.rho0/self.rho);
        D2=0.5*(self.rho/self.rho0-1)*self.grun-1
        self.Th = (((self.U_c+self.P_c/self.rho-D*self.P_c)/self.Cv)/D2)
        self.Ph = self.P_c+self.grun*self.Cv*self.Th*self.rho
        self.Uh = self.U_c+self.Cv*self.Th
        self.us=np.sqrt(self.B_c/self.rho0)+np.sqrt(C*self.Ph/self.rho0)
        self.up=C*self.us
        
## fit to Hugoniot curve
    def fittohugoniot(self,xdata,ydata,lowlimits,uplimits):
        xdata=xdata*self.rho0
        popt, pcov = curve_fit(self.fittohugoniotfunc, xdata, ydata, bounds=(lowlimits, uplimits))
        return popt, pcov
        
    def fittohugoniotfunc(self,x,B,tmin,delta):
        self.test=varpoly()
        self.test.creatematerial(self.rho0,B,self.dBdP0,self.AA,self.ZZ,tmin,delta)
        self.test.createexpstate(self.Pexp_const_a,c=self.grun_const_c)
        self.test.gethugoniot(x)
        return self.test.Ph
## fit cold expanded state curve to Evap 
    def fittovapenergy(self,E):
        self.setup_integrator()
        res = minimize(self.fittovapenergyfunc,2.0,args=(E))
        return res.x
        
    def fittovapenergyfunc(self,x,E):
        self.test.createexpstate(x,c=self.grun_const_c)
        integE=self.B0/((x-self.Pexp_const_b)*self.rho0)*(1./(x-1.)-1./(self.Pexp_const_b-1.))
        print(integE[0],integE[0],E,x)
        return abs(integE[0]+E)
        
    def fittovapenergyinteg(self,rho):
        self.calculate_P_cold(rho)
        return self.P_c/rho**2
    
## Fit c_exp to a certain Tcrit       
    def fittoTcrit(self,Tc,c0=0.6):
        self.setup_integrator()
        res = minimize(self.fittoTcritfunc,c0,args=(Tc),method='Nelder-Mead')
        return res.x

    def fittoTcritfunc(self,x,Tc):
        print(x)
        self.test.createexpstate(a=self.Pexp_const_a,c=x)
        Tcritfound=self.test.findTcrit()
        print("Crit T: ",Tcritfound,Tc)
        return abs(Tcritfound-Tc)
        
## Get Tcrit value        
    def findTcrit(self):
        rho=np.array(np.linspace(0,self.rho0,2000))
        self.checkmat(rho)
        Tmax=100000
        Tstart=1000
        Tint=1000
        while Tint>=1.:
            self.calculate_B_iso(rho,Tstart)
            zero_crossings = np.where(np.diff(np.signbit(self.B_iso)))[0]
            #print(Tstart,Tint,zero_crossings,len(zero_crossings));
            if len(zero_crossings)==0:
                Tstart-=Tint;
                Tint=Tint*0.1;
            else:
                Tstart+=Tint;
            if Tstart>Tmax:
                print("Could not find critical temperature Tmax: ",Tmax)
                return 0.0
        self.T_crit=Tstart;
        self.calculate_B_iso(rho,Tstart-1.0)
        self.calculate_P(rho,Tstart-1.0)
        zero_crossings = np.where(np.diff(np.signbit(self.B_iso)))[0]
        print("Found crit T:",self.T_crit," rho crit: ",rho[zero_crossings]," Pcrit ",self.P[zero_crossings]/10**9)
        return Tstart        
        
    def vap_integrand(self,rho,T):
        self.calculate_P(rho,T)
        us=self.P/rho**2
        return us
## Calculate vapor curve, very clunky and slow implementation right now
## Requires larger n to sample low T
## Will probably improve in the future
    def getvaporcurve(self,n=14000000,Tmin=3225,rhomin=0.0001):
        self.findTcrit();
        rho=np.logspace(np.log10(rhomin),np.log10(self.rho0*3.0/4.0),round(3*n/4))
        rho2=np.linspace(self.rho0*3.0/4.0+1.0,self.rho0,round(n/4))
        rho=np.append(rho,rho2)
        self.setup_integrator()
        self.checkmat(rho);
        Trange=np.linspace(Tmin,self.T_crit-5,10)
        #Trange=np.linspace(Tmin,Tmin+100,2)
        for T in Trange:
            self.setup_integrator()
            self.calculate_B_iso(self.rho,T)
            self.calculate_P(self.rho,T)
            self.calculate_U(self.rho,T)
            clim = np.where(np.diff(np.signbit(self.B_iso)))[0]
            crange=clim
            v1=0.
            v2=1.
            varea=130000.
            tol=3000
            cmid=int(np.round(clim[0]+clim[1])/2.)
            cmid0=cmid
            print(crange,cmid,T,len(rho))
            while np.abs(varea) > tol:
                Pvap=self.P[cmid]
                if(Pvap < 0):
                    crange=np.array([crange[0],cmid])
                    cmid=int(np.round(crange[0]+crange[1])/2.)
                    continue
                Pnew=self.P-self.P[cmid]
                cP=np.where(np.diff(np.signbit(Pnew)))[0]
                self.test.calculate_F(rho[[cP[0],cP[-1]]],T)
                print(Pnew[cP[0]],Pnew[cP[-1]])
                Fnew=self.test.F+self.P[cmid]/self.test.rho
                
                varea=Fnew[-1]-Fnew[0]
                #print("varea",varea,"T",T,"tol")
                if np.abs(varea) > tol:
                    if (crange[1]-crange[0] < 2):
                        exit
                    if varea > 0:
                        if crange[1]-crange[0] < 8.0:
                            cmid=crange[1]-1
                        crange=np.array([crange[0],cmid])
                    if varea < 0:
                        if crange[1]-crange[0] < 8.0:
                            cmid=crange[0]+1
                        crange=np.array([cmid,crange[1]])
                    cmid=int(np.round(crange[0]+crange[1])/2.)

            print('PVAP',self.P[cmid])
            print("P1",self.P[cP[0]]/10**9,"P2",self.P[cP[-1]]/10**9)
            print("S1",self.test.S[0]-self.test.S[-1]+3450,"S2",self.test.S[-1])
            print("rho1",self.rho[cP[0]],"rho2",self.rho[cP[-1]])
            self.Pvapstate=np.append([self.Pvapstate],[self.P[cmid],self.P[cmid]])
            self.rhovapstate=np.append([self.rhovapstate],[self.rho[cP[-1]],self.rho[cP[0]]])
            self.Svapstate=np.append([self.Svapstate],[self.test.S[-1],self.test.S[0]])
            self.Uvapstate=np.append([self.Uvapstate],[self.U[cP[-1]],self.U[cP[0]]])
            self.Tvapstate=np.append([self.Tvapstate],[self.T,self.T])
        argidx=np.argsort(self.rhovapstate)
        self.Pvapstate=self.Pvapstate[argidx]
        self.rhovapstate=self.rhovapstate[argidx]
        self.Tvapstate=self.Tvapstate[argidx]
        self.Svapstate=self.Svapstate[argidx]
        self.Uvapstate=self.Uvapstate[argidx]

## Melt from Lindemann, however not that valid of a method,
## should be found by Gibbs equality of melt and solid
    def getmeltcurve(self,rhorange,Tmstart,u):
        Tm=integrate.odeint(self.func_lindemann, Tmstart, rhorange, args=(u,), rtol=1e-11, mxhnil=0)
        return Tm        
    
    def fitparameters(self,a,b,B0):
        pass