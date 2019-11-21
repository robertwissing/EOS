# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:41:54 2017

@author: Khyzath
"""
from eos import eos
import math
import scipy.integrate as integrate

class tillotson(eos):
    
    def __init__(self):
        pass
    
    def creatematerial(self,rho0,a,b,B0,C,u0):
        self.rho0=rho0
        self.a=a
        self.b=b
        self.B0=B0
        self.C=C
        self.u0=u0
        self.D=2*b*rho0/u0;
    
    def getmaterial(self):
        pass
    
    def calculate_P(self,rho,u):
        self.checkmat(rho,u)
        if not self.P:
            self.P=(self.a+self.b/self.w)*self.u*self.rho+self.B0*self.eta2+self.C*self.eta2**2;
        # check if same rho and u as before
        # check what variables have been calculated
        # allow for rho and u arrays
        
    def checkmat(self,rho,u):
        if self.rho==rho and self.u==u:
            pass
        else:
            self.updatemat(rho,u)
            
    def updatemat(self,rho,u):
        self.rho=rho
        self.u=u
        self.eta=rho/self.rho0
        self.w=u/(self.u0*self.eta**2)+1;
        self.eta2=self.eta-1
        self.P=self.B=self.dBdP=self.y=self.vp=[]
    
    def calculate_B(self,rho,u):
        self.calculate_P(rho,u)
        if not self.B:
            self.B=self.B0+(1+self.a+self.b/self.w**2)*self.P+self.C*(self.eta**2-1)+self.D*(self.u**2./(self.eta*self.w**2));
    
    def calculate_dBdP(self,rho,u):
        self.calculate_B(rho,u)
        self.calculate_y(rho,u)
        if not self.dBdP:
            omg=self.P/(u*rho);
            self.dBdP=1+self.y+2*self.C*self.eta**2./self.B+self.D*self.u**2/(self.eta*self.w**3*self.B)*(3*omg+3*self.w-4-omg**2);
    
    def calculate_y(self,rho,u):
        self.checkmat(rho,u)
        if not self.y:
            self.y=self.a+self.b/self.w**2
    
    def calculate_vp(self,rho,u):
        self.calculate_B(rho,u)
        if not self.vp:
            self.vp=math.sqrt(self.B/self.rho)
    
    def func_dudrho(self,u,rho):
        self.calculate_P(rho,u)
        du=self.P/rho**2
        return du
    
    def func_lindemann(self,Tm,rho,u):
        self.calculate_y(rho,u)
        dTm=2*Tm/self.rho*(self.y-1/3);  
        return dTm

    def calculate_all(self,rho,u):
        pass
    
    def getisentrope(self,rhorange,ustart):
        us=integrate.odeint(self.func_dudrho, ustart, rhorange, rtol=1e-11, mxhnil=0)
        us=us[:,0]
        return us
    
    def calculate_coldcurve(self,rhorange):
        self.E0=self.getisentrope(rhorange,0)
    
    def gethugoniot(self,rho,u):
        self.checkmat(rho,u);
        k=self.u0*self.eta**2;
        L=self.B0*self.eta2+self.C*self.eta2**2;
        S=0.5*(1/self.rho0-1/self.rho);
        ah=(S-self.a*self.rho*S**2)/k;
        bh=1-(self.a+self.b)*self.rho*S-(S*L)/k;
        ch=L;
        self.Ph=(-bh+math.sqrt(bh**2+4*ah*ch))/(2*ah);
        self.uh=0.5*self.Ph*(1/self.rho0-1/self.rho);

    def getmeltcurve(self,rhorange,Tmstart,u):
        Tm=integrate.odeint(self.func_lindemann, Tmstart, rhorange, args=(u,), rtol=1e-11, mxhnil=0)
        return Tm        
    
    def fitparameters(self,a,b,B0):
        pass