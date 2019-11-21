# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:16:49 2017

@author: Khyzath
"""
from abc import ABCMeta, abstractmethod
class eos(object):
    __metaclass__ = ABCMeta
    u=[]
    u0=[]
    rho0=[]
    P=[]
    def __init__(self):
        pass
    def creatematerial(self):
        pass
    def getmaterial(self):
        pass
    def calculate_P(self):
        pass
    def calculate_B(self):
        pass
    def calculate_dBdP(self):
        pass
