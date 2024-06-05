# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:32:01 2024

@author: ylugn
"""
import simulation as sm
import saveload as sl
import matplotlib.pyplot as plt
import numpy as np
import os

#I wrote this under the assumption that d = 2
class PlotStats():
    def __init__(self,filebase):
        cp = sm.CoupledParticle()
        
        #load stored data (if filebase does not exist, creates an empty directory)
        self.ssl = sl.SimulateSaveLoad(filebase,op=1,cp=cp)
        
        if self.ssl.cp.d != 2:
            raise Exception("PlotStats class only works in 2 dimensions.")
            
    #Plot means
    def pltmeans(self,sm,smMF,ttl,ttlMF):
        #Get the maximal bounds of the plot
        maxlim = np.ceil(np.max(np.abs(sm[:,0,:]))/3)*3
        maxlimMF = np.ceil(np.max(np.abs(smMF[:,0,:]))/3)*3

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlim(-maxlim,maxlim)
        axes[0].set_ylim(-maxlim,maxlim)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Mean x coordinate of particles (in sqrt(n) units)")
        axes[1].set_xlim(-maxlimMF,maxlimMF)
        axes[1].set_ylim(-maxlimMF,maxlimMF)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Mean x coordinate of MF particles (in sqrt(n) units)")
        
        #Add title
        axes[0].set_title(ttl)
        axes[1].set_title(ttlMF)
        
        #Initialize scatter plots
        axes[0].boxplot(sm[0,:,:].T*np.sqrt(self.ssl.cp.n))
        axes[1].boxplot(smMF[0,:,:].T*np.sqrt(self.ssl.cp.n))
        
        fig.savefig(self.ssl.pth+"\\xplt.pdf")
        