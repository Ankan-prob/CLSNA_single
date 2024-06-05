# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:22:21 2024

@author: ylugn
"""

import simulation as sm
import saveload as sl
import animate as an

#Set parameters
N=10000
SS=10
t0=1
noi=0
n=10
T=100
gam=0.9
d=2
C=0.5
p=0.15
a=1
delt=1
dist=1
iv=2
ev=2
its=10
rfilebase = "runv1n10"

#Create Coupled Particle object (this object holds all simulation logic)
cp = sm.CoupledParticle(N=N,SS=SS,t0=t0,noi=noi,n=n,T=T,gam=gam,d=d,C=C,p=p,a=a\
                        ,delt=delt,dist=dist,iv=iv,ev=ev,its=its)
  
#Create Coupled Particle object
#cp = sm.CoupledParticle()
  
#Generate a SimulateSaveLoad object (a wrapper class for generating/saving/loading data)
Present = sl.SimulateSaveLoad("runv1n10",cp=cp,op=0,m=100)

#Construct reference measure for MF computations if necessary
#If not necessary, this will return a warning and do nothing
Present.saveref()

#Run the n particle simulation and coupled MF simulation
Present.saveCoupleGivenRef(Present.filebase)

#Get the statistics from m runs of both simulations
Present.savemCoupledStatistics(Present.filebase,DEBUG = True)

#Create an animation object
ann = an.Animate(filebase = Present.filebase)

#Create animations
ann.CoupledParticleCloud()          #The full particle cloud
ann.CoupledParticleCloudBR()        #The particle cloud highlighting the center of mass (in black) and one particle's trajectory (in red)
ann.CoupledParticleCloudR()         #The above but without the center of mass
ann.AnimateParticleSample()         #Animation showing SS particle trajectories
ann.AnimateSubNetwork()             #Animation showing the subnetwork connecting SS particles
ann.AnimateParticleNetworkSample()  #Animation combining the two animations above
