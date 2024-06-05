# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:22:21 2024

@author: ylugn
"""

import simulation as sm
import saveload as sl
import animate as an

N=1000
SS=10
t0=1
noi=0
n=1000
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

#Create Coupled Particle object
cp = sm.CoupledParticle(N=N,SS=SS,t0=t0,noi=noi,n=n,T=T,gam=gam,d=d,C=C,p=p,a=a\
                        ,delt=delt,dist=dist,iv=iv,ev=ev,its=its)
  
#Create Coupled Particle object
#cp = sm.CoupledParticle()
  
#Generate a SimulateSaveLoad object 
Present = sl.SimulateSaveLoad("runv1",cp=cp,op=0,m=100)

#Construct reference measure
Present.saveref()

#Run a coupled version
Present.saveCoupleGivenRef(Present.filebase)

#Run m copies
Present.savemCoupledStatistics(Present.filebase,DEBUG = True)

#Create an animation
ann = an.Animate(filebase = Present.filebase)

#Create animations
ann.CoupledParticleCloud()
ann.CoupledParticleCloudBR()
ann.CoupledParticleCloudR()
ann.AnimateParticleSample()
ann.AnimateSubNetwork()
ann.AnimateParticleNetworkSample()
