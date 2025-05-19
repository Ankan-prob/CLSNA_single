# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:49:52 2024

@author: ylugn
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:22:21 2024

@author: ylugn
"""

import simulation as sm
import saveload as sl
import animate as an

#Set parameters
q=49
nl=3
m=101
N=4000
SS=10
t0=1
noi=0
n=10
T=100
gam=0.3
d=2
C=0.5
p=0.15
a=1
delt=1
dist=2
iv=3
ev=3
its=100
rf = "runwv2n10N4k" #rf is the rfilebase parameter

#Create Coupled Particle object (this object holds all simulation logic)
cp = sm.CoupledParticle(N=N,SS=SS,t0=t0,noi=noi,n=n,T=T,gam=gam,d=d,C=C,p=p,a=a\
                        ,delt=delt,dist=dist,iv=iv,ev=ev,its=its)
  
#Create Coupled Particle object
#cp = sm.CoupledParticle()
  
#Generate a SimulateSaveLoad object (a wrapper class for generating/saving/loading data)
#op is a safety parameter.
    #op=0: if safety checks fail, the program will return an error and do nothing
    #op=1: if safety checks fail, the program will override the input parameters in favor of local parameters
Present = sl.SimulateSaveLoad("MF4000",rfilebase=rf,cp=cp,op=0,q=q,nl=nl,m=m)

#Construct reference measure for MF computations if necessary
#If not necessary, this will return a warning and do nothing
Present.saveref()

#Run the n particle simulation and coupled MF simulation
Present.saveCoupleGivenRef(Present.filebase)

#Get the statistics from m runs of both simulations
#Optional argument: DEBUG (default: False)
#DEBUG saves the full particle/network trajectories as well as the statistics. Only use with small parameters!
Present.savemCoupledStatistics(Present.filebase)

#Create an animation object
#Creating this object will return the warning "Parameters may have changed."
#The parameters have not changed.
ann = an.Animate(filebase = Present.filebase)

#Create animations
ann.CoupledParticleCloud()          #The full particle cloud
ann.CoupledParticleCloudBR()        #The particle cloud highlighting the center of mass (in black) and one particle's trajectory (in red)
ann.CoupledParticleCloudR()         #The above but without the center of mass
ann.AnimateParticleSample()         #Animation showing SS particle trajectories
ann.AnimateSubNetwork()             #Animation showing the subnetwork connecting SS particles
ann.AnimateParticleNetworkSample()  #Animation combining the two animations above
