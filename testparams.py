# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 23:26:47 2024

@author: ylugn
"""
import simulation as sm
import saveload as sl
import animate as an

#Set parameters
q=49
nl=3
m=1000
N=2000
SS=10
t0=1
noi=0
n=1000
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
rf = "testparams6" #rf is the rfilebase parameter

#Create Coupled Particle object (this object holds all simulation logic)
cp = sm.CoupledParticle(N=N,SS=SS,t0=t0,noi=noi,n=n,T=T,gam=gam,d=d,C=C,p=p,a=a\
                        ,delt=delt,dist=dist,iv=iv,ev=ev,its=its)
    
#Generate a SimulateSaveLoad object (a wrapper class for generating/saving/loading data)
#op is a safety parameter.
    #op=0: if safety checks fail, the program will return an error and do nothing
    #op=1: if safety checks fail, the program will override the input parameters in favor of local parameters
Present = sl.SimulateSaveLoad("testparams12",rfilebase=rf,cp=cp,op=0,q=q,nl=nl,m=m)

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
ann = an.Animate(filebase = Present.filebase)

#Create animations
ann.CoupledParticleCloud()          #The full particle cloud
ann.CoupledParticleCloudBR()        #The particle cloud highlighting the center of mass (in black) and one particle's trajectory (in red)
#ann.CoupledParticleCloudR()         #The above but without the center of mass
ann.AnimateParticleSample(10)         #Animation showing SS particle trajectories
ann.AnimateSubNetwork(10)             #Animation showing the subnetwork connecting SS particles
ann.AnimateParticleNetworkSample(10)  #Animation combining the two animations above
