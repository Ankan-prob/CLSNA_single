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

filebase1 = "testparams12"
filebase2 = "testparams13"
filebase3 = "testparams14"
filebase4 = "testparams15"
filebase5 = "testparams16"
filebase6 = "testparams17"
filebase7 = "testparams18"

cp1 = sm.CoupledParticle()
cp2 = sm.CoupledParticle()
cp3 = sm.CoupledParticle()
cp4 = sm.CoupledParticle()
cp5 = sm.CoupledParticle()
cp6 = sm.CoupledParticle()
cp7 = sm.CoupledParticle()


#load stored data (if filebase does not exist, creates an empty directory)
ssl1 = sl.SimulateSaveLoad(filebase1,op=1,cp=cp1)
ssl2 = sl.SimulateSaveLoad(filebase2,op=1,cp=cp2)
ssl3 = sl.SimulateSaveLoad(filebase3,op=1,cp=cp3)
ssl4 = sl.SimulateSaveLoad(filebase4,op=1,cp=cp4)
ssl5 = sl.SimulateSaveLoad(filebase5,op=1,cp=cp5)
ssl6 = sl.SimulateSaveLoad(filebase6,op=1,cp=cp6)
ssl7 = sl.SimulateSaveLoad(filebase7,op=1,cp=cp7)

sm1 = ssl1.loadmCoupledZStatistics()
smMF1 = sm1[5]
sm1 = sm1[0]
sm2 = ssl2.loadmCoupledZStatistics()
smMF2 = sm2[5]
sm2 = sm2[0]
sm3 = ssl3.loadmCoupledZStatistics()
smMF3 = sm3[5]
sm3 = sm3[0]
sm4 = ssl4.loadmCoupledZStatistics()
smMF4 = sm4[5]
sm4 = sm4[0]
sm5 = ssl5.loadmCoupledZStatistics()
smMF5 = sm5[5]
sm5 = sm5[0]
sm6 = ssl6.loadmCoupledZStatistics()
smMF6 = sm6[5]
sm6 = sm6[0]
sm7 = ssl7.loadmCoupledZStatistics()
smMF7 = sm7[5]
sm7 = sm7[0]

#ml1=0
ml1 = np.max(np.abs(np.mean(sm1[0,:,:],axis=1)))*np.sqrt(1000.0*1000.0)
mlMF1 = np.max(np.abs(np.mean(smMF1[0,:,:],axis=1)))*np.sqrt(2000)

#ml2=0
ml2 = np.max(np.abs(np.mean(sm2[0,:,:],axis=1)))*np.sqrt(500.0*1000.0)
#mlMF2 = np.max(np.abs(np.mean(smMF2[0,:,:],axis=1)))*np.sqrt(2000)
mlMF2=0

#ml3=0
ml3 = np.max(np.abs(np.mean(sm3[0,:,:],axis=1)))*np.sqrt(200.0*1000.0)
#mlMF3 = np.max(np.abs(np.mean(smMF3[0,:,:],axis=1)))*np.sqrt(2000)
mlMF3=0

#ml4=0
ml4 = np.max(np.abs(np.mean(sm4[0,:,:],axis=1)))*np.sqrt(100.0*1000.0)
#mlMF4 = np.max(np.abs(np.mean(smMF4[0,:,:],axis=1)))*np.sqrt(2000)
mlMF4=0

#ml5=0
ml5 = np.max(np.abs(np.mean(sm5[0,:,:],axis=1)))*np.sqrt(50.0*1000.0)
#mlMF5 = np.max(np.abs(np.mean(smMF5[0,:,:],axis=1)))*np.sqrt(2000)
mlMF5=0

#ml6=0
ml6 = np.max(np.abs(np.mean(sm6[0,:,:],axis=1)))*np.sqrt(20.0*1000.0)
#mlMF6 = np.max(np.abs(np.mean(smMF6[0,:,:],axis=1)))*np.sqrt(2000)
mlMF6=0

#ml7=0
ml7 = np.max(np.abs(np.mean(sm7[0,:,:],axis=1)))*np.sqrt(10.0*1000.0)
#mlMF7 = np.max(np.abs(np.mean(smMF7[0,:,:],axis=1)))*np.sqrt(2000)
mlMF7=0

mxlim = max(ml1,mlMF1,ml2,mlMF2,ml3,mlMF3,ml4,mlMF4,ml5,mlMF5,ml6,mlMF6,ml7,mlMF7)

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(-mxlim,mxlim)
axes.set_xlabel("Time")
axes.set_ylabel("Mean x coordinate of particles")

#Add title
axes.set_title("Mean x coordinates")

#Initialize line graphs
axes.plot(np.mean(smMF1[0,:,:]*np.sqrt(2000),axis=1), color = 'black',label = 'MF')
#axes.plot(np.mean(smMF2[0,:,:],axis=1), color = 'red')
#axes.plot(np.mean(smMF3[0,:,:],axis=1), color = 'blue')
#axes.plot(np.mean(smMF4[0,:,:],axis=1), color = 'green')
#axes.plot(np.mean(smMF5[0,:,:],axis=1), color = 'yellow')
#axes.plot(np.mean(smMF6[0,:,:],axis=1), color = 'orange')
axes.plot(np.mean(sm1[0,:,:]*np.sqrt(1000.0*1000.0),axis=1), color = 'red',label = 'n=1000')
axes.plot(np.mean(sm2[0,:,:]*np.sqrt(500.0*1000.0),axis=1), color = 'blue',label = 'n=500')
axes.plot(np.mean(sm3[0,:,:]*np.sqrt(200.0*1000.0),axis=1), color = 'green',label = 'n=200')
axes.plot(np.mean(sm4[0,:,:]*np.sqrt(100.0*1000.0),axis=1), color = 'yellow',label = 'n=100')
axes.plot(np.mean(sm5[0,:,:]*np.sqrt(50.0*1000.0),axis=1), color = 'orange',label = 'n=50')
axes.plot(np.mean(sm6[0,:,:]*np.sqrt(20.0*1000.0),axis=1), color = 'purple',label = 'n=20')
axes.plot(np.mean(sm7[0,:,:]*np.sqrt(10.0*1000.0),axis=1), color = 'black', linestyle = '--',label = 'n=10')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        