# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:32:01 2024

@author: ylugn
"""
import simulation as sm
import saveload as sl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import scipy.stats as stats

##################Load statistics
# =============================================================================
# filebase1 = "runwnv210"
# filebase2 = "runwnv220"
# filebase3 = "runwnv250"
# filebase4 = "runwnv2100"
# filebase5 = "runwnv2200"
# filebase6 = "runwnv2500"
# filebase7 = "runwnv21000"
# =============================================================================

#Generate using script.py
#eb = "runwv2N4kn" #N = 4000
eb = "runwv2n" #N=3000
#eb = "runwn" #N=2000
#eb = "runwv2n1kn" #N = 1000
#eb = "runwv2n5Hn" #N = 500

filebase1 = eb + "10"
filebase2 = eb + "20"
filebase3 = eb + "50"
filebase4 = eb + "100"
filebase5 = eb + "200"
filebase6 = eb + "500"
filebase7 = eb + "1000"

cp1 = sm.CoupledParticle()
cp2 = sm.CoupledParticle()
cp3 = sm.CoupledParticle()
cp4 = sm.CoupledParticle()
cp5 = sm.CoupledParticle()
cp6 = sm.CoupledParticle()
cp7 = sm.CoupledParticle()

#Start time for statistics that average over time
st = 20
tms = np.log(np.array([10,20,50,100,200,500,1000]))

#load stored data (if filebase does not exist, creates an empty directory)
ssl1 = sl.SimulateSaveLoad(filebase1,op=1,cp=cp1)
ssl2 = sl.SimulateSaveLoad(filebase2,op=1,cp=cp2)
ssl3 = sl.SimulateSaveLoad(filebase3,op=1,cp=cp3)
ssl4 = sl.SimulateSaveLoad(filebase4,op=1,cp=cp4)
ssl5 = sl.SimulateSaveLoad(filebase5,op=1,cp=cp5)
ssl6 = sl.SimulateSaveLoad(filebase6,op=1,cp=cp6)
ssl7 = sl.SimulateSaveLoad(filebase7,op=1,cp=cp7)

#Fix parameters
T = ssl7.cp.T
m = ssl7.m
d = ssl7.cp.d
N = ssl7.cp.N

#load reference MF particle trajectory
refmf = ssl1.loadref()

#Run an MF simulation.
#This should be deterministic and not depend on which ssl we choose
ssl1.saveMeanMF()
mMF = ssl1.loadMeanMF()

#load Z Statistics
sm1 = ssl1.loadmCoupledZStatistics()
sm2 = ssl2.loadmCoupledZStatistics()
sm3 = ssl3.loadmCoupledZStatistics()
sm4 = ssl4.loadmCoupledZStatistics()
sm5 = ssl5.loadmCoupledZStatistics()
sm6 = ssl6.loadmCoupledZStatistics()
sm7 = ssl7.loadmCoupledZStatistics()

#load A statistics
am1 = ssl1.loadmCoupledAStatistics()
am2 = ssl2.loadmCoupledAStatistics()
am3 = ssl3.loadmCoupledAStatistics()
am4 = ssl4.loadmCoupledAStatistics()
am5 = ssl5.loadmCoupledAStatistics()
am6 = ssl6.loadmCoupledAStatistics()
am7 = ssl7.loadmCoupledAStatistics()

###############Load Specific Statistics

#Load means and MF means
smMF1 = sm1[5]
smp1 = sm1[0]
smMF2 = sm2[5]
smp2 = sm2[0]
smMF3 = sm3[5]
smp3 = sm3[0]
smMF4 = sm4[5]
smp4 = sm4[0]
smMF5 = sm5[5]
smp5 = sm5[0]
smMF6 = sm6[5]
smp6 = sm6[0]
smMF7 = sm7[5]
smp7 = sm7[0]

#load mse
mse1 = sm1[10]
mse2 = sm2[10]
mse3 = sm3[10]
mse4 = sm4[10]
mse5 = sm5[10]
mse6 = sm6[10]
mse7 = sm7[10]

#load quantiles
qua1 = sm1[2]
qua2 = sm2[2]
qua3 = sm3[2]
qua4 = sm4[2]
qua5 = sm5[2]
qua6 = sm6[2]
qua7 = sm7[2]

quaMF1 = sm1[7]
quaMF2 = sm2[7]
quaMF3 = sm3[7]
quaMF4 = sm4[7]
quaMF5 = sm5[7]
quaMF6 = sm6[7]
quaMF7 = sm7[7]

#load graph densities
de1 = am1[0]
de2 = am2[0]
de3 = am3[0]
de4 = am4[0]
de5 = am5[0]
de6 = am6[0]
de7 = am7[0]
deMF1 = am1[5]
deMF2 = am2[5]
deMF3 = am3[5]
deMF4 = am4[5]
deMF5 = am5[5]
deMF6 = am6[5]
deMF7 = am7[5]

#load triangle densities
tde1 = am1[1]
tde2 = am2[1]
tde3 = am3[1]
tde4 = am4[1]
tde5 = am5[1]
tde6 = am6[1]
tde7 = am7[1]
tdeMF1 = am1[6]
tdeMF2 = am2[6]
tdeMF3 = am3[6]
tdeMF4 = am4[6]
tdeMF5 = am5[6]
tdeMF6 = am6[6]
tdeMF7 = am7[6]

#load clustering coefficients
cl1 = am1[2]
cl2 = am2[2]
cl3 = am3[2]
cl4 = am4[2]
cl5 = am5[2]
cl6 = am6[2]
cl7 = am7[2]
clMF1 = am1[7]
clMF2 = am2[7]
clMF3 = am3[7]
clMF4 = am4[7]
clMF5 = am5[7]
clMF6 = am6[7]
clMF7 = am7[7]

#load largest eigenvalues
le1 = am1[3][2,:,:]
le2 = am2[3][2,:,:]
le3 = am3[3][2,:,:]
le4 = am4[3][2,:,:]
le5 = am5[3][2,:,:]
le6 = am6[3][2,:,:]
le7 = am7[3][2,:,:]
leMF1 = am1[8][2,:,:]
leMF2 = am2[8][2,:,:]
leMF3 = am3[8][2,:,:]
leMF4 = am4[8][2,:,:]
leMF5 = am5[8][2,:,:]
leMF6 = am6[8][2,:,:]
leMF7 = am7[8][2,:,:]

#load second largest eigenvalues
sle1 = am1[3][1,:,:]
sle2 = am2[3][1,:,:]
sle3 = am3[3][1,:,:]
sle4 = am4[3][1,:,:]
sle5 = am5[3][1,:,:]
sle6 = am6[3][1,:,:]
sle7 = am7[3][1,:,:]
sleMF1 = am1[8][1,:,:]
sleMF2 = am2[8][1,:,:]
sleMF3 = am3[8][1,:,:]
sleMF4 = am4[8][1,:,:]
sleMF5 = am5[8][1,:,:]
sleMF6 = am6[8][1,:,:]
sleMF7 = am7[8][1,:,:]

#load third largest eigenvalues
tle1 = am1[3][0,:,:]
tle2 = am2[3][0,:,:]
tle3 = am3[3][0,:,:]
tle4 = am4[3][0,:,:]
tle5 = am5[3][0,:,:]
tle6 = am6[3][0,:,:]
tle7 = am7[3][0,:,:]
tleMF1 = am1[8][0,:,:]
tleMF2 = am2[8][0,:,:]
tleMF3 = am3[8][0,:,:]
tleMF4 = am4[8][0,:,:]
tleMF5 = am5[8][0,:,:]
tleMF6 = am6[8][0,:,:]
tleMF7 = am7[8][0,:,:]

#load smallest eigenvalues
se1 = am1[4][0,:,:]
se2 = am2[4][0,:,:]
se3 = am3[4][0,:,:]
se4 = am4[4][0,:,:]
se5 = am5[4][0,:,:]
se6 = am6[4][0,:,:]
se7 = am7[4][0,:,:]
seMF1 = am1[9][0,:,:]
seMF2 = am2[9][0,:,:]
seMF3 = am3[9][0,:,:]
seMF4 = am4[9][0,:,:]
seMF5 = am5[9][0,:,:]
seMF6 = am6[9][0,:,:]
seMF7 = am7[9][0,:,:]

#load second smallest eigenvalues
sse1 = am1[4][1,:,:]
sse2 = am2[4][1,:,:]
sse3 = am3[4][1,:,:]
sse4 = am4[4][1,:,:]
sse5 = am5[4][1,:,:]
sse6 = am6[4][1,:,:]
sse7 = am7[4][1,:,:]
sseMF1 = am1[9][1,:,:]
sseMF2 = am2[9][1,:,:]
sseMF3 = am3[9][1,:,:]
sseMF4 = am4[9][1,:,:]
sseMF5 = am5[9][1,:,:]
sseMF6 = am6[9][1,:,:]
sseMF7 = am7[9][1,:,:]

#load third smallest eigenvalues
ste1 = am1[4][2,:,:]
ste2 = am2[4][2,:,:]
ste3 = am3[4][2,:,:]
ste4 = am4[4][2,:,:]
ste5 = am5[4][2,:,:]
ste6 = am6[4][2,:,:]
ste7 = am7[4][2,:,:]
steMF1 = am1[9][2,:,:]
steMF2 = am2[9][2,:,:]
steMF3 = am3[9][2,:,:]
steMF4 = am4[9][2,:,:]
steMF5 = am5[9][2,:,:]
steMF6 = am6[9][2,:,:]
steMF7 = am7[9][2,:,:]

#load the number of edges in the symmetric difference graph
sd1 = am1[10]/2
sd2 = am2[10]/2
sd3 = am3[10]/2
sd4 = am4[10]/2
sd5 = am5[10]/2
sd6 = am6[10]/2
sd7 = am7[10]/2

# # of edges in AMF - # of edges in An
MFmn1 = -am1[11]/2
MFmn2 = -am2[11]/2
MFmn3 = -am3[11]/2
MFmn4 = -am4[11]/2
MFmn5 = -am5[11]/2
MFmn6 = -am6[11]/2
MFmn7 = -am7[11]/2

############Plot Means

#ml1=0
msm1 = np.mean(smp1[0,:,:],axis=1)*np.sqrt(10.0*m)
msm2 = np.mean(smp2[0,:,:],axis=1)*np.sqrt(20.0*m)
msm3 = np.mean(smp3[0,:,:],axis=1)*np.sqrt(50.0*m)
msm4 = np.mean(smp4[0,:,:],axis=1)*np.sqrt(100.0*m)
msm5 = np.mean(smp5[0,:,:],axis=1)*np.sqrt(200.0*m)
msm6 = np.mean(smp6[0,:,:],axis=1)*np.sqrt(500.0*m)
msm7 = np.mean(smp7[0,:,:],axis=1)*np.sqrt(1000.0*m)
msmMF1 = np.mean(smMF1[0,:,:],axis=1)*np.sqrt(10.0*m)
msmMF2 = np.mean(smMF2[0,:,:],axis=1)*np.sqrt(20.0*m)
msmMF3 = np.mean(smMF3[0,:,:],axis=1)*np.sqrt(50.0*m)
msmMF4 = np.mean(smMF4[0,:,:],axis=1)*np.sqrt(100.0*m)
msmMF5 = np.mean(smMF5[0,:,:],axis=1)*np.sqrt(200.0*m)
msmMF6 = np.mean(smMF6[0,:,:],axis=1)*np.sqrt(500.0*m)
msmMF7 = np.mean(smMF7[0,:,:],axis=1)*np.sqrt(1000.0*m)

#Get mean reference MF x coordinate (normalized to match msmMF7)
mrefmf = np.mean(refmf[0,:,:],0)*np.sqrt(ssl1.cp.N)*np.sqrt(10.0)
smMF = np.mean(mMF[0,:,:],0)*np.sqrt(1000.0*m)

#plot bounds
mxm1 = np.max(np.abs(msm1))
mxm2 = np.max(np.abs(msm2))
mxm3 = np.max(np.abs(msm3))
mxm4 = np.max(np.abs(msm4))
mxm5 = np.max(np.abs(msm5))
mxm6 = np.max(np.abs(msm6))
mxm7 = np.max(np.abs(msm7))

mxmMF1 = np.max(np.abs(msmMF1))
mxmMF2 = np.max(np.abs(msmMF2))
mxmMF3 = np.max(np.abs(msmMF3))
mxmMF4 = np.max(np.abs(msmMF4))
mxmMF5 = np.max(np.abs(msmMF5))
mxmMF6 = np.max(np.abs(msmMF6))
mxmMF7 = np.max(np.abs(msmMF7))

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxmMF7)
mxref = np.max(mrefmf)
mxmMF = np.max(smMF)
mxlimMFref = max(mxlimMF,mxref,mxmMF)

#%%
# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimMF,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Mean (normalized)")

# #Add title
# axes.set_title("Normalized Mean of x coordinates of MF particles")

# #Initialize line graphs
# axes.plot(msmMF1, color = 'black',label = 'n=10')
# axes.plot(msmMF2, color = 'red',label = 'n=20')
# axes.plot(msmMF3, color = 'blue',label = 'n=50')
# axes.plot(msmMF4, color = 'green',label = 'n=100')
# axes.plot(msmMF5, color = 'yellow',label = 'n=200')
# axes.plot(msmMF6, color = 'orange',label = 'n=500')
# axes.plot(msmMF7, color = 'purple',label = 'n=1000')
# #axes.plot(mrefmf, color = 'black', label = 'reference', linewidth = 3.0)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\meanMF"+eb+".pdf", bbox_inches = 'tight')

####Set bds for MF mean x coord simulations (iteration normalization)
imsmMF1 = np.mean(smMF1[0,:,:],axis=1)*np.sqrt(m)
imsmMF2 = np.mean(smMF2[0,:,:],axis=1)*np.sqrt(m)
imsmMF3 = np.mean(smMF3[0,:,:],axis=1)*np.sqrt(m)
imsmMF4 = np.mean(smMF4[0,:,:],axis=1)*np.sqrt(m)
imsmMF5 = np.mean(smMF5[0,:,:],axis=1)*np.sqrt(m)
imsmMF6 = np.mean(smMF6[0,:,:],axis=1)*np.sqrt(m)
imsmMF7 = np.mean(smMF7[0,:,:],axis=1)*np.sqrt(m)

mxmMF1 = np.max(np.abs(imsmMF1))
mxmMF2 = np.max(np.abs(imsmMF2))
mxmMF3 = np.max(np.abs(imsmMF3))
mxmMF4 = np.max(np.abs(imsmMF4))
mxmMF5 = np.max(np.abs(imsmMF5))
mxmMF6 = np.max(np.abs(imsmMF6))
mxmMF7 = np.max(np.abs(imsmMF7))

mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)

# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimMF,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Mean (normalized)")

# #Add title
# axes.set_title("Normalized Mean of x coordinates of MF particles (iteration normalization only)")

# #Initialize line graphs
# axes.plot(imsmMF1, color = 'black',linestyle = '--', label = 'n=10')
# axes.plot(imsmMF2, color = 'red',label = 'n=20')
# axes.plot(imsmMF3, color = 'blue',label = 'n=50')
# axes.plot(imsmMF4, color = 'green',label = 'n=100')
# axes.plot(imsmMF5, color = 'yellow',label = 'n=200')
# axes.plot(imsmMF6, color = 'orange',label = 'n=500')
# axes.plot(imsmMF7, color = 'purple',label = 'n=1000')
# axes.plot(smMF/np.sqrt(1000), color = 'black', label = 'noiseless', linewidth = 3.0)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\meanMFit"+eb+".pdf", bbox_inches = 'tight')

# ##Create figure and axes for MF/reference/noiseless x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimMFref,mxlimMFref)
# axes.set_xlabel("Time")
# axes.set_ylabel("Mean (scaled)")

# #Add title
# axes.set_title("Scaled Mean x Coordinate of Reference vs MF vs Noiseless MF Particles")

# #Initialize line graphs
# axes.plot(mrefmf, color = 'black', label = 'reference')
# axes.plot(msmMF7, color = 'red', label = 'MF (n=1000)')
# axes.plot(smMF, color = 'blue', label = 'Noiseless MF')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
# #Save figure
# plt.savefig("plots\\meanrefMF"+eb+".pdf", bbox_inches = 'tight')

# ######Create figure and axes for cross correlation (reference)
# fig, axes = plt.subplots()

# #generate data
# corr = sp.signal.correlate(msmMF7,mrefmf)
# corr = corr/np.max(corr)
# lags = sp.signal.correlation_lags(len(mrefmf),len(msmMF7))
# a = np.argmax(corr)
# mxcorr = lags[a]
# ys = np.linspace(0,1)
# xs = np.ones(ys.shape)*mxcorr

# #Create axes
# axes.set_xlim(np.min(lags),np.max(lags))
# axes.set_ylim(0,np.max(corr))
# axes.set_xlabel("Time Lag")
# axes.set_ylabel("Normalized Time-Shifted Convolution")

# #Add title
# axes.set_title("Time Lag Analysis of MF and Reference Particles (x Coord)")

# #Generate the plot
# axes.plot(lags,corr, color = 'black', label = 'time-shifted convolution')
# axes.plot(xs,ys, color = 'black', linestyle = '--', label = 'maximum time lag')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save the figure
# plt.savefig("plots\\crosscorr"+eb+".pdf", bbox_inches = 'tight')

# ######Create figure and axes for cross correlation (noiseless)
# fig, axes = plt.subplots()

# #generate data
# corr = sp.signal.correlate(msmMF7,smMF)
# corr = corr/np.max(corr)
# lags = sp.signal.correlation_lags(len(smMF),len(msmMF7))
# a = np.argmax(corr)
# mxcorr = lags[a]
# ys = np.linspace(0,1)
# xs = np.ones(ys.shape)*mxcorr

# #Create axes
# axes.set_xlim(np.min(lags),np.max(lags))
# axes.set_ylim(0,np.max(corr))
# axes.set_xlabel("Time Lag")
# axes.set_ylabel("Normalized Time-Shifted Convolution")

# #Add title
# axes.set_title("Time Lag Analysis of MF and noiseless MF Particles (x Coord)")

# #Generate the plot
# axes.plot(lags,corr, color = 'black', label = 'time-shifted convolution')
# axes.plot(xs,ys, color = 'black', linestyle = '--', label = 'maximum time lag')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save the figure
# plt.savefig("plots\\crosscorrNL"+eb+".pdf", bbox_inches = 'tight')

# ######Create figure and axes for cross correlation (MF vs MF)
# fig, axes = plt.subplots()

# #generate data
# corr = sp.signal.correlate(msmMF7,msmMF7)
# corr = corr/np.max(corr)
# lags = sp.signal.correlation_lags(len(msmMF7),len(msmMF7))
# a = np.argmax(corr)
# mxcorr = lags[a]
# ys = np.linspace(0,1)
# xs = np.ones(ys.shape)*mxcorr

# #Create axes
# axes.set_xlim(np.min(lags),np.max(lags))
# axes.set_ylim(0,np.max(corr))
# axes.set_xlabel("Time Lag")
# axes.set_ylabel("Normalized Time-Shifted Convolution")

# #Add title
# axes.set_title("Time Lag Analysis of MF Particles with themselves (x Coord)")

# #Generate the plot
# axes.plot(lags,corr, color = 'black', label = 'time-shifted convolution')
# axes.plot(xs,ys, color = 'black', linestyle = '--', label = 'maximum time lag')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save the figure
# plt.savefig("plots\\crosscorrMF"+eb+".pdf", bbox_inches = 'tight')

# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimT,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Mean (normalized)")

# #Add title
# axes.set_title("Normalized Mean of x Coordinates")

# #Initialize line graphs
# axes.plot(msm1, color = 'black', linestyle = '--', label = 'n=10')
# axes.plot(msm2, color = 'red',label = 'n=20')
# axes.plot(msm3, color = 'blue',label = 'n=50')
# axes.plot(msm4, color = 'green',label = 'n=100')
# axes.plot(msm5, color = 'yellow',label = 'n=200')
# axes.plot(msm6, color = 'orange',label = 'n=500')
# axes.plot(msm7, color = 'purple', label = 'n=1000')
# axes.plot(msmMF7, color = 'black', label = 'MF (n=1000)')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
# #Save figure
# plt.savefig("plots\\mean"+eb+".pdf", bbox_inches = 'tight')

# ###########Plot reference centered mean 
# mcsm1 = (np.mean(smp1[0,:,:],axis=1)- mrefmf/np.sqrt(N)) *np.sqrt(10.0*m)
# mcsm2 = (np.mean(smp2[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(20.0*m)
# mcsm3 = (np.mean(smp3[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(50.0*m)
# mcsm4 = (np.mean(smp4[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(100.0*m)
# mcsm5 = (np.mean(smp5[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(200.0*m)
# mcsm6 = (np.mean(smp6[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(500.0*m)
# mcsm7 = (np.mean(smp7[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(1000.0*m)
# mcsmMF1 = (np.mean(smMF1[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(10.0*m)
# mcsmMF2 = (np.mean(smMF2[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(20.0*m)
# mcsmMF3 = (np.mean(smMF3[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(50.0*m)
# mcsmMF4 = (np.mean(smMF4[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(100.0*m)
# mcsmMF5 = (np.mean(smMF5[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(200.0*m)
# mcsmMF6 = (np.mean(smMF6[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(500.0*m)
# mcsmMF7 = (np.mean(smMF7[0,:,:],axis=1)- mrefmf/np.sqrt(N))*np.sqrt(1000.0*m)

# #plot bounds
# mxm1 = np.max(np.abs(mcsm1))
# mxm2 = np.max(np.abs(mcsm2))
# mxm3 = np.max(np.abs(mcsm3))
# mxm4 = np.max(np.abs(mcsm4))
# mxm5 = np.max(np.abs(mcsm5))
# mxm6 = np.max(np.abs(mcsm6))
# mxm7 = np.max(np.abs(mcsm7))

# mxmMF1 = np.max(np.abs(mcsmMF1))
# mxmMF2 = np.max(np.abs(mcsmMF2))
# mxmMF3 = np.max(np.abs(mcsmMF3))
# mxmMF4 = np.max(np.abs(mcsmMF4))
# mxmMF5 = np.max(np.abs(mcsmMF5))
# mxmMF6 = np.max(np.abs(mcsmMF6))
# mxmMF7 = np.max(np.abs(mcsmMF7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
# mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
# mxlimT = max(mxlim,mxmMF7)

# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimMF,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Centered Mean (normalized)")

# #Add title
# axes.set_title("Reference Centered Normalized Mean of x Coordinates of MF Particles")

# #Initialize line graphs
# axes.plot(mcsmMF1, color = 'black',label = 'n=10')
# axes.plot(mcsmMF2, color = 'red',label = 'n=20')
# axes.plot(mcsmMF3, color = 'blue',label = 'n=50')
# axes.plot(mcsmMF4, color = 'green',label = 'n=100')
# axes.plot(mcsmMF5, color = 'yellow',label = 'n=200')
# axes.plot(mcsmMF6, color = 'orange',label = 'n=500')
# axes.plot(mcsmMF7, color = 'purple',label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\cmeanMF"+eb+".pdf", bbox_inches = 'tight')

# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimT,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Centered Mean (normalized)")

# #Add title
# axes.set_title("Reference Centered Normalized Mean of x Coordinates of Particles")

# #Initialize line graphs
# axes.plot(mcsm1, color = 'black', linestyle = '--', label = 'n=10')
# axes.plot(mcsm2, color = 'red',label = 'n=20')
# axes.plot(mcsm3, color = 'blue',label = 'n=50')
# axes.plot(mcsm4, color = 'green',label = 'n=100')
# axes.plot(mcsm5, color = 'yellow',label = 'n=200')
# axes.plot(mcsm6, color = 'orange',label = 'n=500')
# axes.plot(mcsm7, color = 'purple', label = 'n=1000')
# axes.plot(mcsmMF7, color = 'black', label = 'MF (n=1000)')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
# #Save figure
# plt.savefig("plots\\cmean"+eb+".pdf", bbox_inches = 'tight')

# ###########Plot MF centered mean 
# mMFcsm1 = np.mean(smp1[0,:,:]-smMF1[0,:,:],axis=1)*np.sqrt(10.0*m)
# mMFcsm2 = np.mean(smp2[0,:,:]-smMF2[0,:,:],axis=1)*np.sqrt(20.0*m)
# mMFcsm3 = np.mean(smp3[0,:,:]-smMF3[0,:,:],axis=1)*np.sqrt(50.0*m)
# mMFcsm4 = np.mean(smp4[0,:,:]-smMF4[0,:,:],axis=1)*np.sqrt(100.0*m)
# mMFcsm5 = np.mean(smp5[0,:,:]-smMF5[0,:,:],axis=1)*np.sqrt(200.0*m)
# mMFcsm6 = np.mean(smp6[0,:,:]-smMF6[0,:,:],axis=1)*np.sqrt(500.0*m)
# mMFcsm7 = np.mean(smp7[0,:,:]-smMF7[0,:,:],axis=1)*np.sqrt(1000.0*m)

# #plot bounds
# mxm1 = np.max(np.abs(mMFcsm1))
# mxm2 = np.max(np.abs(mMFcsm2))
# mxm3 = np.max(np.abs(mMFcsm3))
# mxm4 = np.max(np.abs(mMFcsm4))
# mxm5 = np.max(np.abs(mMFcsm5))
# mxm6 = np.max(np.abs(mMFcsm6))
# mxm7 = np.max(np.abs(mMFcsm7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)

# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlim,mxlim)
# axes.set_xlabel("Time")
# axes.set_ylabel("Centered Mean (normalized)")

# #Add title
# axes.set_title("MF Centered mean of x Coordinates of Particles")

# #Initialize line graphs
# axes.plot(mMFcsm1, color = 'black',label = 'n=10')
# axes.plot(mMFcsm2, color = 'red',label = 'n=20')
# axes.plot(mMFcsm3, color = 'blue',label = 'n=50')
# axes.plot(mMFcsm4, color = 'green',label = 'n=100')
# axes.plot(mMFcsm5, color = 'yellow',label = 'n=200')
# axes.plot(mMFcsm6, color = 'orange',label = 'n=500')
# axes.plot(mMFcsm7, color = 'purple',label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\MFcmean"+eb+".pdf", bbox_inches = 'tight')

# ###########Plot noisless MF centered mean 
# mMFmMF1 = (np.mean(smMF1[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(10.0*m)
# mMFmMF2 = (np.mean(smMF2[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(20.0*m)
# mMFmMF3 = (np.mean(smMF3[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(50.0*m)
# mMFmMF4 = (np.mean(smMF4[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(100.0*m)
# mMFmMF5 = (np.mean(smMF5[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(200.0*m)
# mMFmMF6 = (np.mean(smMF6[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(500.0*m)
# mMFmMF7 = (np.mean(smMF7[0,:,:],axis=1)-mMF[0,0,:])*np.sqrt(1000.0*m)

# #plot bounds
# mxm1 = np.max(np.abs(mMFmMF1))
# mxm2 = np.max(np.abs(mMFmMF2))
# mxm3 = np.max(np.abs(mMFmMF3))
# mxm4 = np.max(np.abs(mMFmMF4))
# mxm5 = np.max(np.abs(mMFmMF5))
# mxm6 = np.max(np.abs(mMFmMF6))
# mxm7 = np.max(np.abs(mMFmMF7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)

# ##Create figure and axes for MF mean x coord simulations
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlim,mxlim)
# axes.set_xlabel("Time")
# axes.set_ylabel("Centered Mean (normalized)")

# #Add title
# axes.set_title("Noiseless MF Centered MF")

# #Initialize line graphs
# axes.plot(mMFmMF1, color = 'black',label = 'n=10')
# axes.plot(mMFmMF2, color = 'red',label = 'n=20')
# axes.plot(mMFmMF3, color = 'blue',label = 'n=50')
# axes.plot(mMFmMF4, color = 'green',label = 'n=100')
# axes.plot(mMFmMF5, color = 'yellow',label = 'n=200')
# axes.plot(mMFmMF6, color = 'orange',label = 'n=500')
# axes.plot(mMFmMF7, color = 'purple',label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\MFNcmean"+eb+".pdf", bbox_inches = 'tight')
#%%


###########Plot MSE

#Get the mean MSE
mmse1 = np.mean(mse1,1)
mmse2 = np.mean(mse2,1)
mmse3 = np.mean(mse3,1)
mmse4 = np.mean(mse4,1)
mmse5 = np.mean(mse5,1)
mmse6 = np.mean(mse6,1)
mmse7 = np.mean(mse7,1)

#Get the mean MSE over time
mtmse1 = np.mean(mmse1[st:])
mtmse2 = np.mean(mmse2[st:])
mtmse3 = np.mean(mmse3[st:])
mtmse4 = np.mean(mmse4[st:])
mtmse5 = np.mean(mmse5[st:])
mtmse6 = np.mean(mmse6[st:])
mtmse7 = np.mean(mmse7[st:])
mtmse = np.array([mtmse1,mtmse2,mtmse3,mtmse4,mtmse5,mtmse6,mtmse7])

mxm1 = np.max(mmse1)
mxm2 = np.max(mmse2)
mxm3 = np.max(mmse3)
mxm4 = np.max(mmse4)
mxm5 = np.max(mmse5)
mxm6 = np.max(mmse6)
mxm7 = np.max(mmse7)
mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimred = max(mxm2,mxm4,mxm6,mxm7)
mxlimlg = max(mxm5,mxm6,mxm7)
mxtlim = max(mtmse)

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(0,mxlim)
axes.set_xlabel("Time")
axes.set_ylabel("Average MSE")

#Add title
#axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.plot(mmse1, color = 'gray', label = 'n=10', marker = '+')
axes.plot(mmse2, color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(mmse3, color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(mmse4, color = 'gray', label = 'n=100')
axes.plot(mmse5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(mmse6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(mmse7, color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig1a.eps", bbox_inches = 'tight', format = 'eps')

#%%
####Same calculation for reduced choices of n
##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimred)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average MSE")

# #Add title
# axes.set_title("Average MSE of the Particles wrt the MF Approximation")

# #Initialize line graphs
# axes.plot(mmse2, color = 'red', label = 'n=20')
# axes.plot(mmse4, color = 'green', label = 'n=100')
# axes.plot(mmse6, color = 'orange', label = 'n=500')
# axes.plot(mmse7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\MSEred"+eb+".pdf", bbox_inches = 'tight')
#%%


####Same calculation for large n
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(0,mxlimlg)
axes.set_xlabel("Time")
axes.set_ylabel("Average MSE")

#Add title
#axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.plot(mmse5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(mmse6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(mmse7, color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig1b.eps", bbox_inches = 'tight', format = 'eps')

##Index by n instead of t
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(0,mxtlim)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Average MSE")

#Add title
#axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.plot(tms, mtmse, color = 'black')

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig1c.eps", bbox_inches = 'tight')
###########Plot Graph Density

#Get the mean Graph Density
mde1 = np.mean(de1,1)
mde2 = np.mean(de2,1)
mde3 = np.mean(de3,1)
mde4 = np.mean(de4,1)
mde5 = np.mean(de5,1)
mde6 = np.mean(de6,1)
mde7 = np.mean(de7,1)

mdeMF1 = np.mean(deMF1,1)
mdeMF2 = np.mean(deMF2,1)
mdeMF3 = np.mean(deMF3,1)
mdeMF4 = np.mean(deMF4,1)
mdeMF5 = np.mean(deMF5,1)
mdeMF6 = np.mean(deMF6,1)
mdeMF7 = np.mean(deMF7,1)

mdediff1 = mdeMF1-mde1
mdediff2 = mdeMF2-mde2
mdediff3 = mdeMF3-mde3
mdediff4 = mdeMF4-mde4
mdediff5 = mdeMF5-mde5
mdediff6 = mdeMF6-mde6
mdediff7 = mdeMF7-mde7

mxm1 = np.max(mde1)
mxm2 = np.max(mde2)
mxm3 = np.max(mde3)
mxm4 = np.max(mde4)
mxm5 = np.max(mde5)
mxm6 = np.max(mde6)
mxm7 = np.max(mde7)

mxmMF1 = np.max(mdeMF1)
mxmMF2 = np.max(mdeMF2)
mxmMF3 = np.max(mdeMF3)
mxmMF4 = np.max(mdeMF4)
mxmMF5 = np.max(mdeMF5)
mxmMF6 = np.max(mdeMF6)
mxmMF7 = np.max(mdeMF7)

mxmdiff1 = np.max(np.abs(mdediff1))
mxmdiff2 = np.max(np.abs(mdediff2))
mxmdiff3 = np.max(np.abs(mdediff3))
mxmdiff4 = np.max(np.abs(mdediff4))
mxmdiff5 = np.max(np.abs(mdediff5))
mxmdiff6 = np.max(np.abs(mdediff6))
mxmdiff7 = np.max(np.abs(mdediff7))

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimdiff = max(mxmdiff1,mxmdiff2,mxmdiff3,mxmdiff4,mxmdiff5,mxmdiff6,mxmdiff7)
mxlimT = max(mxlim,mxlimMF)

#%%
# #########MF edge density plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Edge Density")

# #Add title
# axes.set_title("Average Edge Density of the MF Network")

# #Initialize line graphs
# axes.plot(mdeMF1, color = 'black', label = 'n=10')
# axes.plot(mdeMF2, color = 'red', label = 'n=20')
# axes.plot(mdeMF3, color = 'blue', label = 'n=50')
# axes.plot(mdeMF4, color = 'green', label = 'n=100')
# axes.plot(mdeMF5, color = 'yellow', label = 'n=200')
# axes.plot(mdeMF6, color = 'orange', label = 'n=500')
# axes.plot(mdeMF7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\egdedensitydiff"+eb+".pdf", bbox_inches = 'tight')

# #########Edge density plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Edge Density")

# #Add title
# axes.set_title("Average Edge Density of the Network")

# #Initialize line graphs
# axes.plot(mde1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mde2, color = 'red', label = 'n=20')
# axes.plot(mde3, color = 'blue', label = 'n=50')
# axes.plot(mde4, color = 'green', label = 'n=100')
# axes.plot(mde5, color = 'yellow', label = 'n=200')
# axes.plot(mde6, color = 'orange', label = 'n=500')
# axes.plot(mde7, color = 'purple', label = 'n=1000')
# axes.plot(mdeMF7, color = 'black', label = 'MF')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\edgedensity"+eb+".pdf", bbox_inches = 'tight')

# #########Edge density plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimdiff,mxlimdiff)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Edge Density")

# #Add title
# axes.set_title("Average Edge Density (MF-n particle)")

# #Initialize line graphs
# axes.plot(mdediff1, color = 'black', label = 'n=10')
# axes.plot(mdediff2, color = 'red', label = 'n=20')
# axes.plot(mdediff3, color = 'blue', label = 'n=50')
# axes.plot(mdediff4, color = 'green', label = 'n=100')
# axes.plot(mdediff5, color = 'yellow', label = 'n=200')
# axes.plot(mdediff6, color = 'orange', label = 'n=500')
# axes.plot(mdediff7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\egdedensitydiff"+eb+".pdf", bbox_inches = 'tight')
#%%

# ###########Plot Triangle Density

#Get the mean Triangle Density
mtde1 = np.mean(tde1,1)
mtde2 = np.mean(tde2,1)
mtde3 = np.mean(tde3,1)
mtde4 = np.mean(tde4,1)
mtde5 = np.mean(tde5,1)
mtde6 = np.mean(tde6,1)
mtde7 = np.mean(tde7,1)

mtdeMF1 = np.mean(tdeMF1,1)
mtdeMF2 = np.mean(tdeMF2,1)
mtdeMF3 = np.mean(tdeMF3,1)
mtdeMF4 = np.mean(tdeMF4,1)
mtdeMF5 = np.mean(tdeMF5,1)
mtdeMF6 = np.mean(tdeMF6,1)
mtdeMF7 = np.mean(tdeMF7,1)

mtdediff1 = mtdeMF1-mtde1
mtdediff2 = mtdeMF2-mtde2
mtdediff3 = mtdeMF3-mtde3
mtdediff4 = mtdeMF4-mtde4
mtdediff5 = mtdeMF5-mtde5
mtdediff6 = mtdeMF6-mtde6
mtdediff7 = mtdeMF7-mtde7

mtdetdiff1 = np.mean(mtdediff1[st:])
mtdetdiff2 = np.mean(mtdediff2[st:])
mtdetdiff3 = np.mean(mtdediff3[st:])
mtdetdiff4 = np.mean(mtdediff4[st:])
mtdetdiff5 = np.mean(mtdediff5[st:])
mtdetdiff6 = np.mean(mtdediff6[st:])
mtdetdiff7 = np.mean(mtdediff7[st:])
mtdetdiff = np.array([mtdetdiff1,mtdetdiff2,mtdetdiff3,mtdetdiff4,mtdetdiff5,mtdetdiff6,mtdetdiff7])

mxm1 = np.max(mtde1)
mxm2 = np.max(mtde2)
mxm3 = np.max(mtde3)
mxm4 = np.max(mtde4)
mxm5 = np.max(mtde5)
mxm6 = np.max(mtde6)
mxm7 = np.max(mtde7)

mxmMF1 = np.max(mtdeMF1)
mxmMF2 = np.max(mtdeMF2)
mxmMF3 = np.max(mtdeMF3)
mxmMF4 = np.max(mtdeMF4)
mxmMF5 = np.max(mtdeMF5)
mxmMF6 = np.max(mtdeMF6)
mxmMF7 = np.max(mtdeMF7)

mxmdiff1 = np.max(mtdediff1)
mxmdiff2 = np.max(mtdediff2)
mxmdiff3 = np.max(mtdediff3)
mxmdiff4 = np.max(mtdediff4)
mxmdiff5 = np.max(mtdediff5)
mxmdiff6 = np.max(mtdediff6)
mxmdiff7 = np.max(mtdediff7)

mndiff1 = np.min(mtdediff1)
mndiff2 = np.min(mtdediff2)
mndiff3 = np.min(mtdediff3)
mndiff4 = np.min(mtdediff4)
mndiff5 = np.min(mtdediff5)
mndiff6 = np.min(mtdediff6)
mndiff7 = np.min(mtdediff7)

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimdiff = max(mxmdiff1,mxmdiff2,mxmdiff3,mxmdiff4,mxmdiff5,mxmdiff6,mxmdiff7)
mnlimdiff = min(mndiff1,mndiff2,mndiff3,mndiff4,mndiff5,mndiff6,mndiff7)
mxlimdiffred = max(mxmdiff2,mxmdiff4,mxmdiff6,mxmdiff7)
mnlimdiffred = min(mndiff2,mndiff4,mndiff6,mndiff7)
mxlimdifflg = max(mxmdiff5,mxmdiff6,mxmdiff7)
mnlimdifflg = min(mndiff5,mndiff6,mndiff7)
mxtlimdiff = np.max(mtdetdiff)
mntlimdiff = np.min(mtdetdiff)
mxlimT = max(mxlim,mxlimMF)


#%%
# #########MF triangle density plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Triangle Density")

# #Add title
# axes.set_title("Average Triangle Density of the MF Network")

# #Initialize line graphs
# axes.plot(mtdeMF1, color = 'black', label = 'n=10')
# axes.plot(mtdeMF2, color = 'red', label = 'n=20')
# axes.plot(mtdeMF3, color = 'blue', label = 'n=50')
# axes.plot(mtdeMF4, color = 'green', label = 'n=100')
# axes.plot(mtdeMF5, color = 'yellow', label = 'n=200')
# axes.plot(mtdeMF6, color = 'orange', label = 'n=500')
# axes.plot(mtdeMF7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\triangledensityMF"+eb+".pdf", bbox_inches = 'tight')

# #########Triangle density plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Triangle Density of the Network")

# #Add title
# axes.set_title("Average Triangle Density")

# #Initialize line graphs
# axes.plot(mtde1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mtde2, color = 'red', label = 'n=20')
# axes.plot(mtde3, color = 'blue', label = 'n=50')
# axes.plot(mtde4, color = 'green', label = 'n=100')
# axes.plot(mtde5, color = 'yellow', label = 'n=200')
# axes.plot(mtde6, color = 'orange', label = 'n=500')
# axes.plot(mtde7, color = 'purple', label = 'n=1000')
# axes.plot(mtdeMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\triangledensity"+eb+".pdf", bbox_inches = 'tight')

#%%

#########MF triangle density plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(mnlimdiff,mxlimdiff)
axes.set_xlabel("Time")
axes.set_ylabel("Error")

#Add title
#axes.set_title("Average Triangle Density Error")

#Initialize line graphs
axes.plot(mtdediff1, color = 'gray', label = 'n=10', marker = '+')
axes.plot(mtdediff2, color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(mtdediff3, color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(mtdediff4, color = 'gray', label = 'n=100')
axes.plot(mtdediff5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(mtdediff6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(mtdediff7, color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left',framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3a.eps", bbox_inches = 'tight', format = 'eps')

#%%
##Same calculation for reduced n
##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(mnlimdiffred,mxlimdiffred)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Triangle Density")

# #Add title
# axes.set_title("Average Triangle Density (MF-n particle)")

# #Initialize line graphs
# axes.plot(mtdediff2, color = 'red', label = 'n=20')
# axes.plot(mtdediff4, color = 'green', label = 'n=100')
# axes.plot(mtdediff6, color = 'orange', label = 'n=500')
# axes.plot(mtdediff7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\triangledensitydiffred"+eb+".pdf", bbox_inches = 'tight')

#%%

#Same calculation for large n
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(mnlimdifflg,mxlimdifflg)
axes.set_xlabel("Time")
axes.set_ylabel("Error")

#Add title
#axes.set_title("Average Triangle Density Error")

#Initialize line graphs
axes.plot(mtdediff5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(mtdediff6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(mtdediff7, color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left',framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3b.eps", bbox_inches = 'tight', format = 'eps')

###Param by n instead of t
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(mntlimdiff,mxtlimdiff)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Error")

#Add title
#axes.set_title("Average Triangle Density Error")

#Initialize line graphs
axes.plot(tms, mtdetdiff, color = 'black')

#Add legend
#axes.legend(loc='upper left',framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3c.eps", bbox_inches = 'tight', format = 'eps')


#######Triangle Density MF vs Erdos Renyi
#Get expected Erdos Renyi triangle density
ERtde = 6*sp.special.comb(1000,3)*mdeMF7**3/(1000**3)
mx = max(np.max(mtdeMF7),np.max(ERtde))

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(0,mx)
axes.set_xlabel("Time")
axes.set_ylabel("Triangle Density")

#Add title
#axes.set_title("Average Triangle Density")

#Initialize line graphs
axes.plot(mtdeMF7, color = 'black', label = 'MF')
axes.plot(ERtde, color = 'gray', label = 'Erdos Renyi')

#Add legend
axes.legend(loc='lower left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3d.eps", bbox_inches = 'tight', format = 'eps')

#######Triangle Density MF vs Erdos Renyi
#Get expected Erdos Renyi triangle density
EGratio = ERtde/mtdeMF7
mx = np.max(EGratio)
mn = np.min(EGratio)


#%%
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(mn,mx)
# axes.set_xlabel("Time")
# axes.set_ylabel("Triangle Density Ratio")

# #Add title
# axes.set_title("Ratio of Triangle Densities: ER vs MF Network")

# #Initialize line graphs
# axes.plot(EGratio, color = 'black')

# #Save figure
# plt.savefig("plots\\triangledensityERratio"+eb+".pdf", bbox_inches = 'tight')

# ###########Plot Clustering Coefficient

# #Get the mean clustering coefficient
# mcl1 = np.mean(cl1,1)
# mcl2 = np.mean(cl2,1)
# mcl3 = np.mean(cl3,1)
# mcl4 = np.mean(cl4,1)
# mcl5 = np.mean(cl5,1)
# mcl6 = np.mean(cl6,1)
# mcl7 = np.mean(cl7,1)

# mclMF1 = np.mean(clMF1,1)
# mclMF2 = np.mean(clMF2,1)
# mclMF3 = np.mean(clMF3,1)
# mclMF4 = np.mean(clMF4,1)
# mclMF5 = np.mean(clMF5,1)
# mclMF6 = np.mean(clMF6,1)
# mclMF7 = np.mean(clMF7,1)

# mcldiff1 = mclMF1-mcl1
# mcldiff2 = mclMF2-mcl2
# mcldiff3 = mclMF3-mcl3
# mcldiff4 = mclMF4-mcl4
# mcldiff5 = mclMF5-mcl5
# mcldiff6 = mclMF6-mcl6
# mcldiff7 = mclMF7-mcl7

# mxm1 = np.max(mcl1)
# mxm2 = np.max(mcl2)
# mxm3 = np.max(mcl3)
# mxm4 = np.max(mcl4)
# mxm5 = np.max(mcl5)
# mxm6 = np.max(mcl6)
# mxm7 = np.max(mcl7)

# mxmMF1 = np.max(mclMF1)
# mxmMF2 = np.max(mclMF2)
# mxmMF3 = np.max(mclMF3)
# mxmMF4 = np.max(mclMF4)
# mxmMF5 = np.max(mclMF5)
# mxmMF6 = np.max(mclMF6)
# mxmMF7 = np.max(mclMF7)

# mxmdiff1 = np.max(np.abs(mcldiff1))
# mxmdiff2 = np.max(np.abs(mcldiff2))
# mxmdiff3 = np.max(np.abs(mcldiff3))
# mxmdiff4 = np.max(np.abs(mcldiff4))
# mxmdiff5 = np.max(np.abs(mcldiff5))
# mxmdiff6 = np.max(np.abs(mcldiff6))
# mxmdiff7 = np.max(np.abs(mcldiff7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
# mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
# mxlimdiff = max(mxmdiff1,mxmdiff2,mxmdiff3,mxmdiff4,mxmdiff5,mxmdiff6,mxmdiff7)
# mxlimT = max(mxlim,mxlimMF)

# #########MF clustering coefficient plots
# ##Create figure and axes for plot
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Clustering Coefficient")

# #Add title
# axes.set_title("Average Clustering Coefficient of the MF Network")

# #Initialize line graphs
# axes.plot(mclMF1, color = 'black', label = 'n=10')
# axes.plot(mclMF2, color = 'red', label = 'n=20')
# axes.plot(mclMF3, color = 'blue', label = 'n=50')
# axes.plot(mclMF4, color = 'green', label = 'n=100')
# axes.plot(mclMF5, color = 'yellow', label = 'n=200')
# axes.plot(mclMF6, color = 'orange', label = 'n=500')
# axes.plot(mclMF7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\clusteringMF"+eb+".pdf", bbox_inches = 'tight')

# #########clustering coefficient plots
# ##Create figure and axes for plot
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Clustering Coefficient")

# #Add title
# axes.set_title("Average Clustering Coefficient of the Network")

# #Initialize line graphs
# axes.plot(mcl1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mcl2, color = 'red', label = 'n=20')
# axes.plot(mcl3, color = 'blue', label = 'n=50')
# axes.plot(mcl4, color = 'green', label = 'n=100')
# axes.plot(mcl5, color = 'yellow', label = 'n=200')
# axes.plot(mcl6, color = 'orange', label = 'n=500')
# axes.plot(mcl7, color = 'purple', label = 'n=1000')
# axes.plot(mclMF7, color = 'black', label = 'MF')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\clustering"+eb+".pdf", bbox_inches = 'tight')

# #########clustering coefficient plots
# ##Create figure and axes for plot
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimdiff,mxlimdiff)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Clustering Coefficient")

# #Add title
# axes.set_title("Average Clustering Coefficient (MF-n particle)")

# #Initialize line graphs
# axes.plot(mcldiff1, color = 'black', label = 'n=10')
# axes.plot(mcldiff2, color = 'red', label = 'n=20')
# axes.plot(mcldiff3, color = 'blue', label = 'n=50')
# axes.plot(mcldiff4, color = 'green', label = 'n=100')
# axes.plot(mcldiff5, color = 'yellow', label = 'n=200')
# axes.plot(mcldiff6, color = 'orange', label = 'n=500')
# axes.plot(mcldiff7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\clusteringdiff"+eb+".pdf", bbox_inches = 'tight')

# #######Clustering Coefficient MF vs Erdos Renyi

# #Get asymptotic Erdos Renyi clustering coefficient
# ERcl = mdeMF7
# mx = max(np.max(mclMF7),np.max(ERcl))

# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mx)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average/Asymptotic Clustering Coefficient")

# #Add title
# axes.set_title("Average/Asymptotic Clustering Coefficient of MF Network vs Erdos Renyi")

# #Initialize line graphs
# axes.plot(mclMF7, color = 'black', label = 'MF')
# axes.plot(ERcl, color = 'red', label = 'Erdos Renyi')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\clusteringER"+eb+".pdf", bbox_inches = 'tight')

# #######Clustering Coefficient MF vs Erdos Renyi
# #Get expected Erdos Renyi triangle density
# CCratio = ERcl/mclMF7
# mx = np.max(CCratio)
# mn = np.min(CCratio)

# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(mn,mx)
# axes.set_xlabel("Time")
# axes.set_ylabel("Clustering Coefficient Ratio")

# #Add title
# axes.set_title("Ratio of Clustering Coefficients: ER vs MF Network")

# #Initialize line graphs
# axes.plot(CCratio, color = 'black')

# #Save figure
# plt.savefig("plots\\clusteringERratio"+eb+".pdf", bbox_inches = 'tight')

# #########Largest eigenvalue plot

# #Get the largest eigenvalue bounds
# mle1 = np.mean(le1,1)/10
# mle2 = np.mean(le2,1)/20
# mle3 = np.mean(le3,1)/50
# mle4 = np.mean(le4,1)/100
# mle5 = np.mean(le5,1)/200
# mle6 = np.mean(le6,1)/500
# mle7 = np.mean(le7,1)/1000

# mleMF1 = np.mean(leMF1,1)/10
# mleMF2 = np.mean(leMF2,1)/20
# mleMF3 = np.mean(leMF3,1)/50
# mleMF4 = np.mean(leMF4,1)/100
# mleMF5 = np.mean(leMF5,1)/200
# mleMF6 = np.mean(leMF6,1)/500
# mleMF7 = np.mean(leMF7,1)/1000

# mlediff1 = mleMF1-mle1
# mlediff2 = mleMF2-mle2
# mlediff3 = mleMF3-mle3
# mlediff4 = mleMF4-mle4
# mlediff5 = mleMF5-mle5
# mlediff6 = mleMF6-mle6
# mlediff7 = mleMF7-mle7

# mxm1 = np.max(np.abs(mle1))
# mxm2 = np.max(np.abs(mle2))
# mxm3 = np.max(np.abs(mle3))
# mxm4 = np.max(np.abs(mle4))
# mxm5 = np.max(np.abs(mle5))
# mxm6 = np.max(np.abs(mle6))
# mxm7 = np.max(np.abs(mle7))

# mxmMF1 = np.max(np.abs(mleMF1))
# mxmMF2 = np.max(np.abs(mleMF2))
# mxmMF3 = np.max(np.abs(mleMF3))
# mxmMF4 = np.max(np.abs(mleMF4))
# mxmMF5 = np.max(np.abs(mleMF5))
# mxmMF6 = np.max(np.abs(mleMF6))
# mxmMF7 = np.max(np.abs(mleMF7))

# mxmdiff1 = np.max(np.abs(mlediff1))
# mxmdiff2 = np.max(np.abs(mlediff2))
# mxmdiff3 = np.max(np.abs(mlediff3))
# mxmdiff4 = np.max(np.abs(mlediff4))
# mxmdiff5 = np.max(np.abs(mlediff5))
# mxmdiff6 = np.max(np.abs(mlediff6))
# mxmdiff7 = np.max(np.abs(mlediff7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
# mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
# mxlimdiff = max(mxmdiff1,mxmdiff2,mxmdiff3,mxmdiff4,mxmdiff5,mxmdiff6,mxmdiff7)
# mxlimT = max(mxlim,mxlimMF)

# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Largest Eigenvalue")

# #Add title
# axes.set_title("Average Largest Eigenvalue of the MF Network")

# #Initialize line graphs
# axes.plot(mleMF1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mleMF2, color = 'red', label = 'n=20')
# axes.plot(mleMF3, color = 'blue', label = 'n=50')
# axes.plot(mleMF4, color = 'green', label = 'n=100')
# axes.plot(mleMF5, color = 'yellow', label = 'n=200')
# axes.plot(mleMF6, color = 'orange', label = 'n=500')
# axes.plot(mleMF7, color = 'purple', label = 'n=1000')
# #axes.plot(mtdeMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\leigMF"+eb+".pdf", bbox_inches = 'tight')

# #########Largest eigenvalue plots

# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Largest Eigenvalue")

# #Add title
# axes.set_title("Average Largest Eigenvalue of the Network")

# #Initialize line graphs
# axes.plot(mle1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mle2, color = 'red', label = 'n=20')
# axes.plot(mle3, color = 'blue', label = 'n=50')
# axes.plot(mle4, color = 'green', label = 'n=100')
# axes.plot(mle5, color = 'yellow', label = 'n=200')
# axes.plot(mle6, color = 'orange', label = 'n=500')
# axes.plot(mle7, color = 'purple', label = 'n=1000')
# axes.plot(mleMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\leig"+eb+".pdf", bbox_inches = 'tight')

# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimdiff,mxlimdiff)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Largest Eigenvalue")

# #Add title
# axes.set_title("Average Largest Eigenvalue (MF-n particle)")

# #Initialize line graphs
# axes.plot(mlediff1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mlediff2, color = 'red', label = 'n=20')
# axes.plot(mlediff3, color = 'blue', label = 'n=50')
# axes.plot(mlediff4, color = 'green', label = 'n=100')
# axes.plot(mlediff5, color = 'yellow', label = 'n=200')
# axes.plot(mlediff6, color = 'orange', label = 'n=500')
# axes.plot(mlediff7, color = 'purple', label = 'n=1000')
# #axes.plot(mtdeMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\leigdiff"+eb+".pdf", bbox_inches = 'tight')

#%%

#########Second Largest eigenvalue plot

#Get the second largest eigenvalue bounds
msle1 = np.mean(sle1,1)/10
msle2 = np.mean(sle2,1)/20
msle3 = np.mean(sle3,1)/50
msle4 = np.mean(sle4,1)/100
msle5 = np.mean(sle5,1)/200
msle6 = np.mean(sle6,1)/500
msle7 = np.mean(sle7,1)/1000

msleMF1 = np.mean(sleMF1,1)/10
msleMF2 = np.mean(sleMF2,1)/20
msleMF3 = np.mean(sleMF3,1)/50
msleMF4 = np.mean(sleMF4,1)/100
msleMF5 = np.mean(sleMF5,1)/200
msleMF6 = np.mean(sleMF6,1)/500
msleMF7 = np.mean(sleMF7,1)/1000

mslediff1 = msleMF1-msle1
mslediff2 = msleMF2-msle2
mslediff3 = msleMF3-msle3
mslediff4 = msleMF4-msle4
mslediff5 = msleMF5-msle5
mslediff6 = msleMF6-msle6
mslediff7 = msleMF7-msle7

msletdiff1 = np.mean(mslediff1)
msletdiff2 = np.mean(mslediff2)
msletdiff3 = np.mean(mslediff3)
msletdiff4 = np.mean(mslediff4)
msletdiff5 = np.mean(mslediff5)
msletdiff6 = np.mean(mslediff6)
msletdiff7 = np.mean(mslediff7)
msletdiff = np.array([msletdiff1,msletdiff2,msletdiff3,msletdiff4,msletdiff5,msletdiff6,msletdiff7])

mxm1 = np.max(np.abs(msle1))
mxm2 = np.max(np.abs(msle2))
mxm3 = np.max(np.abs(msle3))
mxm4 = np.max(np.abs(msle4))
mxm5 = np.max(np.abs(msle5))
mxm6 = np.max(np.abs(msle6))
mxm7 = np.max(np.abs(msle7))

mxmMF1 = np.max(np.abs(msleMF1))
mxmMF2 = np.max(np.abs(msleMF2))
mxmMF3 = np.max(np.abs(msleMF3))
mxmMF4 = np.max(np.abs(msleMF4))
mxmMF5 = np.max(np.abs(msleMF5))
mxmMF6 = np.max(np.abs(msleMF6))
mxmMF7 = np.max(np.abs(msleMF7))

mxmdiff1 = np.max(mslediff1)
mxmdiff2 = np.max(mslediff2)
mxmdiff3 = np.max(mslediff3)
mxmdiff4 = np.max(mslediff4)
mxmdiff5 = np.max(mslediff5)
mxmdiff6 = np.max(mslediff6)
mxmdiff7 = np.max(mslediff7)

mndiff1 = np.min(mslediff1)
mndiff2 = np.min(mslediff2)
mndiff3 = np.min(mslediff3)
mndiff4 = np.min(mslediff4)
mndiff5 = np.min(mslediff5)
mndiff6 = np.min(mslediff6)
mndiff7 = np.min(mslediff7)

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimdiff = max(mxmdiff1,mxmdiff2,mxmdiff3,mxmdiff4,mxmdiff5,mxmdiff6,mxmdiff7)
mnlimdiff = min(mndiff1,mndiff2,mndiff3,mndiff4,mndiff5,mndiff6,mndiff7)
mxlimdiffred = max(mxmdiff2,mxmdiff4,mxmdiff6,mxmdiff7)
mxlimdifflg = max(mxmdiff5,mxmdiff6,mxmdiff7)
mnlimdiffred = min(mndiff2,mndiff4,mndiff6,mndiff7)
mnlimdifflg = min(mndiff5,mndiff6,mndiff7)
mxlimtdiff = np.max(msletdiff)
mnlimtdiff = np.min(msletdiff)
mxlimT = max(mxlim,mxlimMF)

#%%
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0, mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Second Largest Eigenvalue")

# #Add title
# axes.set_title("Average Second Largest Eigenvalue of the MF Network")

# #Initialize line graphs
# axes.plot(msleMF1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(msleMF2, color = 'red', label = 'n=20')
# axes.plot(msleMF3, color = 'blue', label = 'n=50')
# axes.plot(msleMF4, color = 'green', label = 'n=100')
# axes.plot(msleMF5, color = 'yellow', label = 'n=200')
# axes.plot(msleMF6, color = 'orange', label = 'n=500')
# axes.plot(msleMF7, color = 'purple', label = 'n=1000')
# #axes.plot(mtdeMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\sleigMF"+eb+".pdf", bbox_inches = 'tight')

# #########Second Largest eigenvalue plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Second Largest Eigenvalue")

# #Add title
# axes.set_title("Average Second Largest Eigenvalue of the Network")

# #Initialize line graphs
# axes.plot(msle1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(msle2, color = 'red', label = 'n=20')
# axes.plot(msle3, color = 'blue', label = 'n=50')
# axes.plot(msle4, color = 'green', label = 'n=100')
# axes.plot(msle5, color = 'yellow', label = 'n=200')
# axes.plot(msle6, color = 'orange', label = 'n=500')
# axes.plot(msle7, color = 'purple', label = 'n=1000')
# axes.plot(msleMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\sleig"+eb+".pdf", bbox_inches = 'tight')

#%%

#########Second Largest eigenvalue plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(mnlimdiff,mxlimdiff)
axes.set_xlabel("Time")
axes.set_ylabel("Eigenvalue Error")

#Add title
#axes.set_title("Average Second Largest Eigenvalue Error")

#Initialize line graphs
axes.plot(mslediff1, color = 'gray', label = 'n=10', marker = '+')
axes.plot(mslediff2, color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(mslediff3, color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(mslediff4, color = 'gray', label = 'n=100')
axes.plot(mslediff5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(mslediff6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(mslediff7, color = 'black', label = 'n=1000')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig4a.eps", bbox_inches = 'tight', format = 'eps')

#%%
###Same calculation, reduced n
##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(mnlimdiffred,mxlimdiffred)
# axes.set_xlabel("Time")
# axes.set_ylabel("Eigenvalue Error")

# #Add title
# axes.set_title("Average Second Largest Eigenvalue Error")

# #Initialize line graphs
# axes.plot(mslediff2, color = 'red', label = 'n=20')
# axes.plot(mslediff4, color = 'green', label = 'n=100')
# axes.plot(mslediff6, color = 'orange', label = 'n=500')
# axes.plot(mslediff7, color = 'purple', label = 'n=1000')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# #axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\sleigdiffred"+eb+".pdf", bbox_inches = 'tight')

#%%

###Same calculation, large n
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(mnlimdifflg,mxlimdifflg)
axes.set_xlabel("Time")
axes.set_ylabel("Eigenvalue Error")

#Add title
#axes.set_title("Average Second Largest Eigenvalue Error")

#Initialize line graphs
axes.plot(mslediff5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(mslediff6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(mslediff7, color = 'black', label = 'n=1000')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig4b.eps", bbox_inches = 'tight', format = 'eps')

###Param by n instead of t
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(mnlimtdiff,mxlimtdiff)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Eigenvalue Error")

#Add title
#axes.set_title("Average Second Largest Eigenvalue Error")

#Initialize line graphs
axes.plot(tms,msletdiff, color = 'black')

#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig4c.eps", bbox_inches = 'tight', format = 'eps')

#%%
# #########Third Largest eigenvalue plot

# #Get the third largest eigenvalue bounds
# mtle1 = np.mean(tle1,1)/10
# mtle2 = np.mean(tle2,1)/20
# mtle3 = np.mean(tle3,1)/50
# mtle4 = np.mean(tle4,1)/100
# mtle5 = np.mean(tle5,1)/200
# mtle6 = np.mean(tle6,1)/500
# mtle7 = np.mean(tle7,1)/1000

# mtleMF1 = np.mean(tleMF1,1)/10
# mtleMF2 = np.mean(tleMF2,1)/20
# mtleMF3 = np.mean(tleMF3,1)/50
# mtleMF4 = np.mean(tleMF4,1)/100
# mtleMF5 = np.mean(tleMF5,1)/200
# mtleMF6 = np.mean(tleMF6,1)/500
# mtleMF7 = np.mean(tleMF7,1)/1000

# mtlediff1 = mtleMF1-mtle1
# mtlediff2 = mtleMF2-mtle2
# mtlediff3 = mtleMF3-mtle3
# mtlediff4 = mtleMF4-mtle4
# mtlediff5 = mtleMF5-mtle5
# mtlediff6 = mtleMF6-mtle6
# mtlediff7 = mtleMF7-mtle7

# mxm1 = np.max(np.abs(mtle1))
# mxm2 = np.max(np.abs(mtle2))
# mxm3 = np.max(np.abs(mtle3))
# mxm4 = np.max(np.abs(mtle4))
# mxm5 = np.max(np.abs(mtle5))
# mxm6 = np.max(np.abs(mtle6))
# mxm7 = np.max(np.abs(mtle7))

# mxmMF1 = np.max(np.abs(mtleMF1))
# mxmMF2 = np.max(np.abs(mtleMF2))
# mxmMF3 = np.max(np.abs(mtleMF3))
# mxmMF4 = np.max(np.abs(mtleMF4))
# mxmMF5 = np.max(np.abs(mtleMF5))
# mxmMF6 = np.max(np.abs(mtleMF6))
# mxmMF7 = np.max(np.abs(mtleMF7))

# mxmdiff1 = np.max(np.abs(mtlediff1))
# mxmdiff2 = np.max(np.abs(mtlediff2))
# mxmdiff3 = np.max(np.abs(mtlediff3))
# mxmdiff4 = np.max(np.abs(mtlediff4))
# mxmdiff5 = np.max(np.abs(mtlediff5))
# mxmdiff6 = np.max(np.abs(mtlediff6))
# mxmdiff7 = np.max(np.abs(mtlediff7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
# mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
# mxlimdiff = max(mxmdiff1,mxmdiff2,mxmdiff3,mxmdiff4,mxmdiff5,mxmdiff6,mxmdiff7)
# mxlimT = max(mxlim,mxlimMF)

# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0, mxlimMF)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Third Largest Eigenvalue")

# #Add title
# axes.set_title("Average Third Largest Eigenvalue of the MF Network")

# #Initialize line graphs
# axes.plot(mtleMF1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mtleMF2, color = 'red', label = 'n=20')
# axes.plot(mtleMF3, color = 'blue', label = 'n=50')
# axes.plot(mtleMF4, color = 'green', label = 'n=100')
# axes.plot(mtleMF5, color = 'yellow', label = 'n=200')
# axes.plot(mtleMF6, color = 'orange', label = 'n=500')
# axes.plot(mtleMF7, color = 'purple', label = 'n=1000')
# #axes.plot(mtdeMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\tleigMF"+eb+".pdf", bbox_inches = 'tight')

# #########Third Largest eigenvalue plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimT)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Third Largest Eigenvalue")

# #Add title
# axes.set_title("Average Third Largest Eigenvalue of the Network")

# #Initialize line graphs
# axes.plot(mtle1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mtle2, color = 'red', label = 'n=20')
# axes.plot(mtle3, color = 'blue', label = 'n=50')
# axes.plot(mtle4, color = 'green', label = 'n=100')
# axes.plot(mtle5, color = 'yellow', label = 'n=200')
# axes.plot(mtle6, color = 'orange', label = 'n=500')
# axes.plot(mtle7, color = 'purple', label = 'n=1000')
# axes.plot(mtleMF7, color = 'black', label = 'MF')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\tleig"+eb+".pdf", bbox_inches = 'tight')

# #########Third Largest eigenvalue plots
# ##Create figure and axes for animation
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlimdiff,mxlimdiff)
# axes.set_xlabel("Time")
# axes.set_ylabel("Average Third Largest Eigenvalue")

# #Add title
# axes.set_title("Average Third Largest Eigenvalue (MF-n particle)")

# #Initialize line graphs
# axes.plot(mtlediff1, color = 'black', label = 'n=10', linestyle = '--')
# axes.plot(mtlediff2, color = 'red', label = 'n=20')
# axes.plot(mtlediff3, color = 'blue', label = 'n=50')
# axes.plot(mtlediff4, color = 'green', label = 'n=100')
# axes.plot(mtlediff5, color = 'yellow', label = 'n=200')
# axes.plot(mtlediff6, color = 'orange', label = 'n=500')
# axes.plot(mtlediff7, color = 'purple', label = 'n=1000')
# #axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\tleigdiff"+eb+".pdf", bbox_inches = 'tight')

#%%


###########Plot Density of the Symmetric Difference Graph

#Get the mean density
msd1 = np.mean(sd1,1)/(10.0*9.0/2.0)
msd2 = np.mean(sd2,1)/(20.0*19.0/2.0)
msd3 = np.mean(sd3,1)/(50.0*49.0/2.0)
msd4 = np.mean(sd4,1)/(100.0*99.0/2.0)
msd5 = np.mean(sd5,1)/(200.0*199.0/2.0)
msd6 = np.mean(sd6,1)/(500.0*499.0/2.0)
msd7 = np.mean(sd7,1)/(1000.0*999.0/2.0)

msdt1 = np.mean(msd1[st:])
msdt2 = np.mean(msd2[st:])
msdt3 = np.mean(msd3[st:])
msdt4 = np.mean(msd4[st:])
msdt5 = np.mean(msd5[st:])
msdt6 = np.mean(msd6[st:])
msdt7 = np.mean(msd7[st:])
msdt = np.array([msdt1,msdt2,msdt3,msdt4,msdt5,msdt6,msdt7])

mxm1 = np.max(msd1)
mxm2 = np.max(msd2)
mxm3 = np.max(msd3)
mxm4 = np.max(msd4)
mxm5 = np.max(msd5)
mxm6 = np.max(msd6)
mxm7 = np.max(msd7)

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimred = max(mxm2,mxm4,mxm6,mxm7)
mxlimlg = max(mxm5,mxm6,mxm7)
mxtlim = np.max(msdt)

#########MF clustering coefficient plots
##Create figure and axes for plot
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(0,mxlim)
axes.set_xlabel("Time")
axes.set_ylabel("Density")

#Add title
#axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.plot(msd1, color = 'gray', label = 'n=10', marker = '+')
axes.plot(msd2, color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(msd3, color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(msd4, color = 'gray', label = 'n=100')
axes.plot(msd5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(msd6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(msd7, color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig2a.eps", bbox_inches = 'tight', format = 'eps')

#%%
###Same calculation, reduced n
##Create figure and axes for plot
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(0,mxlimred)
# axes.set_xlabel("Time")
# axes.set_ylabel("Density")

# #Add title
# axes.set_title("Average Density of the n-particle/MF Symmetric Difference Network")

# #Initialize line graphs
# axes.plot(msd2, color = 'red', label = 'n=20')
# axes.plot(msd4, color = 'green', label = 'n=100')
# axes.plot(msd6, color = 'orange', label = 'n=500')
# axes.plot(msd7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\symmdiffdensered"+eb+".pdf", bbox_inches = 'tight')

#%%

###Same calculation, large n
##Create figure and axes for plot
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(0,mxlimlg)
axes.set_xlabel("Time")
axes.set_ylabel("Density")

#Add title
#axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.plot(msd5, color = 'black', label = 'n=200', linestyle = ':')
axes.plot(msd6, color = 'black', label = 'n=500', linestyle = '--')
axes.plot(msd7, color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='lower right', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig2b.eps", bbox_inches = 'tight', format = 'eps')

###Param by n instead of t
##Create figure and axes for plot
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(0,mxtlim)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Density")

#Add title
#axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.plot(tms,msdt, color = 'black')

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig2c.eps", bbox_inches = 'tight', format = 'eps')

###########Symmetric diff density for different N

#Filebases
Nfilebase1 = "runwv2N4kn1000" #N = 4000
Nfilebase2 = "runwv2n1000" #N=3000
Nfilebase3 = "runwn1000" #N=2000
#Nfilebase4 = "runwv2n1kn1000" #N = 1000
Nfilebase5 = "runwv2n5Hn1000" #N = 500

#Coupled particle objects
Ncp1 = sm.CoupledParticle()
Ncp2 = sm.CoupledParticle()
Ncp3 = sm.CoupledParticle()
#Ncp4 = sm.CoupledParticle()
Ncp5 = sm.CoupledParticle()

#load stored data (if filebase does not exist, creates an empty directory)
Nssl1 = sl.SimulateSaveLoad(Nfilebase1,op=1,cp=Ncp1)
Nssl2 = sl.SimulateSaveLoad(Nfilebase2,op=1,cp=Ncp2)
Nssl3 = sl.SimulateSaveLoad(Nfilebase3,op=1,cp=Ncp3)
#Nssl4 = sl.SimulateSaveLoad(Nfilebase4,op=1,cp=Ncp4)
Nssl5 = sl.SimulateSaveLoad(Nfilebase5,op=1,cp=Ncp5)

#Load A statistics
Nam1 = Nssl1.loadmCoupledAStatistics()
Nam2 = Nssl2.loadmCoupledAStatistics()
Nam3 = Nssl3.loadmCoupledAStatistics()
#Nam4 = Nssl4.loadmCoupledAStatistics()
Nam5 = Nssl5.loadmCoupledAStatistics()

#Get number of edges in the symmetric difference graph
#load the number of edges in the symmetric difference graph
Nsd1 = Nam1[10]/2
Nsd2 = Nam2[10]/2
Nsd3 = Nam3[10]/2
#Nsd4 = Nam4[10]/2
Nsd5 = Nam5[10]/2

#Get the mean density of the symmetric difference graph
Nmsd1 = np.mean(Nsd1,1)/(1000.0*999.0/2.0)
Nmsd2 = np.mean(Nsd2,1)/(1000.0*999.0/2.0)
Nmsd3 = np.mean(Nsd3,1)/(1000.0*999.0/2.0)
#Nmsd4 = np.mean(Nsd4,1)/(1000.0*999.0/2.0)
Nmsd5 = np.mean(Nsd5,1)/(1000.0*999.0/2.0)

Nmxm1 = np.max(Nmsd1)
Nmxm2 = np.max(Nmsd2)
Nmxm3 = np.max(Nmsd3)
#Nmxm4 = np.max(Nmsd4)
Nmxm5 = np.max(Nmsd5)

Nmxlim = max(Nmxm5,Nmxm3,Nmxm2,Nmxm1)

fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(0,Nmxlim)
axes.set_xlabel("Time")
axes.set_ylabel("Density")

#Add title
axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.plot(Nmsd1, color = 'black', label = 'N=4000', linestyle = ':')
axes.plot(Nmsd2, color = 'black', label = 'N=3000', linestyle = '--')
axes.plot(Nmsd3, color = 'black', label = 'N=2000')

#Add legend
axes.legend(loc='lower right', framealpha=0.1)

#Save figure
plt.savefig("plots\\d1symmdiffdenseNs.pdf", bbox_inches = 'tight')


#%%
###########Plot Average Excess Edge Density in MF

# #Get the mean density
# mMFmn1 = np.mean(MFmn1,1)/(10.0*9.0/2.0)
# mMFmn2 = np.mean(MFmn2,1)/(20.0*19.0/2.0)
# mMFmn3 = np.mean(MFmn3,1)/(50.0*49.0/2.0)
# mMFmn4 = np.mean(MFmn4,1)/(100.0*99.0/2.0)
# mMFmn5 = np.mean(MFmn5,1)/(200.0*199.0/2.0)
# mMFmn6 = np.mean(MFmn6,1)/(500.0*499.0/2.0)
# mMFmn7 = np.mean(MFmn7,1)/(1000.0*999.0/2.0)

# mxm1 = np.max(np.abs(mMFmn1))
# mxm2 = np.max(np.abs(mMFmn2))
# mxm3 = np.max(np.abs(mMFmn3))
# mxm4 = np.max(np.abs(mMFmn4))
# mxm5 = np.max(np.abs(mMFmn5))
# mxm6 = np.max(np.abs(mMFmn6))
# mxm7 = np.max(np.abs(mMFmn7))

# mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)

# #########MF average excess edge density
# ##Create figure and axes for plot
# fig, axes = plt.subplots()

# #Create axes
# axes.set_xlim(0,T+1)
# axes.set_ylim(-mxlim,mxlim)
# axes.set_xlabel("Time")
# axes.set_ylabel("Excess Edge Density")

# #Add title
# axes.set_title("Average Excess Edge Density in the MF Network")

# #Initialize line graphs
# axes.plot(mMFmn1, color = 'black', label = 'n=10')
# axes.plot(mMFmn2, color = 'red', label = 'n=20')
# axes.plot(mMFmn3, color = 'blue', label = 'n=50')
# axes.plot(mMFmn4, color = 'green', label = 'n=100')
# axes.plot(mMFmn5, color = 'yellow', label = 'n=200')
# axes.plot(mMFmn6, color = 'orange', label = 'n=500')
# axes.plot(mMFmn7, color = 'purple', label = 'n=1000')

# #Add legend
# axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# #Save figure
# plt.savefig("plots\\MFexcessedges"+eb+".pdf", bbox_inches = 'tight')
