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
import scipy.stats as stats

##################Load statistics
filebase1 = "runwnv210"
filebase2 = "runwnv220"
filebase3 = "runwnv250"
filebase4 = "runwnv2100"
filebase5 = "runwnv2200"
filebase6 = "runwnv2500"
filebase7 = "runwnv21000"

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

############Plot Means

#ml1=0
msm1 = np.mean(smp1[0,:,:],axis=1)*np.sqrt(10.0*100.0)
msm2 = np.mean(smp2[0,:,:],axis=1)*np.sqrt(20.0*100.0)
msm3 = np.mean(smp3[0,:,:],axis=1)*np.sqrt(50.0*100.0)
msm4 = np.mean(smp4[0,:,:],axis=1)*np.sqrt(100.0*100.0)
msm5 = np.mean(smp5[0,:,:],axis=1)*np.sqrt(200.0*100.0)
msm6 = np.mean(smp6[0,:,:],axis=1)*np.sqrt(500.0*100.0)
msm7 = np.mean(smp7[0,:,:],axis=1)*np.sqrt(1000.0*100.0)
msmMF1 = np.mean(smMF1[0,:,:],axis=1)*np.sqrt(10.0*100.0)
msmMF2 = np.mean(smMF2[0,:,:],axis=1)*np.sqrt(20.0*100.0)
msmMF3 = np.mean(smMF3[0,:,:],axis=1)*np.sqrt(50.0*100.0)
msmMF4 = np.mean(smMF4[0,:,:],axis=1)*np.sqrt(100.0*100.0)
msmMF5 = np.mean(smMF5[0,:,:],axis=1)*np.sqrt(200.0*100.0)
msmMF6 = np.mean(smMF6[0,:,:],axis=1)*np.sqrt(500.0*100.0)
msmMF7 = np.mean(smMF7[0,:,:],axis=1)*np.sqrt(1000.0*100.0)

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

##Create figure and axes for MF mean x coord simulations
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(-mxlimMF,mxlimMF)
axes.set_xlabel("Time")
axes.set_ylabel("Mean x coordinate*sqrt(nm)")

#Add title
axes.set_title("Mean (sqrt(nm) normalized) x coordinates of MF particles")

#Initialize line graphs
axes.plot(msmMF1, color = 'black',label = 'n=10')
axes.plot(msmMF2, color = 'red',label = 'n=20')
axes.plot(msmMF3, color = 'blue',label = 'n=50')
axes.plot(msmMF4, color = 'green',label = 'n=100')
axes.plot(msmMF5, color = 'yellow',label = 'n=200')
axes.plot(msmMF6, color = 'orange',label = 'n=500')
axes.plot(msmMF7, color = 'purple',label = 'n=1000')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

##Create figure and axes for MF mean x coord simulations
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(-mxlimT,mxlimT)
axes.set_xlabel("Time")
axes.set_ylabel("Mean x coordinate*sqrt(nm)")

#Add title
axes.set_title("Mean (sqrt(nm) normalized) x coordinates particles")

#Initialize line graphs
axes.plot(msm1, color = 'black', linestyle = '--', label = 'n=10')
axes.plot(msm2, color = 'red',label = 'n=20')
axes.plot(msm3, color = 'blue',label = 'n=50')
axes.plot(msm4, color = 'green',label = 'n=100')
axes.plot(msm5, color = 'yellow',label = 'n=200')
axes.plot(msm6, color = 'orange',label = 'n=500')
axes.plot(msm7, color = 'purple', label = 'n=1000')
axes.plot(msmMF7, color = 'black', label = 'MF (n=1000)')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
###########Plot MSE

#Get the mean MSE
mmse1 = np.mean(mse1,1)
mmse2 = np.mean(mse2,1)
mmse3 = np.mean(mse3,1)
mmse4 = np.mean(mse4,1)
mmse5 = np.mean(mse5,1)
mmse6 = np.mean(mse6,1)
mmse7 = np.mean(mse7,1)

mxm1 = np.max(mmse1)
mxm2 = np.max(mmse2)
mxm3 = np.max(mmse3)
mxm4 = np.max(mmse4)
mxm5 = np.max(mmse5)
mxm6 = np.max(mmse6)
mxm7 = np.max(mmse7)
mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlim)
axes.set_xlabel("Time")
axes.set_ylabel("Average MSE")

#Add title
axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.plot(mmse1, color = 'black', label = 'n=10')
axes.plot(mmse2, color = 'red', label = 'n=20')
axes.plot(mmse3, color = 'blue', label = 'n=50')
axes.plot(mmse4, color = 'green', label = 'n=100')
axes.plot(mmse5, color = 'yellow', label = 'n=200')
axes.plot(mmse6, color = 'orange', label = 'n=500')
axes.plot(mmse7, color = 'purple', label = 'n=1000')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxlimMF)

#########MF graph density plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimMF)
axes.set_xlabel("Time")
axes.set_ylabel("Average Graph Density")

#Add title
axes.set_title("Average Graph Density of the MF Simulation")

#Initialize line graphs
axes.plot(mdeMF1, color = 'black', label = 'n=10')
axes.plot(mdeMF2, color = 'red', label = 'n=20')
axes.plot(mdeMF3, color = 'blue', label = 'n=50')
axes.plot(mdeMF4, color = 'green', label = 'n=100')
axes.plot(mdeMF5, color = 'yellow', label = 'n=200')
axes.plot(mdeMF6, color = 'orange', label = 'n=500')
axes.plot(mdeMF7, color = 'purple', label = 'n=1000')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Triangle density plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimT)
axes.set_xlabel("Time")
axes.set_ylabel("Average Graph Density")

#Add title
axes.set_title("Average Graph Density")

#Initialize line graphs
axes.plot(mde1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(mde2, color = 'red', label = 'n=20')
axes.plot(mde3, color = 'blue', label = 'n=50')
axes.plot(mde4, color = 'green', label = 'n=100')
axes.plot(mde5, color = 'yellow', label = 'n=200')
axes.plot(mde6, color = 'orange', label = 'n=500')
axes.plot(mde7, color = 'purple', label = 'n=1000')
axes.plot(mde7, color = 'black', label = 'MF')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

###########Plot Triangle Density

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

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxlimMF)

#########MF triangle density plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimMF)
axes.set_xlabel("Time")
axes.set_ylabel("Average Triangle Density")

#Add title
axes.set_title("Average Triangle Density of the MF Simulation")

#Initialize line graphs
axes.plot(mtdeMF1, color = 'black', label = 'n=10')
axes.plot(mtdeMF2, color = 'red', label = 'n=20')
axes.plot(mtdeMF3, color = 'blue', label = 'n=50')
axes.plot(mtdeMF4, color = 'green', label = 'n=100')
axes.plot(mtdeMF5, color = 'yellow', label = 'n=200')
axes.plot(mtdeMF6, color = 'orange', label = 'n=500')
axes.plot(mtdeMF7, color = 'purple', label = 'n=1000')

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Triangle density plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimT)
axes.set_xlabel("Time")
axes.set_ylabel("Average Triangle Density")

#Add title
axes.set_title("Average Triangle Density")

#Initialize line graphs
axes.plot(mtde1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(mtde2, color = 'red', label = 'n=20')
axes.plot(mtde3, color = 'blue', label = 'n=50')
axes.plot(mtde4, color = 'green', label = 'n=100')
axes.plot(mtde5, color = 'yellow', label = 'n=200')
axes.plot(mtde6, color = 'orange', label = 'n=500')
axes.plot(mtde7, color = 'purple', label = 'n=1000')
axes.plot(mtdeMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Largest eigenvalue plot

#Get the largest eigenvalue bounds
mle1 = np.mean(le1,1)/10
mle2 = np.mean(le2,1)/20
mle3 = np.mean(le3,1)/50
mle4 = np.mean(le4,1)/100
mle5 = np.mean(le5,1)/200
mle6 = np.mean(le6,1)/500
mle7 = np.mean(le7,1)/1000

mleMF1 = np.mean(leMF1,1)/10
mleMF2 = np.mean(leMF2,1)/20
mleMF3 = np.mean(leMF3,1)/50
mleMF4 = np.mean(leMF4,1)/100
mleMF5 = np.mean(leMF5,1)/200
mleMF6 = np.mean(leMF6,1)/500
mleMF7 = np.mean(leMF7,1)/1000

mxm1 = np.max(np.abs(mle1))
mxm2 = np.max(np.abs(mle2))
mxm3 = np.max(np.abs(mle3))
mxm4 = np.max(np.abs(mle4))
mxm5 = np.max(np.abs(mle5))
mxm6 = np.max(np.abs(mle6))
mxm7 = np.max(np.abs(mle7))

mxmMF1 = np.max(np.abs(mleMF1))
mxmMF2 = np.max(np.abs(mleMF2))
mxmMF3 = np.max(np.abs(mleMF3))
mxmMF4 = np.max(np.abs(mleMF4))
mxmMF5 = np.max(np.abs(mleMF5))
mxmMF6 = np.max(np.abs(mleMF6))
mxmMF7 = np.max(np.abs(mleMF7))

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxlimMF)

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimMF)
axes.set_xlabel("Time")
axes.set_ylabel("Average Largest Eigenvalue")

#Add title
axes.set_title("Average Largest Eigenvalue of the MF Simulation")

#Initialize line graphs
axes.plot(mleMF1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(mleMF2, color = 'red', label = 'n=20')
axes.plot(mleMF3, color = 'blue', label = 'n=50')
axes.plot(mleMF4, color = 'green', label = 'n=100')
axes.plot(mleMF5, color = 'yellow', label = 'n=200')
axes.plot(mleMF6, color = 'orange', label = 'n=500')
axes.plot(mleMF7, color = 'purple', label = 'n=1000')
#axes.plot(mtdeMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Largest eigenvalue plots

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimT)
axes.set_xlabel("Time")
axes.set_ylabel("Average Largest Eigenvalue")

#Add title
axes.set_title("Average Largest Eigenvalue")

#Initialize line graphs
axes.plot(mle1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(mle2, color = 'red', label = 'n=20')
axes.plot(mle3, color = 'blue', label = 'n=50')
axes.plot(mle4, color = 'green', label = 'n=100')
axes.plot(mle5, color = 'yellow', label = 'n=200')
axes.plot(mle6, color = 'orange', label = 'n=500')
axes.plot(mle7, color = 'purple', label = 'n=1000')
axes.plot(mleMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxlimMF)

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0, mxlimMF)
axes.set_xlabel("Time")
axes.set_ylabel("Average Second Largest Eigenvalue")

#Add title
axes.set_title("Average Second Largest Eigenvalue of the MF Simulation")

#Initialize line graphs
axes.plot(msleMF1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(msleMF2, color = 'red', label = 'n=20')
axes.plot(msleMF3, color = 'blue', label = 'n=50')
axes.plot(msleMF4, color = 'green', label = 'n=100')
axes.plot(msleMF5, color = 'yellow', label = 'n=200')
axes.plot(msleMF6, color = 'orange', label = 'n=500')
axes.plot(msleMF7, color = 'purple', label = 'n=1000')
#axes.plot(mtdeMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Second Largest eigenvalue plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimT)
axes.set_xlabel("Time")
axes.set_ylabel("Average Second Largest Eigenvalue")

#Add title
axes.set_title("Average Second Largest Eigenvalue")

#Initialize line graphs
axes.plot(msle1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(msle2, color = 'red', label = 'n=20')
axes.plot(msle3, color = 'blue', label = 'n=50')
axes.plot(msle4, color = 'green', label = 'n=100')
axes.plot(msle5, color = 'yellow', label = 'n=200')
axes.plot(msle6, color = 'orange', label = 'n=500')
axes.plot(msle7, color = 'purple', label = 'n=1000')
axes.plot(msleMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Third Largest eigenvalue plot

#Get the second largest eigenvalue bounds
mtle1 = np.mean(tle1,1)/10
mtle2 = np.mean(tle2,1)/20
mtle3 = np.mean(tle3,1)/50
mtle4 = np.mean(tle4,1)/100
mtle5 = np.mean(tle5,1)/200
mtle6 = np.mean(tle6,1)/500
mtle7 = np.mean(tle7,1)/1000

mtleMF1 = np.mean(tleMF1,1)/10
mtleMF2 = np.mean(tleMF2,1)/20
mtleMF3 = np.mean(tleMF3,1)/50
mtleMF4 = np.mean(tleMF4,1)/100
mtleMF5 = np.mean(tleMF5,1)/200
mtleMF6 = np.mean(tleMF6,1)/500
mtleMF7 = np.mean(tleMF7,1)/1000

mxm1 = np.max(np.abs(mtle1))
mxm2 = np.max(np.abs(mtle2))
mxm3 = np.max(np.abs(mtle3))
mxm4 = np.max(np.abs(mtle4))
mxm5 = np.max(np.abs(mtle5))
mxm6 = np.max(np.abs(mtle6))
mxm7 = np.max(np.abs(mtle7))

mxmMF1 = np.max(np.abs(mtleMF1))
mxmMF2 = np.max(np.abs(mtleMF2))
mxmMF3 = np.max(np.abs(mtleMF3))
mxmMF4 = np.max(np.abs(mtleMF4))
mxmMF5 = np.max(np.abs(mtleMF5))
mxmMF6 = np.max(np.abs(mtleMF6))
mxmMF7 = np.max(np.abs(mtleMF7))

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxlimMF)

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0, mxlimMF)
axes.set_xlabel("Time")
axes.set_ylabel("Average Third Largest Eigenvalue")

#Add title
axes.set_title("Average Third Largest Eigenvalue of the MF Simulation")

#Initialize line graphs
axes.plot(mtleMF1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(mtleMF2, color = 'red', label = 'n=20')
axes.plot(mtleMF3, color = 'blue', label = 'n=50')
axes.plot(mtleMF4, color = 'green', label = 'n=100')
axes.plot(mtleMF5, color = 'yellow', label = 'n=200')
axes.plot(mtleMF6, color = 'orange', label = 'n=500')
axes.plot(mtleMF7, color = 'purple', label = 'n=1000')
#axes.plot(mtdeMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########Second Largest eigenvalue plots
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,101)
axes.set_ylim(0,mxlimT)
axes.set_xlabel("Time")
axes.set_ylabel("Average Second Largest Eigenvalue")

#Add title
axes.set_title("Average Second Largest Eigenvalue")

#Initialize line graphs
axes.plot(mtle1, color = 'black', label = 'n=10', linestyle = '--')
axes.plot(mtle2, color = 'red', label = 'n=20')
axes.plot(mtle3, color = 'blue', label = 'n=50')
axes.plot(mtle4, color = 'green', label = 'n=100')
axes.plot(mtle5, color = 'yellow', label = 'n=200')
axes.plot(mtle6, color = 'orange', label = 'n=500')
axes.plot(mtle7, color = 'purple', label = 'n=1000')
axes.plot(mtleMF7, color = 'black', label = 'MF')
#axes.plot(mdeMF7**3, color = 'red', linestyle = ':',label = 'Erdos Renyi', linewidth = 2)

#Add legend
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

###############QQ Plots

#Get quantiles at time T
mqua1 = np.flatten(qua1[:,ssl1.cp.T-1,:],axis = 1)
mqua2 = qua2[:,ssl2.cp.T-1,:]
mqua3 = qua3[:,ssl3.cp.T-1,:]
mqua4 = qua4[:,ssl4.cp.T-1,:]
mqua5 = qua5[:,ssl5.cp.T-1,:]
mqua6 = qua6[:,ssl6.cp.T-1,:]
mqua7 = qua7[:,ssl7.cp.T-1,:]

mquaMF1 = quaMF1[:,ssl1.cp.T-1,:]
mquaMF2 = quaMF2[:,ssl2.cp.T-1,:]
mquaMF3 = quaMF3[:,ssl3.cp.T-1,:]
mquaMF4 = quaMF4[:,ssl4.cp.T-1,:]
mquaMF5 = quaMF5[:,ssl5.cp.T-1,:]
mquaMF6 = quaMF6[:,ssl6.cp.T-1,:]
mquaMF7 = quaMF7[:,ssl7.cp.T-1,:]

mxm1 = np.max(mqua1)
mxm2 = np.max(mqua2)
mxm3 = np.max(mqua3)
mxm4 = np.max(mqua4)
mxm5 = np.max(mqua5)
mxm6 = np.max(mqua6)
mxm7 = np.max(mqua7)

mxmMF1 = np.max(mquaMF1)
mxmMF2 = np.max(mquaMF2)
mxmMF3 = np.max(mquaMF3)
mxmMF4 = np.max(mquaMF4)
mxmMF5 = np.max(mquaMF5)
mxmMF6 = np.max(mquaMF6)
mxmMF7 = np.max(mquaMF7)

mxlim = max(mxm1,mxm2,mxm3,mxm4,mxm5,mxm6,mxm7)
mxlimMF = max(mxmMF1,mxmMF2,mxmMF3,mxmMF4,mxmMF5,mxmMF6,mxmMF7)
mxlimT = max(mxlim,mxlimMF)

# =============================================================================
# #Max quantile and actual quantiles
# mxqua = np.ceil(stats.chi2.ppf(q=0.98,df=2))
# quas = stats.chi2.ppf(q=np.arange(0.02,1,0.02),df=2)
# 
# i=0
# for n in [10,20,50,100,200,500,1000]:
#     ##Create figure and axes for animation
#     fig, axes = plt.subplots()
#     
#     #Create axes
#     axes.set_xlim(0,mxqua)
#     axes.set_ylim(0,mxlimT)
#     axes.set_xlabel("chi2 quantiles")
#     axes.set_ylabel("sample quantiles")
#     
#     #Add title
#     axes.set_title("Mahalanobis QQ Plot: n = "+str(n))
#     
#     #Initialize line graphs
#     axes.scatter(quas, mqua1, c='b',s=5)
#     
#     ##Create figure and axes for animation
#     fig, axes = plt.subplots()
#     
#     #Create axes
#     axes.set_xlim(0,mxqua)
#     axes.set_ylim(0,mxlimT)
#     axes.set_xlabel("chi2 quantiles")
#     axes.set_ylabel("sample quantiles")
#     
#     #Add title
#     axes.set_title("Mahalanobis QQ Plot (Mean Field): n = "+str(n))
#     
#     #Initialize line graphs
#     axes.scatter(quas, mquaMF1[:,i], c='b',s=5)
#     
#     i=i+1
# =============================================================================
