# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 13:30:06 2025

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
# for k = 10,20,50,100,200,500,1000 and N = 1-20 excluding 13,
# load the filebase simitNnk
# =============================================================================

#Base data file 
eb = "simit" #N=4000

#Parameters
ns = np.array([10,20,50,100,200,500,1000])
#There was an error in simit13. I think there is small chance
#that the MF algorithm diverges from a random walk.
Rs = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21])

#Number of Ns
NR = np.size(Rs)

#Start time for statistics that average over time
st = 20
tms = np.log(np.array([10,20,50,100,200,500,1000]))


##########Primary Plot Objects
#Prepare mse mean object
msemn = {}
for n in ns:
    msemn[str(n)] = np.zeros(100)

#Prepare sample var for n = 1000
msevar = np.zeros(100)

#Prepare avgs MSE over time with error bds
mseavgt = np.zeros(7)
mseavgtvar = np.zeros(7)

#Prepare MSE bounds
msemnbds = np.zeros(2)
mseerrbds = np.zeros(2)
mseavgtbds = np.zeros(2)

#Symmetric difference mean object
sdmn = {}
for n in ns:
    sdmn[str(n)] = np.zeros(100)

#Symmetric difference sample var
sdvar = np.zeros(100)

#Prepare avgs SD over time with error bds
sdavgt = np.zeros(7)
sdavgtvar = np.zeros(7)

#Prepare SD bounds
sdmnbds = np.zeros(2)
sderrbds = np.zeros(2)
sdavgtbds = np.zeros(2)

#Triangle Density Mean Object
tdemn = {}
for n in ns:
    tdemn[str(n)] = np.zeros(100)

#Triangle Density sample var
tdevar = np.zeros(100)

#Prepare avgs TDE over time with error bds
tdeavgt = np.zeros(7)
tdeavgtvar = np.zeros(7)

#tde vs Erdos
erdtde = np.zeros(100)
MFtde = np.zeros(100)

#Prepare TDE bounds
tdemnbds = np.zeros(2)
tdeerrbds = np.zeros(2)
tdeavgtbds = np.zeros(2)
tdeerdbds = np.zeros(2)

#Second Largest Eigenvalue Mean Object
slemn = {}
for n in ns:
    slemn[str(n)] = np.zeros(100)
#Second largest eigenvalue sample var
slevar = np.zeros(100)

#Prepare avgs SLE over time with error bds
sleavgt = np.zeros(7)
sleavgtvar = np.zeros(7)

#Prepare SLE bounds
slemnbds = np.zeros(2)
sleerrbds = np.zeros(2)
sleavgtbds = np.zeros(2)


#########Get some common parameters
#Load basic data
fb1 = eb+"1n10"
cp1 = sm.CoupledParticle()
ssl1 = sl.SimulateSaveLoad(fb1,op=1,cp=cp1)

#start time for stats that average over time
st = 20

#log time
tms = np.log(np.array([10,20,50,100,200,500,1000]))

#Common parameters
T = ssl1.cp.T
m = ssl1.m
d = ssl1.cp.d
N = ssl1.cp.N

#Reference measure
#refmf = ssl1.loadref()

#Run an MF simulation.
#This should be deterministic and not depend on which ssl we choose
#ssl1.saveMeanMF()
#mMF = ssl1.loadMeanMF()


###########Iterate over all N to fill out plot objects
#itn = number of the current iteration.
itn = 0
for R in Rs:
    
    #increment the iteration number
    itn = itn + 1
    
    #define an index number for arrays indexed by n
    indx = -1
    
    for n in ns:
        #increment index
        indx = indx + 1
        
        ##Load data
        #set filebase
        filebase = eb + str(R) + "n" + str(n)
        
        #load coupled particle
        cp = sm.CoupledParticle()
        
        #load stored data
        ssl = sl.SimulateSaveLoad(filebase,op=1,cp=cp)
        
        #load Z stats
        zstat = ssl.loadmCoupledZStatistics()
        
        #load A stats
        astat = ssl.loadmCoupledAStatistics()
        
        ####Generate mse averages and confidence bands
        #Get the average mse at each time
        mse = zstat[10]
        mmse = np.mean(mse,1)
        
        #Get the average mse over time
        mtmse = np.mean(mmse[st:])
        
        #update info using Welford's online algorithm
        #store old means
        oldmsemn = msemn[str(n)]
        oldmseavgt = mseavgt[indx]
        
        #msemn represents online mean
        msemn[str(n)] = oldmsemn + (mmse - oldmsemn)/itn
        #mseavgt represents online mean average over time
        mseavgt[indx] = oldmseavgt + (mtmse - oldmseavgt)/itn
        
        #total square error
        if n == 1000:
            msevar = msevar + (mmse - oldmsemn)*(mmse - msemn[str(n)])
            
        mseavgtvar[indx] = mseavgtvar[indx] + (mtmse - oldmseavgt)*(mtmse - mseavgt[indx])
        
        ####Generate sd averages and confidence bands
        #Get the average mse at each time
        sd = astat[10]/2
        msd = np.mean(sd,1)/(n*(n-1)/2.0)
        
        #Get the average mse over time
        msdt = np.mean(msd[st:])
        
        #update info using Welford's online algorithm
        #store old means
        oldsdmn = sdmn[str(n)]
        oldsdavgt = sdavgt[indx]
        
        #sdmn represents online mean
        sdmn[str(n)] = oldsdmn + (msd - oldsdmn)/itn
        #sdavgt represents online mean average over time
        sdavgt[indx] = oldsdavgt + (msdt - oldsdavgt)/itn
        
        #total square error
        if n == 1000:
            sdvar = sdvar + (msd - oldsdmn)*(msd - sdmn[str(n)])
            
        sdavgtvar[indx] = sdavgtvar[indx] + (msdt - oldsdavgt)*(msdt - sdavgt[indx])
        
        ####Generate tde averages and confidence bands
        #Get the average mse at each time
        tde = astat[1]
        tdeMF = astat[6]
        tdediff = np.abs(tdeMF - tde)
        mtde = np.mean(tdediff,1)
        
        #Get the average tde over time
        mtdet = np.mean(mtde[st:])
        
        #update info using Welford's online algorithm
        #store old means
        oldtdemn = tdemn[str(n)]
        oldtdeavgt = tdeavgt[indx]
        
        #sdmn represents online mean
        tdemn[str(n)] = oldtdemn + (mtde - oldtdemn)/itn
        #sdavgt represents online mean average over time
        tdeavgt[indx] = oldtdeavgt + (mtdet - oldtdeavgt)/itn
        
        #total square error
        if n == 1000:
            tdevar = tdevar + (mtde - oldtdemn)*(mtde - tdemn[str(n)])
            
        tdeavgtvar[indx] = tdeavgtvar[indx] + (mtdet - oldtdeavgt)*(mtdet - tdeavgt[indx])

        #Erdos Renyi
        if n == 1000:
            deMF = astat[5]
            mdeMF = np.mean(deMF,1)
            ERtde = 6*sp.special.comb(1000,3)*mdeMF**3/(1000**3)
            mtdeMF = np.mean(tdeMF,1)
            
            #Welford's online algorithm
            erdtde = erdtde + (ERtde - erdtde)/itn
            MFtde = MFtde + (mtdeMF - MFtde)/itn
        
        ####Generate sle averages and confidence bands
        #Get the average sle at each time
        sle = astat[3][1,:,:]
        sleMF = astat[8][1,:,:]
        slediff = sleMF - sle
        msle = np.mean(slediff,1)/n
        
        #Get the average tde over time
        mslet = np.mean(msle[st:])
        
        #update info using Welford's online algorithm
        #store old means
        oldslemn = slemn[str(n)]
        oldsleavgt = sleavgt[indx]
        
        #sdmn represents online mean
        slemn[str(n)] = oldslemn + (msle - oldslemn)/itn
        #sdavgt represents online mean average over time
        sleavgt[indx] = oldsleavgt + (mslet - oldsleavgt)/itn
        
        #total square error
        if n == 1000:
            slevar = slevar + (msle - oldslemn)*(msle - slemn[str(n)])
            
        sleavgtvar[indx] = sleavgtvar[indx] + (mslet - oldsleavgt)*(mslet - sleavgt[indx])
        
####Calculate Mean 95% confidence bound
msevar = 2*np.sqrt(msevar/(NR*(NR-1)))
mseavgtvar = 2*np.sqrt(mseavgtvar/(NR*(NR-1)))
sdvar = 2*np.sqrt(sdvar/(NR*(NR-1)))
sdavgtvar = 2*np.sqrt(sdavgtvar/(NR*(NR-1)))
tdevar = 2*np.sqrt(tdevar/(NR*(NR-1)))
tdeavgtvar = 2*np.sqrt(tdeavgtvar/(NR*(NR-1)))
slevar = 2*np.sqrt(slevar/(NR*(NR-1)))
sleavgtvar = 2*np.sqrt(sleavgtvar/(NR*(NR-1)))

######MSE bounds
###(a) msemnbds[0] = 0
msemnmx = 0
for n in ns:
    msemnn = msemn[str(n)]
    msemnmx = max(msemnmx,np.max(msemnn))
msemnbds[1] = msemnmx

###(b)
mseerrbds[0] = np.min(msemn["1000"] - msevar)
mseerrbds[1] = np.max(msemn["1000"] + msevar)

###(c) Set mseavgtbds[0] = 0
#mseavgtbds[0] = np.min(mseavgt - mseavgtvar)
mseavgtbds[1] = np.max(mseavgt - mseavgtvar)

######SD bounds
###(a) sdmnbds[0] = 0
sdmnmx = 0
for n in ns:
    sdmnn = sdmn[str(n)]
    sdmnmx = max(sdmnmx,np.max(sdmnn))
sdmnbds[1] = sdmnmx

###(b)
sderrbds[0] = np.min(sdmn["1000"] - sdvar)
sderrbds[1] = np.max(sdmn["1000"] + sdvar)

###(c) Set sdavgtbds[0] = 0
#sdavgtbds[0] = np.min(sdavgt - sdavgtvar)
sdavgtbds[1] = np.max(sdavgt - sdavgtvar)

######TDE bounds
###(a) sdmnbds[0] = 0
tdemnmx = 0
for n in ns:
    tdemnn = tdemn[str(n)]
    tdemnmx = max(tdemnmx,np.max(tdemnn))
tdemnbds[1] = tdemnmx

###(b)
tdeerrbds[0] = np.min(tdemn["1000"] - tdevar)
tdeerrbds[1] = np.max(tdemn["1000"] + tdevar)

###(c) Set tdeavgtbds[0] = 0
#tdeavgtbds[0] = np.min(tdeavgt - tdeavgtvar)
tdeavgtbds[1] = np.max(tdeavgt - tdeavgtvar)

###(d) Set tdeerdbds[0] = 0
tdeerdbds[1] = max(np.max(erdtde),np.max(MFtde))

######SLE bounds
###(a) 
slemnmx = 0
slemnmn = 100
for n in ns:
    slemnn = slemn[str(n)]
    slemnmx = max(slemnmx,np.max(slemnn))
    slemnmn = min(slemnmn,np.min(slemnn))
slemnbds[0] = slemnmn
slemnbds[1] = slemnmx

###(b)
sleerrbds[0] = np.min(slemn["1000"] - slevar)
sleerrbds[1] = np.max(slemn["1000"] + slevar)

###(c) Set sleavgtbds[0] = 0
sleavgtbds[0] = np.min(sleavgt - sleavgtvar)
sleavgtbds[1] = np.max(sleavgt - sleavgtvar)


###############Make plots
#######MSE

###1a
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(msemnbds)
axes.set_xlabel("Time")
axes.set_ylabel("Average MSE")

#Add title
axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.plot(msemn['10'], color = 'gray', label = 'n=10', marker = '+')
axes.plot(msemn['20'], color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(msemn['50'], color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(msemn['100'], color = 'gray', label = 'n=100')
axes.plot(msemn['200'], color = 'black', label = 'n=200', linestyle = ':')
axes.plot(msemn['500'], color = 'black', label = 'n=500', linestyle = '--')
axes.plot(msemn['1000'], color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig1a.eps", bbox_inches = 'tight', format = 'eps')

###1b
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(mseerrbds)
axes.set_xlabel("Time")
axes.set_ylabel("Average MSE")

#Add title
axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.plot(msemn['1000'], color = 'black')
axes.plot(msemn['1000']+msevar, color = 'black', linestyle = '--')
axes.plot(msemn['1000']-msevar, color = 'black', linestyle = '--')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig1b.eps", bbox_inches = 'tight', format = 'eps')

###1c
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(mseavgtbds)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Average MSE")

#Add title
axes.set_title("Average MSE of the MF Approximation")

#Initialize line graphs
axes.errorbar(x=tms, y=mseavgt, yerr = mseavgtvar, color = 'black')

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig1c.eps", bbox_inches = 'tight')

#######SD

###2a
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(sdmnbds)
axes.set_xlabel("Time")
axes.set_ylabel("Density")

#Add title
axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.plot(sdmn['10'], color = 'gray', label = 'n=10', marker = '+')
axes.plot(sdmn['20'], color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(sdmn['50'], color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(sdmn['100'], color = 'gray', label = 'n=100')
axes.plot(sdmn['200'], color = 'black', label = 'n=200', linestyle = ':')
axes.plot(sdmn['500'], color = 'black', label = 'n=500', linestyle = '--')
axes.plot(sdmn['1000'], color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig2a.eps", bbox_inches = 'tight', format = 'eps')

###2b
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(sderrbds)
axes.set_xlabel("Time")
axes.set_ylabel("Density")

#Add title
axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.plot(sdmn['1000'], color = 'black')
axes.plot(sdmn['1000']+sdvar, color = 'black', linestyle = '--')
axes.plot(sdmn['1000']-sdvar, color = 'black', linestyle = '--')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig2b.eps", bbox_inches = 'tight', format = 'eps')

###2c
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(sdavgtbds)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Density")

#Add title
axes.set_title("Average Density of the Symmetric Difference Network")

#Initialize line graphs
axes.errorbar(x=tms, y=sdavgt, yerr = sdavgtvar, color = 'black')

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig2c.eps", bbox_inches = 'tight')

#######TDE

###3a
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(tdemnbds)
axes.set_xlabel("Time")
axes.set_ylabel("Error")

#Add title
axes.set_title("Average Triangle Density Error")

#Initialize line graphs
axes.plot(tdemn['10'], color = 'gray', label = 'n=10', marker = '+')
axes.plot(tdemn['20'], color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(tdemn['50'], color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(tdemn['100'], color = 'gray', label = 'n=100')
axes.plot(tdemn['200'], color = 'black', label = 'n=200', linestyle = ':')
axes.plot(tdemn['500'], color = 'black', label = 'n=500', linestyle = '--')
axes.plot(tdemn['1000'], color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3a.eps", bbox_inches = 'tight', format = 'eps')

###3b
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(tdeerrbds)
axes.set_xlabel("Time")
axes.set_ylabel("Error")

#Add title
axes.set_title("Average Triangle Density Error")

#Initialize line graphs
axes.plot(tdemn['1000'], color = 'black')
axes.plot(tdemn['1000']+tdevar, color = 'black', linestyle = '--')
axes.plot(tdemn['1000']-tdevar, color = 'black', linestyle = '--')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3b.eps", bbox_inches = 'tight', format = 'eps')

###3c
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(tdeavgtbds)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Error")

#Add title
axes.set_title("Average Triangle Density Error")

#Initialize line graphs
axes.errorbar(x=tms, y=tdeavgt, yerr = tdeavgtvar, color = 'black')

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig3c.eps", bbox_inches = 'tight')

###3d

##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(tdeerdbds)
axes.set_xlabel("Time")
axes.set_ylabel("Triangle Density")

#Add title
axes.set_title("Average Triangle Density")

#Initialize line graphs
axes.plot(MFtde, color = 'black', label = 'MF')
axes.plot(erdtde, color = 'gray', label = 'Erdos Renyi')

#Add legend
axes.legend(loc='lower left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig3d.eps", bbox_inches = 'tight', format = 'eps')

#######SLE

###4a
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(slemnbds)
axes.set_xlabel("Time")
axes.set_ylabel("Eigenvalue Error")

#Add title
axes.set_title("Average Second Largest Eigenvalue Error")

#Initialize line graphs
axes.plot(slemn['10'], color = 'gray', label = 'n=10', marker = '+')
axes.plot(slemn['20'], color = 'gray', label = 'n=20', linestyle = ':')
axes.plot(slemn['50'], color = 'gray', label = 'n=50', linestyle = '--')
axes.plot(slemn['100'], color = 'gray', label = 'n=100')
axes.plot(slemn['200'], color = 'black', label = 'n=200', linestyle = ':')
axes.plot(slemn['500'], color = 'black', label = 'n=500', linestyle = '--')
axes.plot(slemn['1000'], color = 'black', label = 'n=1000')

#Add legend
axes.legend(loc='center left',  bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig4a.eps", bbox_inches = 'tight', format = 'eps')

###4b
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_xlim(0,T+1)
axes.set_ylim(sleerrbds)
axes.set_xlabel("Time")
axes.set_ylabel("Eigenvalue Error")

#Add title
axes.set_title("Average Second Largest Eigenvalue Error")

#Initialize line graphs
axes.plot(slemn['1000'], color = 'black')
axes.plot(slemn['1000']+slevar, color = 'black', linestyle = '--')
axes.plot(slemn['1000']-slevar, color = 'black', linestyle = '--')

#Add legend
axes.legend(loc='upper left', framealpha = 0.1)

#Save figure
plt.savefig("plots\\Fig4b.eps", bbox_inches = 'tight', format = 'eps')

###3c
##Create figure and axes for animation
fig, axes = plt.subplots()

#Create axes
axes.set_ylim(sleavgtbds)
axes.set_xlabel("ln(n)")
axes.set_ylabel("Eigenvalue Error")

#Add title
axes.set_title("Average Second Largest Eigenvalue Error")

#Initialize line graphs
axes.errorbar(x=tms, y=sleavgt, yerr = sleavgtvar, color = 'black')

#Add legend
#axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#Save figure
plt.savefig("plots\\Fig4c.eps", bbox_inches = 'tight')