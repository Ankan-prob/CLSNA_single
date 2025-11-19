# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:03:37 2025

@author: ylugn
"""

import numpy as np
import os

ns = [10,20,50,100,200,500,1000]
eb = "simit11"

#Extract and combine zstat data from filebase1 and filebase2 into filebase
def combZstats(filebase1,filebase2,filebase):
    #Open npz file in filebase 1
    with np.load(".//simdata//"+filebase1+"//zstats.npz") as dat:
        sm1 = dat['sm']
        sc1 = dat['sc']
        qua1 = dat['qua']
        ma1 = dat['ma']
        mi1 = dat['mi']
        smMF1 = dat['smMF']
        scMF1 = dat['scMF']
        quaMF1 = dat['quaMF']
        maMF1 = dat['maMF']
        miMF1 = dat['miMF']
        mse1 = dat['mse']
        
    #Open npz file in filebase 2
    with np.load(".//simdata//"+filebase2+"//zstats.npz") as dat:
        sm2 = dat['sm']
        sc2 = dat['sc']
        qua2 = dat['qua']
        ma2 = dat['ma']
        mi2 = dat['mi']
        smMF2 = dat['smMF']
        scMF2 = dat['scMF']
        quaMF2 = dat['quaMF']
        maMF2 = dat['maMF']
        miMF2 = dat['miMF']
        mse2 = dat['mse']
        
    #Combine
    sm = np.concatenate((sm1,sm2),axis = 2)
    sc = np.concatenate((sc1,sc2),axis = 3)
    qua = np.concatenate((qua1,qua2),axis = 2)
    ma = np.concatenate((ma1,ma2),axis = 1)
    mi = np.concatenate((mi1,mi2),axis = 1)
    smMF = np.concatenate((smMF1,smMF2),axis = 2)
    scMF = np.concatenate((scMF1,scMF2),axis = 3)
    quaMF = np.concatenate((quaMF1,quaMF2),axis = 2)
    maMF = np.concatenate((maMF1,maMF2),axis = 1)
    miMF = np.concatenate((miMF1,miMF2),axis = 1)
    mse = np.concatenate((mse1,mse2),axis = 1)
    
    #Save
    np.savez(".//simdata//"+filebase+"//zstats.npz",sm=sm,sc=sc\
             ,qua=qua,ma=ma,mi=mi,smMF=smMF\
             ,scMF=scMF,quaMF=quaMF,maMF=maMF,miMF=miMF\
             ,mse = mse)
        
    return()

#Extract and combine astat data from filebase1 and filebase2 into filebase
def combAstats(filebase1,filebase2,filebase):
    #Open npz file in filebase 1
    with np.load(".//simdata//"+filebase1+"//astats.npz") as dat:
        de1 = dat['de']
        tde1 = dat['tde']
        cl1 = dat['cl']
        le1 = dat['le']
        se1 = dat['se']
        deMF1 = dat['deMF']
        tdeMF1 = dat['tdeMF']
        clMF1 = dat['clMF']
        leMF1 = dat['leMF']
        seMF1 = dat['seMF']
        sd1 = dat['sd']
        ssd1 = dat['ssd']
        
    #Open npz file in filebase 2
    with np.load(".//simdata//"+filebase2+"//astats.npz") as dat:
        de2 = dat['de']
        tde2 = dat['tde']
        cl2 = dat['cl']
        le2 = dat['le']
        se2 = dat['se']
        deMF2 = dat['deMF']
        tdeMF2 = dat['tdeMF']
        clMF2 = dat['clMF']
        leMF2 = dat['leMF']
        seMF2 = dat['seMF']
        sd2 = dat['sd']
        ssd2 = dat['ssd']
        
    #Combine
    de = np.concatenate((de1,de2),axis = 1)
    tde = np.concatenate((tde1,tde2),axis = 1)
    cl = np.concatenate((cl1,cl2),axis = 1)
    le = np.concatenate((le1,le2),axis = 2)
    se = np.concatenate((se1,se2),axis = 2)
    deMF = np.concatenate((deMF1,deMF2),axis = 1)
    tdeMF = np.concatenate((tdeMF1,tdeMF2),axis = 1)
    clMF = np.concatenate((clMF1,clMF2),axis = 1)
    leMF = np.concatenate((leMF1,leMF2),axis = 2)
    seMF = np.concatenate((seMF1,seMF2),axis = 2)
    sd = np.concatenate((sd1,sd2),axis = 1)
    ssd = np.concatenate((ssd1,ssd2),axis = 1)
    
    #Save
    np.savez(".//simdata//"+filebase+"//astats.npz",de=de,tde=tde\
             ,cl=cl,le=le,se=se,deMF=deMF\
             ,tdeMF=tdeMF,clMF=clMF,leMF=leMF,seMF=seMF\
             ,sd=sd,ssd=ssd)
        
    return()

#Extract and combine parameter data from filebase1 and filebase2 into filebase
def combParams(filebase1,filebase2,filebase):
    Ps = np.load(".//simdata//"+filebase1+"//ParamArray.npy")
    Ps[2] = '101'
    np.save(".//simdata//"+filebase+"//ParamArray.npy",Ps)
    
    return()

#Extract and combine parameter text from filebase1 and filebase2 into filebase
def combtext(filebase1,filebase2,filebase):
    newline = '\n' # Defines the newline based on your OS.

    source_fp = open(".//simdata//"+filebase1+"//Params.txt", 'r')
    target_fp = open(".//simdata//"+filebase+"//Params.txt", 'w')
    first_row = True
    for row in source_fp:
        if first_row:
            row = filebase + " Parameters:"
            first_row = False
    target_fp.write(row + newline)
    
#Combine data for all filebases
#Start N=1 case
for n in ns:
    fb1 = eb + "v1n" + str(n)
    fb2 = eb + "v2n" + str(n)
    fb = eb + "q2n" + str(n)
    
    os.mkdir('.//simdata//'+fb)
    
    combZstats(fb1,fb2,fb)
    combAstats(fb1,fb2,fb)
    combParams(fb1,fb2,fb)
    combtext(fb1,fb2,fb)

#N=2 to 9 cases
for N in np.arange(2,9):
    for n in ns:
        fb1 = eb + "c"+ str(N) + "n" + str(n)
        fb2 = eb + "v"+ str(N+1) + "n" + str(n)
        fb = eb + "q"+ str(N) + "n" + str(n)
        
        os.mkdir('.//simdata//'+fb)
        
        combZstats(fb1,fb2,fb)
        combAstats(fb1,fb2,fb)
        combParams(fb1,fb2,fb)
        combtext(fb1,fb2,fb)
        
#Final N=10 case
for n in ns:
    fb1 = eb + "q9n" + str(n)
    fb2 = eb + "v10n" + str(n)
    fb = eb + "n" + str(n)
    
    os.mkdir('.//simdata//'+fb)
    
    combZstats(fb1,fb2,fb)
    combAstats(fb1,fb2,fb)
    combParams(fb1,fb2,fb)
    combtext(fb1,fb2,fb)