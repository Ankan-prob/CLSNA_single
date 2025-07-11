# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:59:45 2024

@author: ylugn
"""
import simulation as sm
import numpy as np
import os
import warnings
#import copy
import pickle as pk

#I'm not creating a method to erase data
#That needs to be done manually (for safety)
class SimulateSaveLoad():
    #m represents number of coupled simulations
    #cp represents a CoupledParticle object
    #filebase represents the directory where everything is saved
    #rfilebase is the directory where the reference MF is saved
    #Options (op):
        #0: Raise error if filebase already exists (saving new data)
        #1: Use existing parameters if filebase already exists (loading old data)
        ##2: Overwrite existing parameters if they exist (not coded)
    #Options are not saved as parameters and can be changed at will
    def __init__(self,filebase,rfilebase=None,op=0,q=49,nl=3,m=12,cp=sm.CoupledParticle()):
        self.filebase = filebase
        self.q = q
        self.nl = nl
        self.m = m
        self.cp = cp
        self.pth = os.path.join("simdata",filebase)     #This is the path where data is saved
        if rfilebase == None or rfilebase == filebase:
            self.rfilebase = filebase
            self.rfile = os.path.join("simdata",self.rfilebase)+"\\Ref.npy"
        else:
            self.rfilebase = rfilebase
            self.rfile = os.path.join("simdata",self.rfilebase)+"\\Ref.npy"
            self.checkrfilebase()
        
        
        #Construct Lines of text outlining parameters
        st = ['']*21
        st[0] = self.filebase + " Parameters:\n"
        st[1] = "q = " + str(self.q) + "\n"
        st[2] = "nl = " + str(self.nl) + "\n"
        st[3] = "m = " + str(self.m) + "\n"
        st[4] = "N = " + str(self.cp.N) + "\n"
        st[5] = "SS = " + str(self.cp.SS) + "\n"
        st[6] = "t0 = " + str(self.cp.t0) + "\n"
        st[7] = "noi = " + str(self.cp.noi) + "\n"
        st[8] = "n = " + str(self.cp.n) + "\n"
        st[9] = "T = " + str(self.cp.T) + "\n"
        st[10] = "gam = " + str(self.cp.gam) + "\n"
        st[11] = "d = " + str(self.cp.d) + "\n"
        st[12] = "C = " + str(self.cp.C) + "\n"
        st[13] = "p = " + str(self.cp.p) + "\n"
        st[14] = "a = " + str(self.cp.a) + "\n"
        st[15] = "delt = " + str(self.cp.delt) + "\n"
        st[16] = "dist = " + str(self.cp.dist) + "\n"
        st[17] = "iv = " + str(self.cp.iv) + "\n"
        st[18] = "ev = " + str(self.cp.ev) + "\n"
        st[19] = "its = " + str(self.cp.its) + "\n"
        st[20] = "rfilebase = " + self.rfilebase
        
        #Check if the path already exists
        if os.path.exists(self.pth):
            if op == 0: 
                raise Exception("This filebase has already been taken.")
            elif op == 1:
                self.changeFilebase(self.filebase)
                warnings.warn("Parameters may have changed.")
                return
            else:
                raise ValueError("op has an invalid value.")

        os.mkdir(self.pth)
        
        with open(self.pth + "\\Params.txt",'x') as f:
            f.writelines(st)
        f.close()
        
        #Construct an np array with parameters
        params = np.array([self.q,self.nl,self.m,self.cp.N,self.cp.SS,self.cp.t0,\
                           self.cp.noi,self.cp.n,self.cp.T,self.cp.gam,self.cp.d,\
                           self.cp.C,self.cp.p,self.cp.a,self.cp.delt,self.cp.dist,\
                           self.cp.iv,self.cp.ev,self.cp.its,self.rfilebase])
        np.save(self.pth + "\\ParamArray.npy",params)
        return
    
        
    #Define print function representations of the class
    def __str__(self):
        return "SimulateSaveLoad Class with filebase "+self.filebase
    
    #load a Coupled Particle object from a (possibly different) filebase
    def loadCPFromFilebase(self,filebase):
        params = np.load(os.path.join("simdata",filebase)+"\\ParamArray.npy")
        q = int(params[0])
        nl = int(params[1])
        m = int(params[2])
        N = int(params[3])
        SS = int(params[4])
        t0 = int(params[5])
        noi = int(params[6])
        n = int(params[7])
        T = int(params[8])
        gam = float(params[9])
        d = int(params[10])
        C = float(params[11])
        p = float(params[12])
        a = float(params[13])
        delt = float(params[14])
        dist = float(params[15])
        iv = int(params[16])
        ev = int(params[17])
        its = int(params[18])
        rfilebase = params[19]
        cp = sm.CoupledParticle(N,SS,t0,noi,n,T,gam,d,C,p,a,delt,dist,iv,ev,its)
        return (q,nl,m,cp,rfilebase)
    
    #Rewrite the parameters of the current object to match an existing filebase
    def changeFilebase(self,filebase):
        self.filebase = filebase
        self.pth = os.path.join("simdata",filebase)
        (self.q,self.nl,self.m,self.cp,self.rfilebase) = self.loadCPFromFilebase(self.filebase)
        self.rfile = os.path.join("simdata",self.rfilebase) + "\\Ref.npy"
        return 
    
    def checkrfilebase(self):
        (q,nl,m,ccp,rf) = self.loadCPFromFilebase(self.rfilebase)
        
        #Check if the relevant parameters match
        t = np.array([False]*14,dtype = bool)
        t[0] = (self.cp.N==ccp.N)
        t[1] = (self.cp.t0==ccp.t0)
        t[2] = (self.cp.noi==ccp.noi)
        t[3] = (self.cp.T==ccp.T)
        t[4] = (self.cp.gam==ccp.gam)
        t[5] = (self.cp.d==ccp.d)
        t[6] = (self.cp.C==ccp.C)
        t[7] = (self.cp.p==ccp.p)
        t[8] = (self.cp.a==ccp.a)
        t[9] = (self.cp.delt==ccp.delt)
        t[10] = (self.cp.dist==ccp.dist)
        t[11] = (self.cp.iv==ccp.iv)
        t[12] = (self.cp.ev==ccp.ev)
        t[13] = (self.cp.its==ccp.its)
        
        if t.all() and os.path.exists(self.rfile):
            return True
        else:
            raise Exception("Invalid parameters for loading reference.")

    #Safety check to make sure I'm always operating the correct file
    #Same options as previously
    def safetyCheck(self,op=0):
        (q,nl,m,ccp,rf) = self.loadCPFromFilebase(self.filebase)
        if ccp == self.cp and (q,nl,m) == (self.q,self.nl,self.m) and rf == self.rfilebase:
                return True
        else:
            if op == 0:
                raise Exception("Parameter mismatch")
            elif op == 1:
                warnings.warn("Parameters have changed.")
                return False
            else:
                raise ValueError("op has an invalid value.") 
        
    #Simulate and save reference particles
    #Assumes parameters have already been saved
    def saveref(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        #reference filebase matches filebase
        if self.rfilebase != self.filebase:
            warnings.warn("Reference is in filebase " + self.rfilebase + "!")
            return 
        
        #Check if the reference already exists
        if os.path.exists(self.rfile):
            return 
        
        #Construct the reference and save it
        rf = self.cp.ConstructReference()
        np.save(self.pth + "\\Ref.npy", rf)
        return 

    #Load reference particles from a (possibly different) filebase
    def loadref(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        return np.load(self.rfile)
    
    #Save a Mean MF (MF generated without noise)
    def saveMeanMF(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        #Check if the Mean MF already exists
        if os.path.exists(self.pth+"\\MRef.npy"):
            return 
        
        #load the reference measure
        zref = self.loadref()
        
        #Construct the noiseless MF process
        zout = self.cp.ConstructMeanMF(zref)
        
        #Save zout
        np.save(self.pth + "\\MRef.npy",zout)
        
        #return
        return
    
    def loadMeanMF(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        return np.load(self.pth + "\\MRef.npy")
                    
    #Save n particle only (useful for finding good parameters)
    def savenpartdata(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        #Check if file exists
        if os.path.exists(self.pth + "\\npartZ.npy"):
            return
        else:
            #Run/Save data
            (At,Zt) = self.cp.simulate_full_lowmem()
            
            #Save network trajectories
            with open(self.pth + '\\npartA.pickle', 'wb') as handle:
                pk.dump(At, handle, protocol=pk.HIGHEST_PROTOCOL)
            handle.close()
            
            #Save Z
            np.save(self.pth + "\\npartZ.npy",Zt)
            
            return
            
    def loadnpartdata(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        #Load At
        with open(self.pth + '\\npartA.pickle', 'rb') as handle:
            At = pk.load(handle)
            handle.close()
            
        Zt = np.load(self.pth + "\\npartZ.npy")
        
        return (At,Zt)

    #Simulate and save a coupled simulation given a reference
    #filebase does not need to match our filebase
    #filebase describes where the reference is kept
    #if we already have a reference, use the existing reference measure
    def saveCoupleGivenRef(self, op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        #load zref
        zref = self.loadref()
            
        #Check if we've already generated this
        if os.path.exists(self.pth + "\\coupledZ.npy"):
            return
        
        (At,Zt,AMFt,ZMFt) = self.cp.CoupledSimulation(zref)
        
        #Save network trajectories
        with open(self.pth + '\\coupledA.pickle', 'wb') as handle:
            pk.dump(At, handle, protocol=pk.HIGHEST_PROTOCOL)
        handle.close()
        
        with open(self.pth + '\\coupledAMF.pickle', 'wb') as handle:
            pk.dump(AMFt, handle, protocol=pk.HIGHEST_PROTOCOL)
        handle.close()
        
        #Save particle trajectories
        np.save(self.pth + "\\coupledZ.npy",Zt)
        np.save(self.pth + "\\coupledZMF.npy",ZMFt)
        
        return
        
    def loadCoupled(self,op=0):
        #Run a safety check
        self.safetyCheck(op)
        
        #Load At
        with open(self.pth + '\\coupledA.pickle', 'rb') as handle:
            At = pk.load(handle)
            handle.close()
            
        with open(self.pth + '\\coupledAMF.pickle', 'rb') as handle:
            AMFt = pk.load(handle)
            handle.close()
            
        Zt = np.load(self.pth + "\\coupledZ.npy")
        ZMFt = np.load(self.pth + "\\coupledZMF.npy")
        
        return (At,Zt,AMFt,ZMFt)
    
    #When both m and n are large, we instead save important statistics
    def savemCoupledStatistics(self, op = 0, DEBUG = False):
        #Run a safety check
        self.safetyCheck(op)
        
        
        if os.path.exists(self.pth+"\\zstats.npz"):
            return
        
        #load the reference
        zref = self.loadref()
        
        #Generate the statistics
        dat = self.cp.mCoupledGivenRefStatistics(zref=zref,m=self.m,q=self.q,nl=self.nl,DEBUG=DEBUG)
        
        #Save all z statistics
        np.savez(self.pth+"\\zstats.npz",sm=dat[0],sc=dat[1]\
                 ,qua=dat[2],ma=dat[3],mi=dat[4],smMF=dat[5]\
                 ,scMF=dat[6],quaMF=dat[7],maMF=dat[8],miMF=dat[9]\
                 ,mse = dat[10])
            
        #Save all A statistics
        np.savez(self.pth+"\\astats.npz",de=dat[11],tde=dat[12]\
                 ,cl=dat[13],le=dat[14],se=dat[15],deMF=dat[16]\
                 ,tdeMF=dat[17],clMF=dat[18],leMF=dat[19],seMF=dat[20]\
                 ,sd=dat[21],ssd=dat[22])
        
        if DEBUG:
            np.savez(self.pth + "\\statsDEBUG.npz",at=dat[23],zt=dat[24]\
                     ,atMF=dat[25],ztMF=dat[26])
        return
    
    #load the z statistics
    def loadmCoupledZStatistics(self, op = 0):
        #Run a safety check
        self.safetyCheck(op)
        
        with np.load(self.pth+"\\zstats.npz") as dat:
            sm = dat['sm']
            sc = dat['sc']
            qua = dat['qua']
            ma = dat['ma']
            mi = dat['mi']
            smMF = dat['smMF']
            scMF = dat['scMF']
            quaMF = dat['quaMF']
            maMF = dat['maMF']
            miMF = dat['miMF']
            mse = dat['mse']
        
        return (sm,sc,qua,ma,mi,smMF,scMF,quaMF,maMF,miMF,mse)
    
    #load the a statistics
    def loadmCoupledAStatistics(self, op = 0):
        #Run a safety check
        self.safetyCheck(op)
        
        with np.load(self.pth+"\\astats.npz") as dat:
            de = dat['de']
            tde = dat['tde']
            cl = dat['cl']
            le = dat['le']
            se = dat['se']
            deMF = dat['deMF']
            tdeMF = dat['tdeMF']
            clMF = dat['clMF']
            leMF = dat['leMF']
            seMF = dat['seMF']
            sd = dat['sd']
            ssd = dat['ssd']
        
        return (de,tde,cl,le,se,deMF,tdeMF,clMF,leMF,seMF,sd,ssd)
            
    #Only works if debug was activated
    def loadmDEBUGstatistics(self, op = 0):
        #Run a safety check
        self.safetyCheck(op)
        
        with np.load(self.pth+"\\statsDEBUG.npz") as dat:
            at = dat['at']
            zt = dat['zt']
            atMF = dat['atMF']
            ztMF = dat['ztMF']
            
        return (at,zt,atMF,ztMF)