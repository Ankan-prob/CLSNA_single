# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:09:46 2024

@author: ylugn
"""
import numpy as np
import scipy as sp
import pickle as pk

###Pausing: Rerun the file to restart the program after each pause
###Functions that can pause will save data to temporary files which are accessed
###after the pause is complete.
###Set cont = False for a fresh implementation of a function, and cont = True to
###continue a paused implementation

class NParticle:
    ###Global Parameters
        #Network Parameters:    iv or ev value      Network/Link type
        #                               0               Erdos-Renyi
        #                               1      Exponential decay link function
        #                               2   Square Exponential decay link function
        #                               3           Logistic link function
        #                               4           Cutoff Link function
        #                               5 Conditional cutoff link function (ev only)
        
    #t0 value                                   Initial Condition
    #   0                                       Delta mass at 0
    #   1                                       i.i.d. Standard normal      
        
    #noi value                                  Noise
    #   0                                       i.i.d. Standard Gaussian Noise
    #   1                                       noiseless (noise = 0)

    def __init__(self,t0=1,noi = 0,n=7,T=5,gam=0.9,d=2,C=0.5,p=0.15,a=1,delt=1,dist=1,iv=2,ev=2):
        self.t0 = t0            #Initial conditions on z
        self.noi = noi          #Noise distribution
        self.n = n              #number of particles
        self.T = T              #Time to run simulation
        self.gam = gam          #Gamma      
        self.d = d              #Dimensions
        self.C = C              #Constant of decay
        self.iv = iv            #Choice of initial network
        self.ev = ev            #Choice of link function
        self.p = p              #network density if iv = 1 or ev = 1
        self.a = a              #offset for logistic link
        self.delt = delt        #impact of previous network
        self.dist = dist        #Distance for cutoff link function
    #Link function (t = 0) for the full matrix
    #z is a dxn matrix
    #Returns a flat vector that turns into a B matrix using sp.spatial.distance.squareform
    def A0fun(self,z):
        B = sp.spatial.distance.pdist(np.transpose(z))
        if self.iv == 0:
            B = self.p*np.ones(B.shape)
        elif self.iv == 1:
            B = np.exp(-self.C*B)
        elif self.iv == 2:
            B = np.exp(-self.C*(B**2))
        elif self.iv == 3:
            B = 1/(1 + np.exp(self.C*B - self.a))
        elif self.iv == 4:
            B = (B < self.dist)
        else:
            raise ValueError("self.iv has an invalid value.")
        return B
    
    #Link function (t > 0)
    #z is a dxn matrix, a is a nxn symmetric, 1-0 matrix
    #Returns a flat vector that turns into a B matrix using sp.spatial.distance.squareform
    def Bfun(self,a,z):
        aa = np.copy(a)
        np.fill_diagonal(aa,0)
        B = sp.spatial.distance.pdist(np.transpose(z))
        A = sp.spatial.distance.squareform(aa)
        if self.ev == 0:
            B = self.p*np.ones(B.shape)
        elif self.ev == 1:
            B = np.exp(self.delt*A-self.delt-self.C*B)
        elif self.ev == 2:
            B = np.exp(self.delt*A-self.delt-self.C*(B**2))
        elif self.ev == 3:
            B = 1/(1 + np.exp(self.C*B - self.a-self.delt*A))
        elif self.ev == 4:
            B = (B < self.dist)
        elif self.ev == 5:
            B = (B < self.dist) & (A > 0.5)
        else:
            raise ValueError("self.ev has an invalid value.")
        return B
    
    #Link function (t > 0)
    #z is a dxn matrix, a is a nxn symmetric, 1-0 matrix
    #Returns a flat vector that turns into a B matrix using sp.spatial.distance.squareform
    #Assume a is a sp.sparse.csr_matrix
    def BfunSparse(self,a,z):
        aa = np.copy(a.toarray())
        np.fill_diagonal(aa,0)
        B = sp.spatial.distance.pdist(np.transpose(z))
        A = sp.spatial.distance.squareform(aa)
        if self.ev == 0:
            B = self.p*np.ones(B.shape)
        elif self.ev == 1:
            B = np.exp(self.delt*A-self.delt-self.C*B)
        elif self.ev == 2:
            B = np.exp(self.delt*A-self.delt-self.C*(B**2))
        elif self.ev == 3:
            B = 1/(1 + np.exp(self.C*B - self.a-self.delt*A))
        elif self.ev == 4:
            B = (B < self.dist)
        elif self.ev == 5:
            B = (B < self.dist) & (A > 0.5)
        else:
            raise ValueError("self.ev has an invalid value.")
        return B
    
    #generates the adjacency matrix of a graph for t=0 given a dxn matrix representing Z
    def A0gen(self,z):
        B = self.A0fun(z)
        U = np.random.uniform(size = B.shape)     #Array of uniform random variables
        A = (B > U).astype(int)
        A = sp.spatial.distance.squareform(A)
        np.fill_diagonal(A,1)        
        return A
    
    #generates the adjacency matrix of a graph for t=0 given a dxn matrix representing Z
    #Preserves noise for coupling
    def A0gen_noise(self,z):
        B = self.A0fun(z)
        U = np.random.uniform(size = B.shape)     #Array of uniform random variables
        A = (B > U).astype(int)
        A = sp.spatial.distance.squareform(A)
        np.fill_diagonal(A,1)     
        return (A,U)

    #generates the adjacency matrix of a graph for t> 0 given a dxn matrix 
    #representing Z and a nxn adjacency matrix a
    def Bgen(self,a,z):
        B = self.Bfun(a,z)
        U = np.random.uniform(size = B.shape)     #Array of uniform random variables
        A = (B > U).astype(int)
        A = sp.spatial.distance.squareform(A)
        np.fill_diagonal(A,1)     
        return A
    
    #generates the adjacency matrix of a graph for t> 0 given a dxn matrix 
    #representing Z and a nxn adjacency matrix a
    #Preserves noise for coupling
    def Bgen_noise(self,a,z):
        B = self.Bfun(a,z)
        U = np.random.uniform(size = B.shape)     #Array of uniform random variables
        A = (B > U).astype(int)
        A = sp.spatial.distance.squareform(A)  
        np.fill_diagonal(A,1)     
        return (A,U)

    #Evolution
    def evolve(self,a,z):
        if self.noi == 0:
            noise = np.random.normal(size = (self.d,self.n))
        elif self.noi == 1:
            noise = np.zeros((self.d,self.n))
        else:
            raise ValueError("self.noi has an invalid value.")
        Lnum = np.matmul(z,a)
        Lden = np.matmul(np.ones(shape = (self.d,self.n)),a)
        L = np.true_divide(Lnum,Lden,where = (Lden!= 0))
        znew = (1 - self.gam)*z + self.gam*L + noise
        anew = self.Bgen(a,znew)
        return (anew,znew)
    
    #Evolution
    #Preserves noise for coupling
    def evolve_noise(self,a,z):
        if self.noi == 0:
            noise = np.random.normal(size = (self.d,self.n))
        elif self.noi == 1:
            noise = np.zeros((self.d,self.n))
        else:
            raise ValueError("self.noi has an invalid value.")
        Lnum = np.matmul(z,a)
        Lden = np.matmul(np.ones(shape = (self.d,self.n)),a)
        L = np.true_divide(Lnum,Lden,where = (Lden!= 0))
        znew = (1 - self.gam)*z + self.gam*L + noise
        anew,unew = self.Bgen_noise(a,znew)
        return (anew,znew,noise,unew)


    #Run the simulation
    def simulate(self):
        #Initialization
        if self.t0 == 0:
            Zcurrd = np.zeros((self.d,self.n))
        elif self.t0 == 1:
            Zcurrd = np.random.normal(size = (self.d,self.n))
        else:
            raise ValueError("self.t0 has an invalid value.")
        Acurrd = self.A0gen(Zcurrd)
    
        Ztraj = np.zeros((self.d,self.n,self.T))
        Ztraj[:,:,0] = Zcurrd
        Atraj = np.zeros((self.n,self.n,self.T))
        Atraj[:,:,0] = Acurrd
        
        #Iterate
        for t in np.arange(self.T-1):
            (Acurrd,Zcurrd) = self.evolve(Acurrd,Zcurrd);
            Ztraj[:,:,t+1] = Zcurrd
            Atraj[:,:,t+1] = Acurrd
            
        return (Atraj,Ztraj)
    
    #Run the simulation storing Atraj as sparse matrices
    def simulate_full_lowmem(self):
        #Initialization
        if self.t0 == 0:
            Zcurrd = np.zeros((self.d,self.n))
        elif self.t0 == 1:
            Zcurrd = np.random.normal(size = (self.d,self.n))
        else:
            raise ValueError("self.t0 has an invalid value.")
        Acurrd = self.A0gen(Zcurrd)
    
        Ztraj = np.zeros((self.d,self.n,self.T))
        Ztraj[:,:,0] = Zcurrd
        Atraj = {}
        Atraj[0] = sp.sparse.csr_matrix(Acurrd, dtype = bool)
        
        #Iterate
        for t in np.arange(self.T-1):
            (Acurrd,Zcurrd) = self.evolve(Acurrd,Zcurrd);
            Ztraj[:,:,t+1] = Zcurrd
            Atraj[t+1] = sp.sparse.csr_matrix(Acurrd)     
            if t % 100 == 0:
                print(t)
        return (Atraj,Ztraj)

    #Run the simulation
    #Preserves noise for coupling
    def simulate_noise(self):
        #Initialization
        M = int(self.n*(self.n-1)/2)
        
        if self.t0 == 0:
            Zcurrd = np.zeros((self.d,self.n))
        elif self.t0 == 1:
            Zcurrd = np.random.normal(size = (self.d,self.n))
        else:
            raise ValueError("self.t0 has an invalid value.")
        Acurrd,Ucurrd = self.A0gen_noise(Zcurrd)
    
        Ztraj = np.zeros((self.d,self.n,self.T))
        Ztraj[:,:,0] = Zcurrd
        Atraj = np.zeros((self.n,self.n,self.T))
        Atraj[:,:,0] = Acurrd
        Utraj = np.zeros((M,self.T))
        Utraj[:,0] = Ucurrd
        noisetraj = np.zeros((self.d,self.n,self.T-1))
        
        #Iterate
        for t in np.arange(self.T-1):
            Acurrd,Zcurrd,noisecurrd,Ucurrd = self.evolve_noise(Acurrd,Zcurrd);
            Ztraj[:,:,t+1] = Zcurrd
            Atraj[:,:,t+1] = Acurrd
            noisetraj[:,:,t] = noisecurrd
            Utraj[:,t+1] = Ucurrd
            
        return (Atraj,Ztraj,noisetraj,Utraj)
    
    #Run the simulation
    #Preserves noise for coupling
    #Minimizes memory
    def simulate_noise_lowmem(self):
        #Initialization
        M = int(self.n*(self.n-1)/2)
        
        if self.t0 == 0:
            Zcurrd = np.zeros((self.d,self.n))
        elif self.t0 == 1:
            Zcurrd = np.random.normal(size = (self.d,self.n))
        else:
            raise ValueError("self.t0 has an invalid value.")
        Acurrd,Ucurrd = self.A0gen_noise(Zcurrd)
    
        Ztraj = np.zeros((self.d,self.n,self.T))
        Ztraj[:,:,0] = Zcurrd
        Atraj = {}
        Atraj[0] = sp.sparse.csr_matrix(Acurrd, dtype = bool)
        Utraj = np.zeros((M,self.T))
        Utraj[:,0] = Ucurrd
        noisetraj = np.zeros((self.d,self.n,self.T-1))
        
        #Iterate
        for t in np.arange(self.T-1):
            Acurrd,Zcurrd,noisecurrd,Ucurrd = self.evolve_noise(Acurrd,Zcurrd);
            Ztraj[:,:,t+1] = Zcurrd
            Atraj[t+1] = sp.sparse.csr_matrix(Acurrd, dtype = bool)
            noisetraj[:,:,t] = noisecurrd
            Utraj[:,t+1] = Ucurrd
            
        return (Atraj,Ztraj,noisetraj,Utraj)
    
class MFParticle(NParticle):
    def __init__(self,t0=1,noi=0,n=7,T=5,gam=0.9,d=2,C=0.5,p=0.15,a=1,delt=1,dist=1,iv=2,ev=2,its=2):
        super().__init__(t0,noi,n,T,gam,d,C,p,a,delt,dist,iv,ev)
        self.its = its
        
    #Link function (t = 0) for the full matrix
    #zold is the old dxn matrix of z particles at time 0
    #znew is the new dxn matrix of z particles at time 0
    #Returns an nxn B matrix
    def newA0fun(self,zold,znew):
        B = sp.spatial.distance.cdist(np.transpose(zold),np.transpose(znew))
        if self.iv == 0:
            B = self.p*np.ones(B.shape)
        elif self.iv == 1:
            B = np.exp(-self.C*B)
        elif self.iv == 2:
            B = np.exp(-self.C*(B**2))
        elif self.iv == 3:
            B = 1/(1 + np.exp(self.C*B - self.a))
        elif self.iv == 4:
            B = (B < self.dist)
        else:
            raise ValueError("self.iv has an invalid value.")
        np.fill_diagonal(B,1)
        return B
        
    #Link function (t > 0) for the full matrix
    #zold is the old dxn matrix of z particles at time t
    #znew is the new dxn matrix of z particles at time t
    #b is the B matrix at time t-1
    #Returns an nxn B matrix
    def newAtfun(self,zold,znew,b):
        B = sp.spatial.distance.cdist(np.transpose(zold),np.transpose(znew))
        if B.shape != b.shape:
            raise Exception("B and b have mismatched shapes")
        A0 = np.zeros(b.shape)
        A1 = np.ones(b.shape)
        if self.ev == 0:
            B = self.p*np.ones(B.shape)
        elif self.ev == 1:
            B = b*np.exp(self.delt*A1-self.delt-self.C*B) + (1 - b)*np.exp(self.delt*A0-self.delt-self.C*B)
        elif self.ev == 2:
            B = b*np.exp(self.delt*A1-self.delt-self.C*(B**2)) + (1 - b)*np.exp(self.delt*A0-self.delt-self.C*(B**2))
        elif self.ev == 3:
            B = b/(1 + np.exp(self.C*B - self.a-self.delt*A1)) + (1-b)/(1 + np.exp(self.C*B - self.a-self.delt*A0))
        elif self.ev == 4:
            B = (B < self.dist)
        elif self.ev == 5:
            B = b*(B < self.dist)
        else:
            raise ValueError("self.ev has an invalid value.")
        np.fill_diagonal(B,1)
        return B
            
    #New evolve
    #zold is the old dxn matrix of z particles at time t-1
    #znew is the new dxn matrix of z particles at time t-1
    #zoldup is the old dxn matrix of z particles at time t
    #b is the B matrix at time t-1
    #Returns the new dxn matrix of z particles at time t
    #and the new nxn B matrix at time t
    def evolveMF(self,b,zold,znew,zoldup):
        if self.noi == 0:
            noise = np.random.normal(size = (self.d,self.n))
        elif self.noi == 1:
            noise = np.zeros((self.d,self.n))
        else:
            raise ValueError("self.noi has an invalid value.")
        Lnum = np.matmul(zold,b)
        Lden = np.matmul(np.ones(shape = zold.shape),b)
        L = np.true_divide(Lnum,Lden,where = (Lden!= 0))
        znew = (1 - self.gam)*znew + self.gam*L + noise
        bnew = self.newAtfun(zoldup,znew,b)
        return (bnew,znew)
        
    #Run one iteration: start with zoldtraj (dxnxT) and create a new z trajectory
    #No need to save the Bs. We can generate them later.
    def oneiter(self,zoldtraj):
        #Initialization
        if self.t0 == 0:
            Zcurrdnew = np.zeros((self.d,self.n))
        elif self.t0 == 1:
            Zcurrdnew = np.random.normal(size = (self.d,self.n))
        else:
            raise ValueError("self.t0 has an invalid value.")
        Bcurrd = self.newA0fun(zoldtraj[:,:,0], Zcurrdnew)
        
        Ztrajnew = np.zeros((self.d,self.n,self.T))
        Ztrajnew[:,:,0] = Zcurrdnew

        for t in np.arange(self.T-1):
            (Bcurrd,Zcurrdnew) = self.evolveMF(Bcurrd,zoldtraj[:,:,t],Zcurrdnew,zoldtraj[:,:,t+1])
            Ztrajnew[:,:,t+1] = Zcurrdnew
        
        return Ztrajnew

    #Run the simulation
    def simulateMF(self):
        #Ztraj = self.simulate()[1];
        Ztraj = self.randWalk();
        for i in np.arange(self.its - 1):
            Ztraj = self.oneiter(Ztraj)
            print(i)
        return Ztraj
    
    #Random walk simulation with a bias towards the origin
    def randWalk(self):
        #Initialization
        if self.t0 == 0:
            Zcurrd = np.zeros((self.d,self.n))
        elif self.t0 == 1:
            Zcurrd = np.random.normal(size = (self.d,self.n))
        else:
            raise ValueError("self.t0 has an invalid value.")
    
        Ztraj = np.zeros((self.d,self.n,self.T))
        Ztraj[:,:,0] = Zcurrd
        
        #Iterate
        for t in np.arange(self.T-1):
            if self.noi == 0:
                noise = np.random.normal(size = (self.d,self.n))
            elif self.noi == 1:
                noise = np.zeros((self.d,self.n))
            else:
                raise ValueError("self.noi has an invalid value.")
            Zcurrd = (1 - self.gam)*Zcurrd + noise    
            Ztraj[:,:,t+1] = Zcurrd
            
        return Ztraj
    
    #Run he simulation from Random Walk and store all iterations
    #If cont = False, this is an initialization. If cont = True, the program
    #is continuing after being paused.
    def simulateMFStoreIterations(self,cont):
        if not cont:
            Zits = np.zeros((self.d,self.n,self.T,self.its))
            Ztraj = self.randWalk()
            Zits[:,:,:,0] = Ztraj
            i = 0
        else:
            Zits = np.load("tempMFzits.npy")
            i = np.load("tempMFi.npy")
            MFParams = np.load("tempMFparams.npy")
            Ztraj = Zits[:,:,:,i]
            self.t0=int(MFParams[0])
            self.noi=int(MFParams[1])
            self.n=int(MFParams[2])
            self.T=int(MFParams[3])
            self.gam=MFParams[4]
            self.d=int(MFParams[5])
            self.C=MFParams[6]
            self.p=MFParams[7]
            self.a=MFParams[8]
            self.delt=MFParams[9]
            self.dist=MFParams[10]
            self.iv=int(MFParams[11])
            self.ev=int(MFParams[12])
            self.its=int(MFParams[13])
            
        while i < self.its-1:
            try:
                Ztraj = self.oneiter(Ztraj)
                Zits[:,:,:,i+1] = Ztraj
                i = i+1
                print(i)
            except KeyboardInterrupt:
                np.save("tempMFzits.npy",Zits)
                np.save("tempMFi.npy",i)
                MFParams = np.array([self.t0,self.noi,self.n,self.T,self.gam,
                                   self.d,self.C,self.p,self.a,self.delt,
                                   self.dist,self.iv,self.ev,self.its])
                np.save("tempMFparams.npy",MFParams)
                return 
        return Zits
    
    #Given a MF simulation, find all the relevant stats
    #ref is the file holding Zits
    #refparams is the file holding the parameters of Zits
    #def genStats(self, ref):
        
    
    
class CoupledParticle(MFParticle):
    
    #Same initialization as MFParticle
    #N represents the # of particles used to compute the MF empirical measure
    #SS is the sample size
    def __init__(self,N=12,SS=3,t0=1,noi=0,n=7,T=5,gam=0.9,d=2,C=0.5,p=0.15,\
                 a=1,delt=1,dist=1,iv=2,ev=2,its=2):
        super().__init__(t0,noi,n,T,gam,d,C,p,a,delt,dist,iv,ev,its)
        self.N = N
        self.SS = SS
        
    #Check that two instances of the class are equal
    def __eq__(self, other):
        t = np.array([False]*16,dtype = bool)
        t[0] = (self.N==other.N)
        t[1] = (self.SS==other.SS)
        t[2] = (self.t0==other.t0)
        t[3] = (self.noi==other.noi)
        t[4] = (self.n==other.n)
        t[5] = (self.T==other.T)
        t[6] = (self.gam==other.gam)
        t[7] = (self.d==other.d)
        t[8] = (self.C==other.C)
        t[9] = (self.p==other.p)
        t[10] = (self.a==other.a)
        t[11] = (self.delt==other.delt)
        t[12] = (self.dist==other.dist)
        t[13] = (self.iv==other.iv)
        t[14] = (self.ev==other.ev)
        t[15] = (self.its==other.its)
        return t.all()
    
    def __str__(self):
        strs = ['']*17
        strs[0] = "CoupledParticle object with parameters:\n"
        strs[1] = str(self.N)+"\n"
        strs[2] = str(self.SS) + "\n"
        strs[3] = str(self.t0) + "\n"
        strs[4] = str(self.noi) + "\n"
        strs[5] = str(self.n) + "\n"
        strs[6] = str(self.T) + "\n"
        strs[7] = str(self.gam) + "\n"
        strs[8] = str(self.d) + "\n"
        strs[9] = str(self.C) + "\n"
        strs[10] = str(self.p) + "\n"
        strs[11] = str(self.a) + "\n"
        strs[12] = str(self.delt) + "\n"
        strs[13] = str(self.dist) + "\n"
        strs[14] = str(self.iv) + "\n"
        strs[15] = str(self.ev) + "\n"
        strs[16] = str(self.its)
        return ''.join(strs)
        
    #Construct a reference MF system. Use N particles.
    def ConstructReference(self):
        ref = MFParticle(t0=self.t0,noi=self.noi,n=self.N,T=self.T,gam=self.gam,\
                         d=self.d,C=self.C,p=self.p,a=self.a,delt=self.delt,\
                         dist=self.dist,iv=self.iv,ev=self.ev,its=self.its)
        zref = ref.simulateMF()
        return zref
    
    #Construct mean MF process from ref (noiseless)
    def ConstructMeanMF(self,zref):
        #Create a single MFParticle object with no noise initialized at 0
        ref = MFParticle(t0=0,noi=1,n=1,T=self.T,gam=self.gam,\
                         d=self.d,C=self.C,p=self.p,a=self.a,delt=self.delt,\
                         dist=self.dist,iv=self.iv,ev=self.ev,its=self.its)
        #Run the MF model with zref as the reference measure (and no noise)
        zout = ref.oneiter(zref)
        return zout
    
    #zref and zrefup are dxN matrices (from the reference measure)
    #b is Nxn matrix
    #znew and noise are dxn matrices
    def coupledEvolveMF(self,b,zref,znew,zrefup,noise):
        Lnum = np.matmul(zref,b)
        Lden = np.matmul(np.ones(shape = zref.shape),b)
        L = np.true_divide(Lnum,Lden,where = (Lden!= 0))
        znew = (1 - self.gam)*znew + self.gam*L + noise
        bnew = self.newAtfun(zrefup,znew,b)
        return (bnew,znew)
    
    
    
    #Run an n particle coupled simulation
    #Takes as input a reference MRF process with N particles
    def CoupledSimulation(self, zref):
        
        #Construct a full n particle simulation with noise (Utraj not necessary)
        (Atraj,Ztraj,noisetraj,Utraj) = self.simulate_noise_lowmem()
        
        #Initialize MF at the same position as the n particle system
        Zcurrdnew = Ztraj[:,:,0]       
        Bcurrd = self.newA0fun(zref[:,:,0], Zcurrdnew)

        #Initialize MF trajectory
        ZMFtraj = np.zeros((self.d,self.n,self.T))
        AMFtraj = {}
        
        #Add time 0 information to trajectory
        ZMFtraj[:,:,0] = Zcurrdnew
        AMFtraj[0] = Atraj[0]
        
        for t in np.arange(self.T-1):
            (Bcurrd,Zcurrdnew) = self.coupledEvolveMF(Bcurrd,zref[:,:,t],Zcurrdnew,zref[:,:,t+1],noisetraj[:,:,t])
            ZMFtraj[:,:,t+1] = Zcurrdnew
            Bgrph = self.BfunSparse(AMFtraj[t],Zcurrdnew)
            
            #Get coupled graph
            Ucurr = Utraj[:,t+1]
            AMF = np.array((Ucurr < Bgrph))
            AMF = sp.spatial.distance.squareform(AMF)
            AMF = sp.sparse.csr_matrix(AMF,dtype = bool)
            AMFtraj[t+1] = AMF
        
        return (Atraj,Ztraj,AMFtraj,ZMFtraj)
    
    #Generate a coupled sample of size SS
    #TODO:: Move to another class and load data instead of running a simulation
    def CoupledSample(self, zref):
        
        #Run the coupled simulation
        (At,Zt,AMFt,ZMFt) = self.CoupledSimulation(zref)
        
        #Initialized U trajectories and A trajectories
        AMFtrajsamp = {}
        Atrajsamp = {}
        
        #Reduce to the sample
        Ztrajsamp = Zt[:,0:self.SS,self.T]
        ZMFtrajsamp = ZMFt[:,0:self.SS,self.T]
        
        for t in np.arange(self.T):           
            #Update MF network
            AMFtrajsamp[t] = AMFt[t][0:self.SS,0:self.SS]
            
            #Update N particle network sample
            Atrajsamp[t] = At[t][0:self.SS,0:self.SS]
            
        return (Atrajsamp,Ztrajsamp,AMFtrajsamp,ZMFtrajsamp)
    
    #Combine m independent simulations
    #Only outputs particles
    def CoupledSimulationsCombine(self,m, zref):
        
        Zs = np.zeros((self.d,self.n*m,self.T))
        ZMFs = np.zeros((self.d,self.n*m,self.T))
        
        for i in np.arange(m):
            #Run a coupled simulation
            (At,Zt,AMFt,ZMFt) = self.CoupledSimulation(zref)
            
            Zs[:,i*self.n:(i+1)*self.n,:] = Zt
            ZMFs[:,i*self.n:(i+1)*self.n,:] = ZMFt
            
        return (Zs,ZMFs)
        
    #When both m and n are large, it isn't reasonable to save everything
    #We instead only save important statistics
    #ideally, Zref has at least n*m particles. Otherwise Zref error may dominate
    #q = number of quantiles (q = 49 means we get 2%, 4% -- 98% quantiles)
    #nl = number of largest/smallest eigenvalues (normalized by n) tracked
    #For T = 100, m = 1000, q = 50, nl = 3 this works out to about 100MB of data
    #If debug is true, includes all particles and network trajectories
    #This is very data intensive for even moderately large parameters
    def mCoupledGivenRefStatistics(self,zref,m=12,q=49,nl=3,op=0,DEBUG=False): 
        
        #Raw data (if debug)
        if DEBUG:
            Atm = {}
            AtmMF = {}
            Ztm = np.zeros((self.d,self.n,self.T,m))
            ZtmMF = np.zeros((self.d,self.n,self.T,m))
        
        #Opinion statistics:
            #sm = sample means
            #sc = sample covariances
            #qua = Mahalanobis magnitude quantiles
            #ma = Mahalanobis magnitude maximum
            #mi = Mahalanobis magnitude minimum
        sm = np.zeros((self.d,self.T,m))                       #Sample means of particle systems
        sc = np.zeros((self.d,self.d,self.T,m))
        qua = np.zeros((q,self.T,m))
        ma = np.zeros((self.T,m))
        mi = np.zeros((self.T,m))
        smMF = np.zeros((self.d,self.T,m))                       #Sample means of particle systems
        scMF = np.zeros((self.d,self.d,self.T,m))
        quaMF = np.zeros((q,self.T,m))
        maMF = np.zeros((self.T,m))
        miMF = np.zeros((self.T,m))
        mse = np.zeros((self.T,m))
        
        #Network statistics
            #de = density
            #tde = triangle density
            #cl = clustering
            #le = largest nl eigenvalues normalized by n
            #se = smallest nl eigenvalues normalized by n
            #sd = #edges in symmetric difference of n-network and MF-network
            #ssd = #edges in n-network only - #edges in MF-network only
        de = np.zeros((self.T,m))
        tde = np.zeros((self.T,m))
        cl = np.zeros((self.T,m))
        le = np.zeros((nl,self.T,m))
        se = np.zeros((nl,self.T,m))
        deMF = np.zeros((self.T,m))
        tdeMF = np.zeros((self.T,m))
        clMF = np.zeros((self.T,m))
        leMF = np.zeros((nl,self.T,m))
        seMF = np.zeros((nl,self.T,m))
        sd = np.zeros((self.T,m))
        ssd = np.zeros((self.T,m))
        
        #Repeat m times
        for i in np.arange(m):
            #run the simulation
            (At,Zt,AMFt,ZMFt) = self.CoupledSimulation(zref)
            
            if DEBUG:
                Atm[i] = At
                AtmMF[i] = AMFt
                Ztm[:,:,:,i] = Zt
                ZtmMF[:,:,:,i] = ZMFt
            
            #Calculate z statistics
            mm = np.mean(Zt,axis=1)
            mmMF = np.mean(ZMFt,axis=1)
            #I compute covariance in a loop
            #I also compute the Mahalanobis metric in the loop
            cm = np.zeros((self.d,self.d,self.T))
            Mahaz = np.zeros((self.n,self.T))
            quam = np.zeros((q,self.T))
            
            cmMF = np.zeros((self.d,self.d,self.T))
            MahazMF = np.zeros((self.n,self.T))
            quamMF = np.zeros((q,self.T))
            
            #Compute Mahalanobis metric  
            #Step 1: Center the data (step 2 is in the loop)
            mmcent = Zt- np.tile(np.expand_dims(mm, axis=1),(1,self.n,1))
            mmMFcent = ZMFt- np.tile(np.expand_dims(mmMF, axis=1),(1,self.n,1))
            
            for t in np.arange(self.T):
                #compute covariance
                cm[:,:,t] = np.cov(Zt[:,:,t])
                cmMF[:,:,t] = np.cov(ZMFt[:,:,t])
                #invert the covariance metric
                cmi = np.linalg.inv(cm[:,:,t])
                cmiMF= np.linalg.inv(cmMF[:,:,t])
                
                #Compute Mahalanobis metric 
                #Step 2: Multiply. Each term should be roughly chi^2_d
                Mahaz[:,t] = np.diag(mmcent[:,:,t].T @ cmi @ mmcent[:,:,t])
                MahazMF[:,t] = np.diag(mmMFcent[:,:,t].T @ cmiMF @ mmMFcent[:,:,t])
                
                #get the quantiles
                quam[:,t] = np.quantile(Mahaz[:,t],np.arange(1/(q+1),1,1/(q+1)))
                quamMF[:,t] = np.quantile(MahazMF[:,t],np.arange(1/(q+1),1,1/(q+1)))
            
            #Get the max and min
            mam = np.max(Mahaz,axis = 0)
            mim = np.min(Mahaz,axis = 0)
            mamMF = np.max(MahazMF,axis = 0)
            mimMF = np.min(MahazMF,axis = 0)
            
            #Get the MSE
            msem = np.mean((Zt[0,:,:]-ZMFt[0,:,:])**2 + (Zt[1,:,:]-ZMFt[1,:,:])**2,axis=0)
            
            #Store these results
            sm[:,:,i] = mm                      
            sc[:,:,:,i] = cm
            qua[:,:,i] = quam
            ma[:,i] = mam
            mi[:,i] = mim
            smMF[:,:,i] = mmMF                     
            scMF[:,:,:,i] = cmMF
            quaMF[:,:,i] = quamMF
            maMF[:,i] = mamMF
            miMF[:,i] = mimMF
            mse[:,i] = msem
            
            #Calculate A statistics
            for t in np.arange(self.T):
                A = At[t]
                A = A.astype(int)
                A.setdiag(0)
                de[t,i] = A.sum()/(self.n**2)
                Asb = A @ A
                Asc = Asb @ A
                tde[t,i] = Asc.trace()/(self.n**3)
                cl[t,i] = tde[t,i]*(self.n**3)/(Asb.sum()-Asb.trace())
                eigs = sp.sparse.linalg.eigsh(A.asfptype(),k=6,which = 'BE',return_eigenvectors = False)
                se[:,t,i]= eigs[0:nl]
                le[:,t,i] = eigs[nl:2*nl]
                
                AMF = AMFt[t]
                AMF = AMF.astype(int)
                AMF.setdiag(0)
                deMF[t,i] = AMF.sum()/(self.n**2)
                AMFsb = AMF @ AMF
                AMFsc = AMFsb @ AMF
                tdeMF[t,i] = AMFsc.trace()/(self.n**3)
                clMF[t,i] = tdeMF[t,i]*(self.n**3)/(AMFsb.sum()-AMFsb.trace())
                eigsMF = sp.sparse.linalg.eigsh(AMF.asfptype(),k=6,which = 'BE',return_eigenvectors = False)
                seMF[:,t,i]= eigsMF[0:nl]
                leMF[:,t,i] = eigsMF[nl:2*nl]
                
                sd[t,i] = abs(A-AMF).sum() 
                ssd[t,i] = (A-AMF).sum()
                
            #Print the iteration number
            print("iteration: " + str(i))
            
        if DEBUG:
            return (sm,sc,qua,ma,mi,smMF,scMF,quaMF,maMF,miMF,de,tde,cl,\
                    le,se,deMF,tdeMF,clMF,leMF,seMF,sd,ssd,Atm,Ztm,AtmMF,ZtmMF)
        else:
            return (sm,sc,qua,ma,mi,smMF,scMF,quaMF,maMF,miMF,mse,de,tde,cl,\
                    le,se,deMF,tdeMF,clMF,leMF,seMF,sd,ssd)