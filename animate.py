# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:25:45 2024

@author: ylugn
"""
import simulation as sm
import saveload as sl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import collections as mc
import numpy as np
import os

#This line is specific to my machine and necessary for plotting the animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\ylugn\\Desktop\\FFMpeg\\ffmpeg-7.0-essentials_build\\bin\\ffmpeg.exe'


class Animate():
    def __init__(self,filebase,q=49,nl=3,m=12):
        cp = sm.CoupledParticle()
        
        #load stored data (if filebase does not exist, creates an empty directory)
        self.ssl = sl.SimulateSaveLoad(filebase,op=1,q=q,nl=nl,m=m,cp=cp)
        
    #Construct an animation of the particle cloud
    def CoupledParticleCloud(self):
        
        #Don't redo if the video already exists
        if os.path.exists(self.ssl.pth + "\\pointcloud.mp4"):
            return
        
        #load the data
        (At,Zt,AMFt,ZMFt) = self.ssl.loadCoupled()
        
        #Get the maximal bounds of the plot
        maxlim = np.ceil(np.max(np.abs(Zt[:,:,:]))/10)*10
        maxlimMF = np.ceil(np.max(np.abs(ZMFt[:,:,:]))/10)*10

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlim(-maxlim,maxlim)
        axes[0].set_ylim(-maxlim,maxlim)
        axes[0].set_xlabel("x coordinate of particles")
        axes[0].set_ylabel("y coordinate of particles")
        axes[1].set_xlim(-maxlimMF,maxlimMF)
        axes[1].set_ylim(-maxlimMF,maxlimMF)
        axes[1].set_xlabel("x coordinate of particles")
        axes[1].set_ylabel("y coordinate of particles")

        #Initialize scatter plots
        sc0 = axes[0].scatter([],[],c='b',s=5, label = "particle cloud")
#        sc0m = axes[0].scatter([],[],c = 'black', label = "center of mass")
#        sc0r = axes[0].scatter([],[],c = 'red', label = "random particle")
        sc1 = axes[1].scatter([],[],c='b',s=5, label = "particle cloud")
 #       sc1m = axes[1].scatter([],[],c = 'black', label = "center of mass")
 #       sc1r = axes[1].scatter([],[],c = 'red', label = "particle cloud")

        #Add legends (looks terrible)
        #axes[0].legend()
        #axes[1].legend()

        #Create animation writer
        metadata = dict(title = 'Point Cloud', artist = 'Ankan Ganguly')
        writer = FFMpegWriter(fps = 15, metadata = metadata)
        
        with writer.saving(fig, self.ssl.pth + "\\pointcloud.mp4", 100):
            for t in np.arange(self.ssl.cp.T):
                #Update the title
                axes[0].set_title("1000 particle: t = "+str(t))
                axes[1].set_title("MF 1000 particles: t = "+str(t))
                
                #Update scatterplot data
                sc0.set_offsets(np.transpose(Zt[:,:,t]))
                #sc0m.set_offsets(np.mean(Zt[:,:,t],1))
                #sc0r.set_offsets(Zt[:,0,t])
                sc1.set_offsets(np.transpose(ZMFt[:,:,t]))
                #sc1m.set_offsets(np.mean(ZMFt[:,:,t],1))
                #sc1r.set_offsets(ZMFt[:,0,t])
                
                #Add frame to mp4 file
                writer.grab_frame()
        
        #Close plot
        plt.close('all')

    #Construct an animation of the particle cloud with center of mass and random particle
    def CoupledParticleCloudBR(self):
        
        #Don't redo if the video already exists
        if os.path.exists(self.ssl.pth + "\\pointcloudBR.mp4"):
            return
        
        #load the data
        (At,Zt,AMFt,ZMFt) = self.ssl.loadCoupled()
        
        #Get the maximal bounds of the plot
        maxlim = np.ceil(np.max(np.abs(Zt[:,:,:]))/10)*10
        maxlimMF = np.ceil(np.max(np.abs(ZMFt[:,:,:]))/10)*10

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlim(-maxlim,maxlim)
        axes[0].set_ylim(-maxlim,maxlim)
        axes[0].set_xlabel("x coordinate of particles")
        axes[0].set_ylabel("y coordinate of particles")
        axes[1].set_xlim(-maxlimMF,maxlimMF)
        axes[1].set_ylim(-maxlimMF,maxlimMF)
        axes[1].set_xlabel("x coordinate of particles")
        axes[1].set_ylabel("y coordinate of particles")

        #Initialize scatter plots
        sc0 = axes[0].scatter([],[],c='b',s=5, label = "particle cloud")
        sc0m = axes[0].scatter([],[],c = 'black', label = "center of mass")
        sc0r = axes[0].scatter([],[],c = 'red', label = "random particle")
        sc1 = axes[1].scatter([],[],c='b',s=5, label = "particle cloud")
        sc1m = axes[1].scatter([],[],c = 'black', label = "center of mass")
        sc1r = axes[1].scatter([],[],c = 'red', label = "particle cloud")

        #Add legends (looks terrible)
        #axes[0].legend()
        #axes[1].legend()

        #Create animation writer
        metadata = dict(title = 'A red particle in a point cloud with mean', artist = 'Ankan Ganguly')
        writer = FFMpegWriter(fps = 15, metadata = metadata)
        
        with writer.saving(fig, self.ssl.pth + "\\pointcloudBR.mp4", 100):
            for t in np.arange(self.ssl.cp.T):
                #Update the title
                axes[0].set_title("1000 particle: t = "+str(t))
                axes[1].set_title("MF 1000 particles: t = "+str(t))
                
                #Update scatterplot data
                sc0.set_offsets(np.transpose(Zt[:,:,t]))
                sc0m.set_offsets(np.mean(Zt[:,:,t],1))
                sc0r.set_offsets(Zt[:,0,t])
                sc1.set_offsets(np.transpose(ZMFt[:,:,t]))
                sc1m.set_offsets(np.mean(ZMFt[:,:,t],1))
                sc1r.set_offsets(ZMFt[:,0,t])
                
                #Add frame to mp4 file
                writer.grab_frame()
                
        #Close plot
        plt.close('all')
        
    #Construct an animation of the particle cloud with center of mass and random particle
    def CoupledParticleCloudR(self):
        
        #Don't redo if the video already exists
        if os.path.exists(self.ssl.pth + "\\pointcloudR.mp4"):
            return
        
        #load the data
        (At,Zt,AMFt,ZMFt) = self.ssl.loadCoupled()
        
        #Get the maximal bounds of the plot
        maxlim = np.ceil(np.max(np.abs(Zt[:,:,:]))/10)*10
        maxlimMF = np.ceil(np.max(np.abs(ZMFt[:,:,:]))/10)*10

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlim(-maxlim,maxlim)
        axes[0].set_ylim(-maxlim,maxlim)
        axes[0].set_xlabel("x coordinate of particles")
        axes[0].set_ylabel("y coordinate of particles")
        axes[1].set_xlim(-maxlimMF,maxlimMF)
        axes[1].set_ylim(-maxlimMF,maxlimMF)
        axes[1].set_xlabel("x coordinate of particles")
        axes[1].set_ylabel("y coordinate of particles")

        #Initialize scatter plots
        sc0 = axes[0].scatter([],[],c='b',s=5, label = "particle cloud")
        #sc0m = axes[0].scatter([],[],c = 'black', label = "center of mass")
        sc0r = axes[0].scatter([],[],c = 'red', label = "random particle")
        sc1 = axes[1].scatter([],[],c='b',s=5, label = "particle cloud")
        #sc1m = axes[1].scatter([],[],c = 'black', label = "center of mass")
        sc1r = axes[1].scatter([],[],c = 'red', label = "particle cloud")

        #Add legends (looks terrible)
        #axes[0].legend()
        #axes[1].legend()

        #Create animation writer
        metadata = dict(title = 'A red particle in a point cloud', artist = 'Ankan Ganguly')
        writer = FFMpegWriter(fps = 15, metadata = metadata)
        
        with writer.saving(fig, self.ssl.pth + "\\pointcloudR.mp4", 100):
            for t in np.arange(self.ssl.cp.T):
                #Update the title
                axes[0].set_title("1000 particle: t = "+str(t))
                axes[1].set_title("MF 1000 particles: t = "+str(t))
                
                #Update scatterplot data
                sc0.set_offsets(np.transpose(Zt[:,:,t]))
                #sc0m.set_offsets(np.mean(Zt[:,:,t],1))
                sc0r.set_offsets(Zt[:,0,t])
                sc1.set_offsets(np.transpose(ZMFt[:,:,t]))
                #sc1m.set_offsets(np.mean(ZMFt[:,:,t],1))
                sc1r.set_offsets(ZMFt[:,0,t])
                
                #Add frame to mp4 file
                writer.grab_frame()
                
        #Close plot
        plt.close('all')
        
    #Get an animation of Particles (given TT = animation time < T)
    def AnimateParticleSample(self,TT = None):
        
        #set input default value of TT to be the time of the simulation
        if TT is None:
            TT = self.ssl.cp.T
        
        #Don't redo if the video already exists
        if os.path.exists(self.ssl.pth + "\\particlesample.mp4"):
            return
        
        #load the data
        (At,Zt,AMFt,ZMFt) = self.ssl.loadCoupled()
        
        #Retrieve sample size/Animation time
        SS = self.ssl.cp.SS
        
        #Extract samples (assuming exchangeability)
        Zt = Zt[:,0:SS,:]
        ZMFt = ZMFt[:,0:SS,:]
        
        #Create another animation writer
        metadata = dict(title = 'Sample Particles', artist = 'Ankan Ganguly')
        writer = FFMpegWriter(fps = 0.75, metadata = metadata)

        #Define bounds
        bds = np.ceil(max(np.max(np.abs(Zt)),np.max(np.abs(ZMFt)))/4)*4

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlabel("x coordinate of particles")
        axes[0].set_ylabel("y coordinate of particles")
        axes[1].set_xlabel("x coordinate of particles")
        axes[1].set_ylabel("y coordinate of particles")
        axes[0].set_xlim(-bds,bds)
        axes[0].set_ylim(-bds,bds)
        axes[1].set_xlim(-bds,bds)
        axes[1].set_ylim(-bds,bds)

        #Initialize scatter plots
        sc0 = axes[0].scatter([],[],c='black',s=10)
        #gph0 = mc.LineCollection([],colors = 'blue', linewidths = 0.5)
        #axes[0].add_collection(gph0)
        sc1 = axes[1].scatter([],[],c='black',s=10)
        #gph1 = mc.LineCollection([],colors = 'blue', linewidths = 0.5)
        #axes[1].add_collection(gph1)

        with writer.saving(fig, self.ssl.pth + "\\particlesample.mp4", 100):
            for t in np.arange(TT):
                #Set title
                axes[0].set_title("random sample: t = "+str(t))
                axes[1].set_title("MF random sample: t = "+str(t))
                
                #Get the network we are sampling from and remove self-loop terms
                #A = Atraj[t][0:SS,0:SS]
                #A.setdiag(0)
                
                #Get the sampled opinions
                z = Zt[:,:,t]
                
                #update the scatter plot
                sc0.set_offsets(np.transpose(z))
                
                #Get the x/y coordinates of the graph endpoints
                #indices = A.nonzero()
                #zlx = z[0,indices[0]]
                #zly = z[1,indices[0]]
                #zrx = z[0,indices[1]]
                #zry = z[1,indices[1]]
                
                #Reshape into segment coordinates (double check, I coded this while tired)
                #zl = np.c_[zlx,zly]
                #zr = np.c_[zrx,zry]
                #zs = np.stack((zl,zr),axis=1)
                
                #Set segments
                #gph0.set_segments(zs)
                
                #Get the sampled opinions
                zmf = ZMFt[:,:,t]
                
                #Update the scatter plot
                sc1.set_offsets(np.transpose(zmf))
                
                #Get the x/y coordinates of the graph endpoints
                #indices = A.nonzero()
                #zmflx = zmf[0,indices[0]]
                #zmfly = zmf[1,indices[0]]
                #zmfrx = zmf[0,indices[1]]
                #zmfry = zmf[1,indices[1]]
                
                #Reshape into segment coordinates (double check, I coded this while tired)
                #zmfl = np.c_[zmflx,zmfly]
                #zmfr = np.c_[zmfrx,zmfry]
                #zmfs = np.stack((zmfl,zmfr),axis=1)
                
                #Set segments
                #gph1.set_segments(zmfs)

                #Add to animation
                writer.grab_frame()
                
        #close plots
        plt.close('all')
            
    #Get an animation of Network
    def AnimateSubNetwork(self, TT = None):
        #set input default value of TT to be the time of the simulation
        if TT is None:
            TT = self.ssl.cp.T
        
        #Don't redo if the video already exists
        if os.path.exists(self.ssl.pth + "\\subnetworksample.mp4"):
            return
        
        #load the data
        (At,Zt,AMFt,ZMFt) = self.ssl.loadCoupled()
        
        #Retrieve sample size/Animation time
        SS = self.ssl.cp.SS
        
        #Define bounds
        bds = 2

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlim(-bds,bds)
        axes[0].set_ylim(-bds,bds)
        axes[1].set_xlim(-bds,bds)
        axes[1].set_ylim(-bds,bds)
        
        #Turn off axes
        axes[0].axis("off")
        axes[1].axis("off")
        
        #Set points of network
        xs = np.sin(np.arange(SS)/SS*2*np.pi)
        ys = np.cos(np.arange(SS)/SS*2*np.pi)
        
        #Initialize scatter plots
        sc0 = axes[0].scatter(xs,ys,c='black',s=10)
        gph0 = mc.LineCollection([],colors = 'blue', linewidths = 0.5)
        axes[0].add_collection(gph0)
        sc1 = axes[1].scatter(xs,ys,c='black',s=10)
        gph1 = mc.LineCollection([],colors = 'blue', linewidths = 0.5)
        axes[1].add_collection(gph1)
        
        #Create another animation writer
        metadata = dict(title = 'Sample Subnetwork', artist = 'Ankan Ganguly')
        writer = FFMpegWriter(fps = 0.75, metadata = metadata)

        with writer.saving(fig, self.ssl.pth + "\\subnetworksample.mp4", 100):
            for t in np.arange(TT):
                #Set title
                axes[0].set_title("random network: t = "+str(t))
                axes[1].set_title("MF random network: t = "+str(t))
                
                #Get the network we are sampling from and remove self-loop terms
                A = At[t][0:SS,0:SS]
                A.setdiag(0)
                
                #Get the sampled opinions
                #z = Zt[:,:,t]
                
                #update the scatter plot
                #sc0.set_offsets(np.transpose(z))
                
                #Get the x/y coordinates of the graph endpoints
                indices = A.nonzero()
                zlx = xs[indices[0]]
                zly = ys[indices[0]]
                zrx = xs[indices[1]]
                zry = ys[indices[1]]
                
                #Reshape into segment coordinates (double check, I coded this while tired)
                zl = np.c_[zlx,zly]
                zr = np.c_[zrx,zry]
                zs = np.stack((zl,zr),axis=1)
                
                #Set segments
                gph0.set_segments(zs)
                
                #Get the sampled opinions
                #zmf = ZMFt[:,:,t]
                
                #Update the scatter plot
                #sc1.set_offsets(np.transpose(zmf))
                
                #Get the MF network we are sampling from and remove self-loop terms
                AMF = AMFt[t][0:SS,0:SS]
                AMF.setdiag(0)
                
                #Get the x/y coordinates of the graph endpoints
                indices = AMF.nonzero()
                zmflx = xs[indices[0]]
                zmfly = ys[indices[0]]
                zmfrx = xs[indices[1]]
                zmfry = ys[indices[1]]
                
                #Reshape into segment coordinates (double check, I coded this while tired)
                zmfl = np.c_[zmflx,zmfly]
                zmfr = np.c_[zmfrx,zmfry]
                zmfs = np.stack((zmfl,zmfr),axis=1)
                
                #Set segments
                gph1.set_segments(zmfs)

                #Add to animation
                writer.grab_frame()
                
        #close plots
        plt.close('all')
        
    #Get an animation of Particles (given TT = animation time < T)
    def AnimateParticleNetworkSample(self,TT=None):
        
        #set input default value of TT to be the time of the simulation
        if TT is None:
            TT = self.ssl.cp.T
        
        #Don't redo if the video already exists
        if os.path.exists(self.ssl.pth + "\\particlesamplenetwork.mp4"):
            return
        
        #load the data
        (At,Zt,AMFt,ZMFt) = self.ssl.loadCoupled()
        
        #Retrieve sample size/Animation time
        SS = self.ssl.cp.SS
        
        #Extract samples (assuming exchangeability)
        Zt = Zt[:,0:SS,:]
        ZMFt = ZMFt[:,0:SS,:]
        
        #Create another animation writer
        metadata = dict(title = 'Sample Particles with Network', artist = 'Ankan Ganguly')
        writer = FFMpegWriter(fps = 0.75, metadata = metadata)

        #Define bounds
        bds = np.ceil(max(np.max(np.abs(Zt)),np.max(np.abs(ZMFt)))/4)*4

        ##Create figure and axes for animation
        fig, axes = plt.subplots(1,2,layout = 'constrained')

        #Create axes
        axes[0].set_xlabel("x coordinate of particles")
        axes[0].set_ylabel("y coordinate of particles")
        axes[1].set_xlabel("x coordinate of particles")
        axes[1].set_ylabel("y coordinate of particles")
        axes[0].set_xlim(-bds,bds)
        axes[0].set_ylim(-bds,bds)
        axes[1].set_xlim(-bds,bds)
        axes[1].set_ylim(-bds,bds)

        #Initialize scatter plots
        sc0 = axes[0].scatter([],[],c='black',s=10)
        gph0 = mc.LineCollection([],colors = 'blue', linewidths = 0.5)
        axes[0].add_collection(gph0)
        sc1 = axes[1].scatter([],[],c='black',s=10)
        gph1 = mc.LineCollection([],colors = 'blue', linewidths = 0.5)
        axes[1].add_collection(gph1)

        with writer.saving(fig, self.ssl.pth + "\\particlesamplenetwork.mp4", 100):
            for t in np.arange(TT):
                #Set title
                axes[0].set_title("random sample: t = "+str(t))
                axes[1].set_title("MF random sample: t = "+str(t))
                
                #Get the network we are sampling from and remove self-loop terms
                A = At[t][0:SS,0:SS]
                A.setdiag(0)
                
                #Get the sampled opinions
                z = Zt[:,:,t]
                
                #update the scatter plot
                sc0.set_offsets(np.transpose(z))
                
                #Get the x/y coordinates of the graph endpoints
                indices = A.nonzero()
                zlx = z[0,indices[0]]
                zly = z[1,indices[0]]
                zrx = z[0,indices[1]]
                zry = z[1,indices[1]]
                
                #Reshape into segment coordinates (double check, I coded this while tired)
                zl = np.c_[zlx,zly]
                zr = np.c_[zrx,zry]
                zs = np.stack((zl,zr),axis=1)
                
                #Set segments
                gph0.set_segments(zs)
                
                #Get the sampled opinions
                zmf = ZMFt[:,:,t]
                
                #Update the scatter plot
                sc1.set_offsets(np.transpose(zmf))
                
                #Get the MF network we are sampling from and remove self-loop terms
                AMF = AMFt[t][0:SS,0:SS]
                AMF.setdiag(0)
                
                #Get the x/y coordinates of the graph endpoints
                indices = AMF.nonzero()
                zmflx = zmf[0,indices[0]]
                zmfly = zmf[1,indices[0]]
                zmfrx = zmf[0,indices[1]]
                zmfry = zmf[1,indices[1]]
                
                #Reshape into segment coordinates (double check, I coded this while tired)
                zmfl = np.c_[zmflx,zmfly]
                zmfr = np.c_[zmfrx,zmfry]
                zmfs = np.stack((zmfl,zmfr),axis=1)
                
                #Set segments
                gph1.set_segments(zmfs)

                #Add to animation
                writer.grab_frame()
                
        #close plots
        plt.close('all')