# CLSNA_single
Single particle type CLSNA type model simulations coupled with mean field.

#Usage
See script.py for an example of usage. Data will be saved and stored inside a new directory in the simdata folder.\

###List of parameters (and default values)\
filebase (no default)         ##The directory in which all data is stored\
q=49                          ##Number of quantiles tracked (wrt Mahalanobis distance). q=49 means 2%-98% quantiles are recorded\
nl=3                          ##Number of upper/lower eigenvalues tracked
m=12                          ##Number of i.i.d. samples taken
N=12                          ##The number of particles in the mean-field reference process
SS=3                          ##Sample size for animations (number of particles animated)
t0=1                          ##Initial conditions:
                                      #t0=0: the particles are initialized at 0
                                      #t0=1: the particles are initialized as i.i.d. standard Gaussians
noi=0                         ##Noise distribution:
                                      #noi=0: standard Gaussian noise
n=7                           ##Number of particles
T=5                           ##Number of time steps we run the simulation
gam=0.9                       ##Gamma value in eqn
d=2                           ##Number of dimensions of the latent space
C=0.5                         ##Constant in link function (see iv/ev)
p=0.15                        ##Constant in link function (see iv/ev)
a=1                           ##Constant in link function (see iv/ev)
delt=1                        ##Constant in link function (see ev)
dist=1                        ##Constant in link function (see iv/ev)
iv=2                          ##Choice of link function (at time 0):
                                      #iv=0: B_0(x,y) = p
                                      #iv=1: B_0(x,y) = exp(-C|x-y|)
                                      #iv=2: B_0(x,y) = exp(-C|x-y|^2)
                                      #iv=3: B_0(x,y) = logistic(a-C*|x-y|)
                                      #iv=4: B_0(x,y) = 1_{|x-y|<dist}
ev=2                          ##Choice of link function (at time > 0):
                                      #ev=0: B(A,x,y) = p
                                      #ev=1: B(A,x,y) = exp(delt*(A-1)-C|x-y|)
                                      #ev=2: B(A,x,y) = exp(delt*(A-1)-C|x-y|^2)
                                      #ev=3: B(A,x,y) = logistic(a+delt*A-C|x-y|)
                                      #ev=4: B(A,x,y) = 1_{|x-y|<dist}
                                      #ev=5: B(A,x,y) = 1_{|x-y|<dist,A=1}
its=2                         ##Number of iterations to generate MF reference
rfilebase=filebase            ##directory from which the MF reference is loaded. By default, the MF reference is generated locally.
  d
