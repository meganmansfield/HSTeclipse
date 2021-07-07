# FUTURE MEGAN - THIS IS THE GOOD FILE 4/30/20
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import emcee
import pysynphot as S
import multiprocessing as mp

#Define parameters
Startemp=5605.
Starm=0.07
Starlogg=4.49
h=6.626*10**-34 #planck constant
c=2.998*10**8 #speed of light
kb=1.381*10**-23 #boltzmann constant
rp=0.13012	# planet-to-star radius ratio

#Upload data file
datfile=np.loadtxt('../EclipsesPaper/wasp77.txt')	#Fp/Fs*(Rp/Rs)^2
waves=datfile[:,0]*10.**-6
fp=datfile[:,1]*10.**-6
fperr=datfile[:,2]*10.**-6
diff=np.mean(np.diff(waves))	#wavelength size of each bin
diffarray=np.diff(waves)

def trapezoidint(xvals,yvals):
	total=0
	for i in np.arange(np.shape(xvals)[0]-1):
		total+=(yvals[i]+yvals[i+1])*(xvals[i+1]-xvals[i])*0.5
	return total

#Get stellar flux
sp = S.Icat('k93models',Startemp,Starm,Starlogg)	#Parameters go temp, metallicity, logg
#units of flam - erg/cm^2/s/Angstrom
sp.convert('photlam') ## initial units photosn/s/cm^2/A
wave=sp.wave*10.**-10  #in meters
photflux = sp.flux*10.**4*10.**10 #in photons/s/m^2/m
fluxmks = np.zeros(np.shape(photflux)[0])
for k in np.arange(np.shape(photflux)[0]):
	Ephoton=h*c/wave[k]
	fluxmks[k]=photflux[k]*Ephoton*10**7.	#in erg/s/m^2/m

masterwavegrid=np.linspace(1.0*10**-6.,2.0*10**-6.,2000)
cfluxmks=np.interp(masterwavegrid,wave,fluxmks)


#Integrate Phoenix model over each bandpass to get stellar flux
stellarfluxes=np.zeros(np.shape(waves)[0])
for i in np.arange(np.shape(waves)[0]):
	if i==0:
		wave1=waves[i]-diffarray[i]/2.
		wave2=waves[i]+diffarray[i]/2.
	elif i==np.shape(diffarray)[0]:
		wave1=waves[i]-diffarray[i-1]/2.
		wave2=waves[i]+diffarray[i-1]/2.
	else:
		wave1=waves[i]-diffarray[i-1]/2.
		wave2=waves[i]+diffarray[i]/2.
	wavevals=masterwavegrid[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
	starvals=cfluxmks[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
	stellarfluxes[i]=trapezoidint(wavevals,starvals)

def blackbodyintegrate(wavelength1,wavelength2,temp):
	total=0
	waveset=np.linspace(wavelength1,wavelength2,10)
	for i in np.arange(9):
		total+=(waveset[i+1]-waveset[i])*((2.0*h*(c**2)/((waveset[i+1])**5)/(np.exp(h*c/(waveset[i+1])/kb/temp)-1))+(2.0*h*(c**2)/((waveset[i])**5)/(np.exp(h*c/(waveset[i])/kb/temp)-1)))/2.*np.pi
	return total

def fakedata_bbod(wave,temp):
	#output units: erg/s/m^2/m
	Bbod=np.zeros(np.shape(wave)[0])
	for x in np.arange(np.shape(wave)[0]):
		#factor of pi to remove sterradians; factor of 10^7 converts from J to erg
		Bbod[x]=2*h*c**2./wave[x]**5./(np.exp(h*c/(wave[x]*kb*temp))-1.)*np.pi*10.**7

	return Bbod

#set up the initial least squares fit
def fluxfunct(p,x,y,err):
	Bday=np.zeros(np.shape(fp)[0])
	planetbbod=fakedata_bbod(masterwavegrid,p[0])
	for i in np.arange(np.shape(fp)[0]):
		if i==0:
			wave1=waves[i]-diffarray[i]/2.
			wave2=waves[i]+diffarray[i]/2.
		elif i==np.shape(diffarray)[0]:
			wave1=waves[i]-diffarray[i-1]/2.
			wave2=waves[i]+diffarray[i-1]/2.
		else:
			wave1=waves[i]-diffarray[i-1]/2.
			wave2=waves[i]+diffarray[i]/2.
		wavevals=masterwavegrid[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
		planetvals=planetbbod[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
		Bday[i]=trapezoidint(wavevals,planetvals)
	model=Bday/stellarfluxes*rp**2
	return np.array(y-model)

params0=np.array([1000.]) #Initial guess for albedo and temperature
m=leastsq(fluxfunct,params0,args=(waves,fp,fperr))

#format parameters for mcmc fit
theta=m[0]
fitparams=m[0]
ndim=np.shape(theta)[0]	#set number of dimensions
nwalkers=10 #number of walkers

def lnlike(theta,x,y,yerr):
	temp=theta
	modeledBday=np.zeros(np.shape(fp)[0])
	planetbbod=fakedata_bbod(masterwavegrid,temp)
	for i in np.arange(np.shape(fp)[0]):
		if i==0:
			wave1=waves[i]-diffarray[i]/2.
			wave2=waves[i]+diffarray[i]/2.
		elif i==np.shape(diffarray)[0]:
			wave1=waves[i]-diffarray[i-1]/2.
			wave2=waves[i]+diffarray[i-1]/2.
		else:
			wave1=waves[i]-diffarray[i-1]/2.
			wave2=waves[i]+diffarray[i]/2.
		wavevals=masterwavegrid[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
		planetvals=planetbbod[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
		modeledBday[i]=trapezoidint(wavevals,planetvals)
	modeledvals=modeledBday/stellarfluxes*rp**2
	resid=y-modeledvals
	chi2=np.sum((resid/yerr)**2)
	dof=np.shape(y)[0]-1.
	chi2red=chi2/dof
	ln_likelihood=-0.5*(np.sum((resid/yerr)**2 + np.log(2.0*np.pi*(yerr)**2)))
	return ln_likelihood

def lnprior(theta):
	lnpriorprob=0.
	temp=theta
	if temp<0.:
		lnpriorprob=-np.inf
	elif temp>6500.:
		lnpriorprob=-np.inf
	return lnpriorprob

def lnprob(theta,x,y,yerr):
	lp=lnprior(theta)
	return lp+lnlike(theta,x,y,yerr)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(waves,fp,fperr),pool=mp.Pool())
pos = [theta + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

burnin=20
nsteps=100

bestfit=np.zeros(ndim+1)
bestfit[0]=10.**8

def chi2(theta,x,y,yerr):
	temp=theta
	modeledBday=np.zeros(np.shape(fp)[0])
	planetbbod=fakedata_bbod(masterwavegrid,temp)
	for i in np.arange(np.shape(fp)[0]):
		if i==0:
			wave1=waves[i]-diffarray[i]/2.
			wave2=waves[i]+diffarray[i]/2.
		elif i==np.shape(diffarray)[0]:
			wave1=waves[i]-diffarray[i-1]/2.
			wave2=waves[i]+diffarray[i-1]/2.
		else:
			wave1=waves[i]-diffarray[i-1]/2.
			wave2=waves[i]+diffarray[i]/2.
		wavevals=masterwavegrid[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
		planetvals=planetbbod[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
		modeledBday[i]=trapezoidint(wavevals,planetvals)
	modeledvals=modeledBday/stellarfluxes*rp**2
	resid=y-modeledvals
	chi2=np.sum((resid/yerr)**2)
	dof=np.shape(y)[0]-1.
	chi2red=chi2/dof
	return chi2

for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
	print('step',i)
	if i>burnin:
		for guy in np.arange(nwalkers):
			chi2val=chi2(result.coords[guy],waves,fp,fperr)
			if chi2val<bestfit[0]:
				bestfit[0]=chi2val
				bestfit[1:]=result.coords[guy]

samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

def quantile(x, q):
	return np.percentile(x, [100. * qi for qi in q])

# calculate chi squared for fit
Terror=quantile(samples[:,0],[0.16, 0.5, 0.84])
fintemp=bestfit[1]
finmodeledBday=np.zeros(np.shape(fp)[0])
planetbbod=fakedata_bbod(masterwavegrid,fintemp)
for i in np.arange(np.shape(fp)[0]):
	if i==0:
		wave1=waves[i]-diffarray[i]/2.
		wave2=waves[i]+diffarray[i]/2.
	elif i==np.shape(diffarray)[0]:
		wave1=waves[i]-diffarray[i-1]/2.
		wave2=waves[i]+diffarray[i-1]/2.
	else:
		wave1=waves[i]-diffarray[i-1]/2.
		wave2=waves[i]+diffarray[i]/2.
	wavevals=masterwavegrid[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
	planetvals=planetbbod[(masterwavegrid>wave1)&(masterwavegrid<wave2)]
	finmodeledBday[i]=trapezoidint(wavevals,planetvals)
finmodeledvals=finmodeledBday/stellarfluxes*rp**2
finresid=fp-finmodeledvals
chi2=np.sum((finresid/fperr)**2)
chi2red=chi2/(np.shape(fp)[0]-1.) #divide by DoF

# create graph showing thermal and reflected contributions
thermal=finmodeledBday/stellarfluxes*rp**2

plt.figure()
plt.errorbar(waves*10**6.,fp*10**6.,yerr=fperr*10**6.,linestyle='none',color='k',marker='.')
plt.plot(waves*10**6.,thermal*10**6.,color='r',label='Blackbody Fit')
plt.legend()
plt.xlabel('Wavelength [microns]')
plt.ylabel('Fp/Fs [ppm]')
plt.show()

print('Mean Temperature = '+str("{:.3f}".format(fintemp))+' K')
print('1 Sigma Error = '+str("{:.3f}".format((Terror[2]-Terror[0])/2.))+' K')





