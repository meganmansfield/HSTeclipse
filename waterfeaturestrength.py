#UP TO DATE COPY 10/14/20
import pysynphot as S
import numpy as np 
import pdb
import matplotlib.pyplot as plt 
import pickle
from scipy.integrate import simps
from scipy.optimize import leastsq
import glob
from matplotlib import rc
import matplotlib.pylab as pl
import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
from scipy import stats
from scipy import special
import emcee
import multiprocessing as mp

#important constants
h=6.626*10.**-34
c=2.998*10.**8
kb=1.381*10.**-23

edgewave1=1.22
edgewave2=1.33
edgewave3=1.35
edgewave4=1.48
edgewave5=1.53
edgewave6=1.61
# edgewave1=1.18
# edgewave2=1.33
# edgewave3=1.33
# edgewave4=1.49
# edgewave5=1.49
# edgewave6=1.57

#the functions
def blackbodyintegrate(wavelength1,wavelength2,temp):
	#output units: erg/s/m^2
	total=0
	waveset=np.linspace(wavelength1,wavelength2,10)
	for i in np.arange(9):
		#factor of pi to remove sterradians; factor of 10^7 converts from J to erg
		total+=(waveset[i+1]-waveset[i])*((2.0*h*(c**2)/((waveset[i+1])**5)/(np.exp(h*c/(waveset[i+1])/kb/temp)-1))+(2.0*h*(c**2)/((waveset[i])**5)/(np.exp(h*c/(waveset[i])/kb/temp)-1)))/2.*np.pi*10.**7
	return total

def make_bbod(wave,diff,temp):
	#output units: erg/s/m^2
	Bbod=np.zeros(np.shape(wave)[0])
	for x in np.arange(np.shape(wave)[0]):
		Bbod[x]=blackbodyintegrate((wave[x]-diff/2.)*10**-6.,(wave[x]+diff/2.)*10.**-6,temp)

	return Bbod

def fakedata_bbod(wave,temp):
	#output units: erg/s/m^2/m
	Bbod=np.zeros(np.shape(wave)[0])
	for x in np.arange(np.shape(wave)[0]):
		#factor of pi to remove sterradians; factor of 10^7 converts from J to erg
		Bbod[x]=2*h*c**2./wave[x]**5./(np.exp(h*c/(wave[x]*kb*temp))-1.)*np.pi*10.**7

	return Bbod

def bbod2point(p,wave,diff,flux):
	#output units: erg/s/m^2
	Bbod=np.zeros(np.shape(wave)[0])
	for x in np.arange(np.shape(wave)[0]):
		Bbod[x]=blackbodyintegrate((wave[x]-diff[x]/2.)*10**-6.,(wave[x]+diff[x]/2.)*10.**-6,p[0])

	return np.array(flux-Bbod)

def trapezoidint(xvals,yvals):
	total=0
	for i in np.arange(np.shape(xvals)[0]-1):
		total+=(yvals[i]+yvals[i+1])*(xvals[i+1]-xvals[i])*0.5
	return total

def colormagmod(outpoint1,outpoint2,inpoint):
	outflux=np.array([outpoint1,outpoint2])
	wave=np.array([(edgewave2+edgewave1)/2.,(edgewave6+edgewave5)/2.])
	diff2=np.array([(edgewave2-edgewave1),(edgewave6-edgewave5)])
	params0=np.array([1000.])
	mpfit=leastsq(bbod2point,params0,args=(wave,diff2,outflux))
	fakeinflux=make_bbod(np.array([(edgewave4+edgewave3)/2.]),np.array([(edgewave4-edgewave3)]),mpfit[0][0])
	return mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(inpoint))

def colormagJHK(Jpoint,Hpoint,Kpoint):
	#1.250,1.635,2.120	0.16,0.29,0.34
	outflux=np.array([Hpoint,Kpoint])
	wave=np.array([1.250,1.635])
	diff2=np.array([0.16,0.29])
	params0=np.array([1000.])
	mpfit=leastsq(bbod2point,params0,args=(wave,diff2,outflux))
	fakeinflux=make_bbod(np.array([1.245]),np.array([0.17]),mpfit[0][0])
	#return mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(Jpoint))
	return mpfit[0][0],np.log10(Jpoint),np.log10(Jpoint)-np.log10(Hpoint)

def colormagdata(outpoint1,outpoint2,inpoint,meanerr,dellam,Fs,Fs2,rprs):
	outflux=np.array([outpoint1,outpoint2])
	wave=np.array([(edgewave2+edgewave1)/2.,(edgewave6+edgewave5)/2.])
	diff2=np.array([(edgewave2-edgewave1),(edgewave6-edgewave5)])
	params0=np.array([1000.])
	mpfit=leastsq(bbod2point,params0,args=(wave,diff2,outflux))
	fakeinflux=make_bbod(np.array([(edgewave4+edgewave3)/2.]),np.array([(edgewave4-edgewave3)]),mpfit[0][0])
	outerr1=meanerr*Fs[0]/rprs**2.*np.sqrt(dellam/(edgewave2-edgewave1))
	temp=outpoint1*(Fs[0]-Fs2[0])/Fs[0]
	couterr1=np.sqrt(outerr1**2.+temp**2.)
	outerr2=meanerr*Fs[1]/rprs**2.*np.sqrt(dellam/(edgewave6-edgewave5))
	temp=outpoint2*(Fs[1]-Fs2[1])/Fs[1]
	couterr2=np.sqrt(outerr2**2.+temp**2.)
	inerr=meanerr*Fs[2]/rprs**2.*np.sqrt(dellam/(edgewave4-edgewave3))
	temp=inpoint*(Fs[2]-Fs2[2])/Fs[2]
	cinerr=np.sqrt(inerr**2.+temp**2.)
	netoerr=np.sqrt(couterr1**2.+couterr2**2.)
	#print(couterr1,couterr2,inerr)
	magerr=np.log10((netoerr+fakeinflux)/fakeinflux)
	colorerr=np.sqrt(netoerr**2./(fakeinflux**2.*np.log(10.)**2.)+cinerr**2./(inpoint**2.*np.log(10.)**2.))
	return mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(inpoint)),magerr,colorerr

def colormagbd(outpoint1,outpoint2,inpoint,outerr1,outerr2,inerr):
	outflux=np.array([outpoint1,outpoint2])
	wave=np.array([(edgewave2+edgewave1)/2.,(edgewave6+edgewave5)/2.])
	diff2=np.array([(edgewave2-edgewave1),(edgewave6-edgewave5)])
	params0=np.array([1000.])
	mpfit=leastsq(bbod2point,params0,args=(wave,diff2,outflux))
	fakeinflux=make_bbod(np.array([(edgewave4+edgewave3)/2.]),np.array([(edgewave4-edgewave3)]),mpfit[0][0])
	netoerr=np.sqrt(outerr1**2.+outerr2**2.)
	magerr=np.log10((netoerr+fakeinflux)/fakeinflux)
	colorerr=np.sqrt(netoerr**2./(fakeinflux**2.*np.log(10.)**2.)+inerr**2./(inpoint**2.*np.log(10.)**2.))
	#pdb.set_trace()
	return mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(inpoint)),magerr,colorerr

#define the small spacing grid in wavelength
masterwavegrid=np.linspace(1.0,2.0,2000) #microns
masterwavegrid2=np.linspace(0.8,5.0,4000)

diff=np.diff(masterwavegrid)[0]
outset1=np.where((masterwavegrid>edgewave1)&(masterwavegrid<edgewave2))
outset2=np.where((masterwavegrid>edgewave5)&(masterwavegrid<edgewave6))
inset=np.where((masterwavegrid>edgewave3)&(masterwavegrid<edgewave4))

############################################## MIKE MODELS #######################################
#reading in all of Mike's models - base units W/m^2
templist=np.array([500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1650,1700,1750,1800,1850,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3200,3400,3600])
flist_fiducial=glob.glob('./a_new_new_grid/FIDUCIAL/*spec.txt')
flist_lowCO=glob.glob('./a_new_new_grid/CtoO/*0.01_spec.txt')
flist_highCO=glob.glob('./a_new_new_grid/CtoO/*0.85_spec.txt')
flist_delayTiO2000=glob.glob('./a_new_new_grid/TiO_VO_COLD_TRAP/*DELAY_TiO_VO_2000*spec.txt')
flist_delayTiO2500=glob.glob('./a_new_new_grid/TiO_VO_COLD_TRAP/*DELAY_TiO_VO_2500*spec.txt')
flist_delayTiO3000=glob.glob('./a_new_new_grid/TiO_VO_COLD_TRAP/*DELAY_TiO_VO_3000*spec.txt')
# flist_delayTiO2800=glob.glob('./a_new_grid_dump/DELAYED_TiO_VO/*DELAYED_TiO_VO_2800*spec.txt')
# flist_delayTiO3600=glob.glob('./a_new_grid_dump/DELAYED_TiO_VO/*DELAYED_TiO_VO_3600*spec.txt')
flist_grav20=glob.glob('./a_new_new_grid/LOGG/LOGG_2.0_*spec.txt')
flist_grav40=glob.glob('./a_new_new_grid/LOGG/LOGG_4.0_*spec.txt')
#flist_grav40=glob.glob('./a_new_grid_dump/GRAVITY/*LOGG_2.0_spec.txt')
flist_metneg15=glob.glob('./a_new_new_grid/METALICITY/*logZ_-1.5*spec.txt')
#flist_metpos1=glob.glob('./a_new_grid_dump/METALLICITY/*logZ_-1.5*spec.txt')
flist_metpos15=glob.glob('./a_new_new_grid/METALICITY/*logZ_+1.5*spec.txt')
# flist_star3300=glob.glob('./a_new_grid_dump/STELLAR_TEFF/*TSTAR_3300*spec.txt')
# flist_star4300=glob.glob('./a_new_grid_dump/STELLAR_TEFF/*TSTAR_4300*spec.txt')
# flist_star6300=glob.glob('./a_new_grid_dump/STELLAR_TEFF/*TSTAR_6300*spec.txt')
# flist_star7200=glob.glob('./a_new_grid_dump/STELLAR_TEFF/*TSTAR_7200*spec.txt')
# flist_star8200=glob.glob('./a_new_grid_dump/STELLAR_TEFF/*TSTAR_8200*spec.txt')
# flist_tint1percent=glob.glob('./a_new_grid_dump/TINT_TREND/*TINT_TRENDS_1percent_eff*spec.txt')
flist_tintTF18=glob.glob('./a_new_new_grid/TINT_TREND/*TF18_TINT*spec.txt')
flist_quench=glob.glob('./a_new_new_grid/CLOUDS_DISEQ/*NOCLOUD*spec.txt')
flist_fsedlow=glob.glob('./a_new_new_grid/CLOUDS_DISEQ/*fsed_0.1*spec.txt')
flist_fsedhigh=glob.glob('./a_new_new_grid/CLOUDS_DISEQ/*fsed_1.0*spec.txt')
flist_bd=glob.glob('./BD_Mods/FIDUCIAL/*spec.txt')
flist_bdmetpos1=glob.glob('./BD_Mods/logMet+1/*spec.txt')
flist_bdmetneg1=glob.glob('./BD_Mods/logMet-1/*spec.txt')
flist_bdlogg3=glob.glob('./BD_Mods/logg3/*spec.txt')
flist_bdlogg4=glob.glob('./BD_Mods/logg4/*spec.txt')

fpmods_fiducial=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_lowCO=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_highCO=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_delayTiO2000=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_delayTiO2500=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_delayTiO3000=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_delayTiO2800=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_delayTiO3600=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_grav20=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
#fpmods_grav35=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_grav40=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_metneg15=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
#fpmods_metpos1=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_metpos15=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_star3300=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_star4300=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_star6300=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_star7200=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_star8200=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
# fpmods_tint1percent=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_tintTF18=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_quench=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_fsedlow=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_fsedhigh=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_fiducial)[0]))
fpmods_bd=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_bd)[0]))
fpmods_bdmetpos1=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_bdmetpos1)[0]))
fpmods_bdmetneg1=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_bdmetneg1)[0]))
fpmods_bdlogg3=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_bdlogg3)[0]))
fpmods_bdlogg4=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist_bdlogg4)[0]))
bdtemplist=np.array([1000.,1200.,1400.,1600.,1800.,2000.,2200.,2400.,2600.,2800.])

longfiducial=np.zeros((np.shape(masterwavegrid2)[0],np.shape(flist_fiducial)[0]))
# fpmods_fiducialJHK=np.zeros((np.shape(masterwavegrid2)[0],np.shape(flist_fiducial)[0]))
#to plot Mike's models all pretty
tplist_fiducial=glob.glob('./a_grid_dump/FIDUCIAL/*TP_GAS.txt')
mikemods=np.zeros((1406,np.shape(flist_fiducial)[0]))
miketp=np.zeros((69,np.shape(flist_fiducial)[0]))

for f in flist_fiducial:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_fiducial[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
	longfiducial[:,placer]=np.interp(masterwavegrid2,mikewaves,mikefluxpl)
	# fpmods_fiducialJHK[:,placer]=np.interp(masterwavegrid2,mikewaves,mikefluxpl)
	mikemods[:,placer]=mike[:,1]*10**7.	#erg/s/m^2/m
for f in tplist_fiducial:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	miketp[:,placer]=mike[:,1]
	mikepressures=mike[:,0]
for f in flist_lowCO:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_lowCO[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_highCO:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_highCO[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_delayTiO2000:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_delayTiO2000[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_delayTiO2500:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_delayTiO2500[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_delayTiO3000:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_delayTiO3000[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_delayTiO2800:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_delayTiO2800[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_delayTiO3600:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_delayTiO3600[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_grav20:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_grav20[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_grav35:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_grav35[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_grav40:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_grav40[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_metneg15:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_metneg15[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_metpos1:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_metpos1[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_metpos15:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_metpos15[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_star3300:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_star3300[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_star4300:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_star4300[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_star6300:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_star6300[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_star7200:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_star7200[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_star8200:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_star8200[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
# for f in flist_tint1percent:
# 	mike=np.loadtxt(f)
# 	s1=f.split('_')
# 	inds1=np.where(np.array(s1)=='Tirr')[0]
# 	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
# 	mikewaves=mike[:,0]	#microns
# 	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
# 	fpmods_tint1percent[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_tintTF18:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_tintTF18[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_quench:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_quench[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_fsedlow:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_fsedlow[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_fsedhigh:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_fsedhigh[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_bd:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Teff')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-bdtemplist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_bd[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_bdmetpos1:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Teff')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-bdtemplist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_bdmetpos1[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_bdmetneg1:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Teff')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-bdtemplist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_bdmetneg1[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_bdlogg3:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Teff')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-bdtemplist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_bdlogg3[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)
for f in flist_bdlogg4:
	mike=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Teff')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-bdtemplist))
	mikewaves=mike[:,0]	#microns
	mikefluxpl=mike[:,1]*10**7.	#erg/s/m^2/m
	fpmods_bdlogg4[:,placer]=np.interp(masterwavegrid,mikewaves,mikefluxpl)

#plotting Mike's models
n=np.shape(templist)[0]
colors=pl.cm.plasma(np.linspace(0,0.9,n))

sp = S.Icat('k93models',5300.,0.0,4.0)	#Parameters go temp, metallicity, logg
sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
wave=sp.wave*10.**-10*10.**6.  #in microns
flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
cflux= np.interp(mikewaves,wave,flux)

# #make blackbodies to compare Mike's models to
# blackT=np.linspace(1000,3600,14)
# blackfpfs=np.zeros((np.shape(mikewaves)[0],np.shape(blackT)[0]))
# for i in np.arange(np.shape(blackT)[0]):
# 	blackfpfs[:,i]=fakedata_bbod(mikewaves*10**-6.,blackT[i])

#load in TP profiles for the gap
# W74fid=np.loadtxt('./thegapmods/WASP-074redist_trend_FIDUCIAL_redist_1.9636490392910262_logZ_+0.0_CtoO_0.55_TP_for_VUL.txt')
# W74noTiO=np.loadtxt('./thegapmods/WASP-074redist_trend_noTiV_redist_1.9636490392910262_logZ_+0.0_CtoO_0.55_TP_for_VUL.txt')
# W74highCO=np.loadtxt('./thegapmods/WASP-074redist_trend_CtO_redist_1.9636490392910262_logZ_+0.0_CtoO_1.0_TP_for_VUL.txt')
# W76fid=np.loadtxt('./thegapmods/WASP-076redist_trend_FIDUCIAL_redist_2.1176397024682183_logZ_+0.0_CtoO_0.55_TP_for_VUL.txt')
# W76noTiO=np.loadtxt('./thegapmods/WASP-076redist_trend_noTiV_redist_2.1176397024682183_logZ_+0.0_CtoO_0.55_TP_for_VUL.txt')
# W76highCO=np.loadtxt('./thegapmods/WASP-076redist_trend_CtO_redist_2.1176397024682183_logZ_+0.0_CtoO_1.0_TP_for_VUL.txt')

# rc('axes',linewidth=2)
# plt.figure(figsize=(12,7))
# for i in np.arange(np.shape(templist)[0]/2.)*2.:
# 	plt.plot(mikewaves,mikemods[:,int(i)],color=colors[int(i)],linewidth=3,zorder=1) #/cflux*0.1**2.
# 	# if i==20:
# 	# 	plt.plot(mikewaves,mikemods[:,int(i)],color='k',linewidth=12,alpha=0.3,zorder=0)
# 	# elif i==24:
# 	# 	plt.plot(mikewaves,mikemods[:,int(i)],color='k',linewidth=12,alpha=0.3,zorder=0)
# # for i in np.arange(np.shape(blackT)[0]):
# 	#plt.plot(mikewaves,blackfpfs[:,i]/cflux,color='k',linestyle='--',linewidth=2)
# plt.xlim((1.0,1.8))
# plt.axvspan(1.22,1.33,color='k',alpha=0.2,zorder=0)
# plt.axvspan(1.53,1.61,color='k',alpha=0.2,zorder=0)
# plt.axvspan(1.35,1.48,color='k',alpha=0.2,zorder=0)
# plt.yscale('log')
# # plt.ylim((4*10.**-10.,3*10**-3.))
# plt.ylim((1.*10.**13.,7.*10**19.))
# plt.xlabel('Wavelength [$\mu$m]',fontsize=20)
# plt.ylabel('Planet Flux [erg/s/m$^{3}$]',fontsize=20)
# # plt.ylabel('Eclipse Depth',fontsize=20)
# plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# #plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12,7))
# for i in np.arange(np.shape(templist)[0]/2.)*2.:
# 	plt.plot(miketp[:,int(i)],mikepressures,color=colors[int(i)],linewidth=3,zorder=1)
# 	# if i==20:
# 	# 	plt.plot(miketp[:,int(i)],mikepressures,color='k',linewidth=12,zorder=0,alpha=0.3)
# 	# elif i==24:
# 	# 	plt.plot(miketp[:,int(i)],mikepressures,color='k',linewidth=12,zorder=0,alpha=0.3)
# # plt.plot(W74fid[:,1],W74fid[:,0]*10**-6.,color='k',linewidth=3)
# # plt.plot(W74noTiO[:,1],W74noTiO[:,0]*10**-6.,color='k',linewidth=3)
# # plt.plot(W74highCO[:,1],W74highCO[:,0]*10**-6.,color='k',linewidth=3)
# # plt.plot(W76fid[:,1],W76fid[:,0]*10**-6.,color='g',linewidth=3)
# # plt.plot(W76noTiO[:,1],W76noTiO[:,0]*10**-6.,color='g',linewidth=3)
# # plt.plot(W76highCO[:,1],W76highCO[:,0]*10**-6.,color='g',linewidth=3)
# plt.xlim((500,4000))
# plt.yscale('log')
# plt.ylim((2*10.**2,5*10.**-4.))
# plt.xlabel('Temperature [K]',fontsize=20)
# plt.ylabel('Pressure [bar]',fontsize=20)
# plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# #plt.tight_layout()
# plt.show()

color_fiducial=np.zeros((np.shape(flist_fiducial)[0],3))
color_lowCO=np.zeros((np.shape(flist_lowCO)[0],3))
color_highCO=np.zeros((np.shape(flist_highCO)[0],3))
color_delayTiO2000=np.zeros((np.shape(flist_delayTiO2000)[0],3))
color_delayTiO2500=np.zeros((np.shape(flist_delayTiO2500)[0],3))
color_delayTiO3000=np.zeros((np.shape(flist_delayTiO3000)[0],3))
# color_delayTiO2800=np.zeros((np.shape(flist_delayTiO2800)[0],3))
# color_delayTiO3600=np.zeros((np.shape(flist_delayTiO3600)[0],3))
color_grav20=np.zeros((np.shape(flist_grav20)[0],3))
#color_grav35=np.zeros((np.shape(flist_grav35)[0],3))
color_grav40=np.zeros((np.shape(flist_grav40)[0],3))
color_metneg15=np.zeros((np.shape(flist_metneg15)[0],3))
#color_metpos1=np.zeros((np.shape(flist_metpos1)[0],3))
color_metpos15=np.zeros((np.shape(flist_metpos15)[0],3))
# color_star3300=np.zeros((np.shape(flist_star3300)[0],3))
# color_star4300=np.zeros((np.shape(flist_star4300)[0],3))
# color_star6300=np.zeros((np.shape(flist_star6300)[0],3))
# color_star7200=np.zeros((np.shape(flist_star7200)[0],3))
# color_star8200=np.zeros((np.shape(flist_star8200)[0],3))
# color_tint1percent=np.zeros((np.shape(flist_tint1percent)[0],3))
color_tintTF18=np.zeros((np.shape(flist_tintTF18)[0],3))
color_quench=np.zeros((np.shape(flist_quench)[0],3))
color_fsedlow=np.zeros((np.shape(flist_fsedlow)[0],3))
color_fsedhigh=np.zeros((np.shape(flist_fsedhigh)[0],3))
color_bd=np.zeros((np.shape(flist_bd)[0],3))
color_bdmetpos1=np.zeros((np.shape(flist_bdmetpos1)[0],3))
color_bdmetneg1=np.zeros((np.shape(flist_bdmetneg1)[0],3))
color_bdlogg3=np.zeros((np.shape(flist_bdlogg3)[0],3))
color_bdlogg4=np.zeros((np.shape(flist_bdlogg4)[0],3))

################ interpolate between models to get fake data
# W19temp=2372.
# W19planet=np.zeros(np.shape(masterwavegrid2)[0])
# for i in np.arange(np.shape(masterwavegrid2)[0]):
# 	W19planet[i]=np.interp(W19temp,color_fiducial[:,0],longfiducial[i,:])

# sp = S.Icat('k93models',5503.,0.21,4.40)
# sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
# wave=sp.wave*10.**-10  #in meters
# flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
# interp=np.interp(masterwavegrid2,wave*10**6.,flux)

# W19Fpfs=W19planet/interp*0.1409**2.

# plt.figure()
# plt.plot(masterwavegrid2,W19Fpfs*10**6.)
# plt.xlabel('Microns')
# plt.ylabel('Fp/Fs [ppm]')
# plt.show()

#testing out the JHK bands
# color_fiducialJHK=np.zeros((np.shape(flist_fiducial)[0],3))

# Jset=np.where((masterwavegrid2>1.17)&(masterwavegrid2<1.33))
# Hset=np.where((masterwavegrid2>1.49)&(masterwavegrid2<1.78))
# Kset=np.where((masterwavegrid2>1.95)&(masterwavegrid2<2.20))

for i in np.arange(np.shape(flist_fiducial)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_fiducial[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_fiducial[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_fiducial[:,i][inset])
	color_fiducial[i]=colormagmod(outpoint1,outpoint2,inpoint)
	# Jpoint=trapezoidint(masterwavegrid2[Jset]*10**-6.,fpmods_fiducialJHK[:,i][Jset])
	# Hpoint=trapezoidint(masterwavegrid2[Hset]*10**-6.,fpmods_fiducialJHK[:,i][Hset])
	# Kpoint=trapezoidint(masterwavegrid2[Kset]*10**-6.,fpmods_fiducialJHK[:,i][Kset])
	# color_fiducialJHK[i]=colormagJHK(Jpoint,Hpoint,Kpoint)
for i in np.arange(np.shape(flist_lowCO)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_lowCO[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_lowCO[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_lowCO[:,i][inset])
	color_lowCO[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_highCO)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_highCO[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_highCO[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_highCO[:,i][inset])
	color_highCO[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_delayTiO2000)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_delayTiO2000[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_delayTiO2000[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_delayTiO2000[:,i][inset])
	color_delayTiO2000[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_delayTiO2500)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_delayTiO2500[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_delayTiO2500[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_delayTiO2500[:,i][inset])
	color_delayTiO2500[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_delayTiO3000)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_delayTiO3000[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_delayTiO3000[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_delayTiO3000[:,i][inset])
	color_delayTiO3000[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_delayTiO2800)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_delayTiO2800[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_delayTiO2800[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_delayTiO2800[:,i][inset])
# 	color_delayTiO2800[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_delayTiO3600)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_delayTiO3600[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_delayTiO3600[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_delayTiO3600[:,i][inset])
# 	color_delayTiO3600[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_grav20)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_grav20[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_grav20[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_grav20[:,i][inset])
	color_grav20[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_grav35)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_grav35[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_grav35[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_grav35[:,i][inset])
# 	color_grav35[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_grav40)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_grav40[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_grav40[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_grav40[:,i][inset])
	color_grav40[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_metneg15)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_metneg15[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_metneg15[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_metneg15[:,i][inset])
	color_metneg15[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_metpos1)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_metpos1[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_metpos1[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_metpos1[:,i][inset])
	# color_metpos1[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_metpos15)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_metpos15[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_metpos15[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_metpos15[:,i][inset])
	color_metpos15[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_star3300)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_star3300[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_star3300[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_star3300[:,i][inset])
# 	color_star3300[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_star4300)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_star4300[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_star4300[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_star4300[:,i][inset])
# 	color_star4300[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_star6300)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_star6300[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_star6300[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_star6300[:,i][inset])
# 	color_star6300[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_star7200)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_star7200[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_star7200[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_star7200[:,i][inset])
# 	color_star7200[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_star8200)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_star8200[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_star8200[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_star8200[:,i][inset])
# 	color_star8200[i]=colormagmod(outpoint1,outpoint2,inpoint)
# for i in np.arange(np.shape(flist_tint1percent)[0]):
# 	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_tint1percent[:,i][outset1])
# 	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_tint1percent[:,i][outset2])
# 	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_tint1percent[:,i][inset])
# 	color_tint1percent[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_tintTF18)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_tintTF18[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_tintTF18[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_tintTF18[:,i][inset])
	color_tintTF18[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_quench)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_quench[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_quench[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_quench[:,i][inset])
	color_quench[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_fsedlow)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_fsedlow[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_fsedlow[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_fsedlow[:,i][inset])
	color_fsedlow[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_fsedhigh)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_fsedhigh[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_fsedhigh[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_fsedhigh[:,i][inset])
	color_fsedhigh[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_bd)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_bd[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_bd[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_bd[:,i][inset])
	color_bd[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_bdmetpos1)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_bdmetpos1[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_bdmetpos1[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_bdmetpos1[:,i][inset])
	color_bdmetpos1[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_bdmetneg1)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_bdmetneg1[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_bdmetneg1[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_bdmetneg1[:,i][inset])
	color_bdmetneg1[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_bdlogg3)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_bdlogg3[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_bdlogg3[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_bdlogg3[:,i][inset])
	color_bdlogg3[i]=colormagmod(outpoint1,outpoint2,inpoint)
for i in np.arange(np.shape(flist_bdlogg4)[0]):
	outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods_bdlogg4[:,i][outset1])
	outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods_bdlogg4[:,i][outset2])
	inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods_bdlogg4[:,i][inset])
	color_bdlogg4[i]=colormagmod(outpoint1,outpoint2,inpoint)

################################################## DATA SETS ##############################
#read in the data (13 data sets we will analyze) - data go wavelength, flux, flux err
C2=np.loadtxt('./EclipsesPaper/CoRoT2.txt') #ppm
H7=np.loadtxt('./EclipsesPaper/HAT7.txt') #ppm
H32=np.loadtxt('./EclipsesPaper/HAT32A.txt') #ppm
H41=np.loadtxt('./EclipsesPaper/hat41_new.txt') #percent
HD189=np.loadtxt('./EclipsesPaper/HD189733.txt') #ppm
HD209=np.loadtxt('./EclipsesPaper/HD209458.txt') #ppm
K7=np.loadtxt('./EclipsesPaper/kelt7_new.txt') #percent
Kep13=np.loadtxt('./EclipsesPaper/kep13_mostrecent.txt') #percent
T3=np.loadtxt('./EclipsesPaper/TrES3.txt') #ppm
W4=np.loadtxt('./EclipsesPaper/WASP4.txt') #ppm
W12=np.loadtxt('./EclipsesPaper/WASP12.txt') #ppm
W18=np.loadtxt('./EclipsesPaper/WASP18.txt') #ppm
W33=np.loadtxt('./EclipsesPaper/WASP33.txt') #ppm
W43=np.loadtxt('./EclipsesPaper/WASP43.txt') #ppm
W74=np.loadtxt('./EclipsesPaper/wasp74_new.txt') #percent
W76=np.loadtxt('./EclipsesPaper/WASP76_starcorr.txt') #percent
W79=np.loadtxt('./EclipsesPaper/wasp79_new.txt') #percent
W103=np.loadtxt('./EclipsesPaper/WASP103.txt') #ppm
W121=np.loadtxt('./EclipsesPaper/wasp121_allvis.txt') #percent
W77=np.loadtxt('./EclipsesPaper/wasp77.txt') #ppm



# W103night=np.loadtxt('WASP103_night.txt') #ppm
# W18night=np.loadtxt('WASP18_night.txt') #ppm
# W43night=np.loadtxt('WASP43_night.txt') #ppm

########### for proposals
W121v2=np.loadtxt('./EclipsesPaper/old_data/WASP121_MikalEvans.txt') #ppm
W76v2=np.loadtxt('./EclipsesPaper/old_data/WASP76_Edwards.txt') #ppm
W76v3=np.loadtxt('./EclipsesPaper/old_data/WASP76_Fu.txt') #ppm
Kep13v2=np.loadtxt('./EclipsesPaper/old_data/Kepler13A_Beatty.txt') #ppm
K7v2=np.loadtxt('./EclipsesPaper/old_data/KELT7_Pluriel.txt') #ppm

#downsample through interpolation to the master wavelength grid
H7down=np.interp(masterwavegrid,H7[:,0],H7[:,1]*10.**-6.)
H32down=np.interp(masterwavegrid,H32[:,0],H32[:,1]*10.**-6.)
H41down=np.interp(masterwavegrid,H41[:,0],H41[:,1]*10.**-2.)
# H41downv2=np.interp(masterwavegrid,H41v2[:,0],H41v2[:,1]*10.**-2.)
HD189down=np.interp(masterwavegrid,HD189[:,0],HD189[:,1]*10.**-6.)
HD209down=np.interp(masterwavegrid,HD209[:,0],HD209[:,1]*10.**-6.)
K7down=np.interp(masterwavegrid,K7[:,0],K7[:,1]*10.**-2.)
# K7downv2=np.interp(masterwavegrid,K7v2[:,0],K7v2[:,1]*10.**-2.)
#Kep13down=np.interp(masterwavegrid,Kep13[:,0],Kep13[:,1]*10.**-6.)
W18down=np.interp(masterwavegrid,W18[:,0],W18[:,1]*10.**-6.)
W33down=np.interp(masterwavegrid,W33[:,0],W33[:,1]*10.**-6.)
W43down=np.interp(masterwavegrid,W43[:,0],W43[:,1]*10.**-6.)
W74down=np.interp(masterwavegrid,W74[:,0],W74[:,1]*10.**-2.)
W76down=np.interp(masterwavegrid,W76[:,0],W76[:,1]*10.**-2.)
W79down=np.interp(masterwavegrid,W79[:,0],W79[:,1]*10.**-2.)
W103down=np.interp(masterwavegrid,W103[:,0],W103[:,1]*10.**-6.)
W121down=np.interp(masterwavegrid,W121[:,0],W121[:,1]*10.**-2.)
W77down=np.interp(masterwavegrid,W77[:,0],W77[:,1]*10.**-6.)

W76downv2=np.interp(masterwavegrid,W76v2[:,0],W76v2[:,1]*10.**-6.)
W76downv3=np.interp(masterwavegrid,W76v3[:,0],W76v3[:,1]*10.**-6.)
W121downv2=np.interp(masterwavegrid,W121v2[:,0],W121v2[:,1]*10.**-6.)
Kep13downv2=np.interp(masterwavegrid,Kep13v2[:,0],Kep13v2[:,1]*10.**-6.)
K7downv2=np.interp(masterwavegrid,K7v2[:,0],K7v2[:,1]*10.**-6.)

# W103nightdown=np.interp(masterwavegrid,W103night[:,0],W103night[:,1]*10.**-6.)
# W18nightdown=np.interp(masterwavegrid,W18night[:,0],W18night[:,1]*10.**-6.)
# W43nightdown=np.interp(masterwavegrid,W43night[:,0],W43night[:,1]*10.**-6.)

C2down=np.interp(masterwavegrid,C2[:,0],C2[:,1]*10.**-6.)
Kep13down=np.interp(masterwavegrid,Kep13[:,0],Kep13[:,1]*10.**-2.)
# Kep13down=np.interp(masterwavegrid,Kep13[:,0],Kep13[:,1]*10.**-6.)
T3down=np.interp(masterwavegrid,T3[:,0],T3[:,1]*10.**-6.)
W4down=np.interp(masterwavegrid,W4[:,0],W4[:,1]*10.**-6.)
W12down=np.interp(masterwavegrid,W12[:,0],W12[:,1]*10.**-6.)

#average in vs. out of band stuff
outpoint1H7=np.mean(H7down[outset1])
outpoint2H7=np.mean(H7down[outset2])
inpointH7=np.mean(H7down[inset])
outpoint1H32=np.mean(H32down[outset1])
outpoint2H32=np.mean(H32down[outset2])
inpointH32=np.mean(H32down[inset])
outpoint1H41=np.mean(H41down[outset1])
outpoint2H41=np.mean(H41down[outset2])
inpointH41=np.mean(H41down[inset])
outpoint1HD189=np.mean(HD189down[outset1])
outpoint2HD189=np.mean(HD189down[outset2])
inpointHD189=np.mean(HD189down[inset])
outpoint1HD209=np.mean(HD209down[outset1])
outpoint2HD209=np.mean(HD209down[outset2])
inpointHD209=np.mean(HD209down[inset])
outpoint1K7=np.mean(K7down[outset1])
outpoint2K7=np.mean(K7down[outset2])
inpointK7=np.mean(K7down[inset])
outpoint1W18=np.mean(W18down[outset1])
outpoint2W18=np.mean(W18down[outset2])
inpointW18=np.mean(W18down[inset])
outpoint1W33=np.mean(W33down[outset1])
outpoint2W33=np.mean(W33down[outset2])
inpointW33=np.mean(W33down[inset])
outpoint1W43=np.mean(W43down[outset1])
outpoint2W43=np.mean(W43down[outset2])
inpointW43=np.mean(W43down[inset])
outpoint1W74=np.mean(W74down[outset1])
outpoint2W74=np.mean(W74down[outset2])
inpointW74=np.mean(W74down[inset])
outpoint1W76=np.mean(W76down[outset1])
outpoint2W76=np.mean(W76down[outset2])
inpointW76=np.mean(W76down[inset])
outpoint1W79=np.mean(W79down[outset1])
outpoint2W79=np.mean(W79down[outset2])
inpointW79=np.mean(W79down[inset])
outpoint1W103=np.mean(W103down[outset1])
outpoint2W103=np.mean(W103down[outset2])
inpointW103=np.mean(W103down[inset])
outpoint1W121=np.mean(W121down[outset1])
outpoint2W121=np.mean(W121down[outset2])
inpointW121=np.mean(W121down[inset])
outpoint1W77=np.mean(W77down[outset1])
outpoint2W77=np.mean(W77down[outset2])
inpointW77=np.mean(W77down[inset])

outpoint1W76v2=np.mean(W76downv2[outset1])
outpoint2W76v2=np.mean(W76downv2[outset2])
inpointW76v2=np.mean(W76downv2[inset])
outpoint1W76v3=np.mean(W76downv3[outset1])
outpoint2W76v3=np.mean(W76downv3[outset2])
inpointW76v3=np.mean(W76downv3[inset])
outpoint1W121v2=np.mean(W121downv2[outset1])
outpoint2W121v2=np.mean(W121downv2[outset2])
inpointW121v2=np.mean(W121downv2[inset])
outpoint1K7v2=np.mean(K7downv2[outset1])
outpoint2K7v2=np.mean(K7downv2[outset2])
inpointK7v2=np.mean(K7downv2[inset])
outpoint1Kep13v2=np.mean(Kep13downv2[outset1])
outpoint2Kep13v2=np.mean(Kep13downv2[outset2])
inpointKep13v2=np.mean(Kep13downv2[inset])
# outpoint1H41v2=np.mean(H41downv2[outset1])
# outpoint2H41v2=np.mean(H41downv2[outset2])
# inpointH41v2=np.mean(H41downv2[inset])

# outpoint1W103night=np.mean(W103nightdown[outset1])
# outpoint2W103night=np.mean(W103nightdown[outset2])
# inpointW103night=np.mean(W103nightdown[inset])
# outpoint1W18night=np.mean(W18nightdown[outset1])
# outpoint2W18night=np.mean(W18nightdown[outset2])
# inpointW18night=np.mean(W18nightdown[inset])
# outpoint1W43night=np.mean(W43nightdown[outset1])
# outpoint2W43night=np.mean(W43nightdown[outset2])
# inpointW43night=np.mean(W43nightdown[inset])

outpoint1C2=np.mean(C2down[outset1])
outpoint2C2=np.mean(C2down[outset2])
inpointC2=np.mean(C2down[inset])
outpoint1Kep13=np.mean(Kep13down[outset1])
outpoint2Kep13=np.mean(Kep13down[outset2])
inpointKep13=np.mean(Kep13down[inset])
outpoint1T3=np.mean(T3down[outset1])
outpoint2T3=np.mean(T3down[outset2])
inpointT3=np.mean(T3down[inset])
outpoint1W4=np.mean(W4down[outset1])
outpoint2W4=np.mean(W4down[outset2])
inpointW4=np.mean(W4down[inset])
outpoint1W12=np.mean(W12down[outset1])
outpoint2W12=np.mean(W12down[outset2])
inpointW12=np.mean(W12down[inset])

def getcolor(Teff,Tserr,met,logg,rprs,outpoint1,outpoint2,inpoint,meanerr,dellam):
	sp = S.Icat('k93models',Teff,met,logg)
	sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave=sp.wave*10.**-10  #in meters
	flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
	interp=np.interp(masterwavegrid,wave*10**6.,flux)
	starout1=trapezoidint(masterwavegrid[outset1]*10.**-6.,interp[outset1])	#unit erg/s/m^2
	starout2=trapezoidint(masterwavegrid[outset2]*10.**-6.,interp[outset2])
	starin=trapezoidint(masterwavegrid[inset]*10.**-6.,interp[inset])
	sp2 = S.Icat('k93models',Teff+Tserr,met,logg)
	sp2.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave2=sp2.wave*10.**-10  #in meters
	flux2 = sp2.flux*10.**4*10.**10 #in erg/m2/m/s
	interp2=np.interp(masterwavegrid,wave2*10**6.,flux2)
	starout12=trapezoidint(masterwavegrid[outset1]*10.**-6.,interp2[outset1])	#unit erg/s/m^2
	starout22=trapezoidint(masterwavegrid[outset2]*10.**-6.,interp2[outset2])
	starin2=trapezoidint(masterwavegrid[inset]*10.**-6.,interp2[inset])
	Fpout1=outpoint1*starout1/rprs**2.	#unit erg/s/m^2
	Fpout2=outpoint2*starout2/rprs**2.
	Fpin=inpoint*starin/rprs**2.
	#print(Fpout1,Fpout2,Fpin)
	Fs=np.array([starout1,starout2,starin])
	Fs2=np.array([starout12,starout22,starin2])
	color=colormagdata(Fpout1,Fpout2,Fpin,meanerr,dellam,Fs,Fs2,rprs)
	return color

meanerrH7=np.mean(H7[:,2]*10.**-6.)
meanerrH32=np.mean(H32[:,2]*10.**-6.)
meanerrH41=np.mean(H41[:,2]*10.**-2.)
meanerrHD189=np.mean(HD189[:,2]*10.**-6.)
meanerrHD209=np.mean(HD209[:,2]*10.**-6.)
meanerrK7=np.mean(K7[:,2]*10.**-2.)
meanerrW18=np.mean(W18[:,2]*10.**-6.)
meanerrW33=np.mean(W33[:,2]*10.**-6.)
meanerrW43=np.mean(W43[:,2]*10.**-6.)
meanerrW74=np.mean(W74[:,2]*10.**-2.)
meanerrW76=np.mean(W76[:,2]*10.**-2.)
meanerrW79=np.mean(W79[:,2]*10.**-2.)
meanerrW103=np.mean(W103[:,2]*10.**-6.)
meanerrW121=np.mean(W121[:,2]*10.**-2.)
meanerrW77=np.mean(W77[:,2]*10.**-6.)

meanerrC2=np.mean(C2[:,2]*10.**-6.)
meanerrKep13=np.mean(Kep13[:,2]*10.**-2.)
# meanerrKep13=np.mean(Kep13[:,2]*10.**-6.)
meanerrT3=np.mean(T3[:,2]*10.**-6.)
meanerrW4=np.mean(W4[:,2]*10.**-6.)
meanerrW12=np.mean(W12[:,2]*10.**-6.)

meanerrW121v2=np.mean(W121v2[:,2]*10**-6.)
meanerrW76v2=np.mean(W76v2[:,2]*10**-6.)
meanerrW76v3=np.mean(W76v3[:,2]*10**-6.)
meanerrKep13v2=np.mean(Kep13v2[:,2]*10**-6.)
meanerrK7v2=np.mean(K7v2[:,2]*10**-6.)

dellamH7=np.mean(np.diff(H7[:,0]))
dellamH32=np.mean(np.diff(H32[:,0]))
dellamH41=np.mean(np.diff(H41[:,0]))
dellamHD189=np.mean(np.diff(HD189[:,0]))
dellamHD209=np.mean(np.diff(HD209[:,0]))
dellamK7=np.mean(np.diff(K7[:,0]))
dellamW18=np.mean(np.diff(W18[:,0]))
dellamW33=np.mean(np.diff(W33[:,0]))
dellamW43=np.mean(np.diff(W43[:,0]))
dellamW74=np.mean(np.diff(W74[:,0]))
dellamW76=np.mean(np.diff(W76[:,0]))
dellamW79=np.mean(np.diff(W79[:,0]))
dellamW103=np.mean(np.diff(W103[:,0]))
dellamW121=np.mean(np.diff(W121[:,0]))
dellamW77=np.mean(np.diff(W77[:,0]))

dellamC2=np.mean(np.diff(C2[:,0]))
dellamKep13=np.mean(np.diff(Kep13[:,0]))
dellamT3=np.mean(np.diff(T3[:,0]))
dellamW4=np.mean(np.diff(W4[:,0]))
dellamW12=np.mean(np.diff(W12[:,0]))

dellamW121v2=np.mean(np.diff(W121v2[:,0]))
dellamW76v2=np.mean(np.diff(W76v2[:,0]))
dellamW76v3=np.mean(np.diff(W76v3[:,0]))
dellamKep13v2=np.mean(np.diff(Kep13v2[:,0]))
dellamK7v2=np.mean(np.diff(K7v2[:,0]))

#param order: Teff, Tserr, met, logg, rprs, outpoint1, outpoint2, inpoint, meanerr, dellam
colorH7=getcolor(6441.,69.,0.15,4.02,0.07809,outpoint1H7,outpoint2H7,inpointH7,meanerrH7,dellamH7)
colorH32=getcolor(6207.,88.,-0.04,4.33,0.1478,outpoint1H32,outpoint2H32,inpointH32,meanerrH32,dellamH32)
colorH41=getcolor(6390.,100.,0.21,4.14,0.1028,outpoint1H41,outpoint2H41,inpointH41,meanerrH41,dellamH41)
colorHD189=getcolor(5111.,77.,-0.04,4.59,0.1514,outpoint1HD189,outpoint2HD189,inpointHD189,meanerrHD189,dellamHD189)
colorHD209=getcolor(6092.,103.,0.0,4.28,0.1174,outpoint1HD209,outpoint2HD209,inpointHD209,meanerrHD209,dellamHD209)
colorK7=getcolor(6789.,50.,0.139,4.149,0.0888,outpoint1K7,outpoint2K7,inpointK7,meanerrK7,dellamK7)
colorW18=getcolor(6368.,66.,0.11,4.37,0.0935,outpoint1W18,outpoint2W18,inpointW18,meanerrW18,dellamW18)
colorW33=getcolor(7430.,100.,0.1,4.3,0.1037,outpoint1W33,outpoint2W33,inpointW33,meanerrW33,dellamW33)
colorW43=getcolor(4520.,120.,-0.01,4.645,0.1558,outpoint1W43,outpoint2W43,inpointW43,meanerrW43,dellamW43)
colorW74=getcolor(5990.,110.,0.39,4.39,0.09803,outpoint1W74,outpoint2W74,inpointW74,meanerrW74,dellamW74)
colorW76=getcolor(6250.,100.,0.23,4.128,0.10873,outpoint1W76,outpoint2W76,inpointW76,meanerrW76,dellamW76)
colorW79=getcolor(6600.,100.,0.03,4.2,0.1049,outpoint1W79,outpoint2W79,inpointW79,meanerrW79,dellamW79)
colorW103=getcolor(6110.,160.,0.06,4.22,0.1093,outpoint1W103,outpoint2W103,inpointW103,meanerrW103,dellamW103)
colorW121=getcolor(6460.,140.,0.13,4.2,0.1245,outpoint1W121,outpoint2W121,inpointW121,meanerrW121,dellamW121)
# colorW77=getcolor(5605.,111.,0.07,4.49,0.13012,outpoint1W77,outpoint2W77,inpointW77,meanerrW77,dellamW77)
colorW77=getcolor(5556.,107.,0.07,4.49,0.13012,outpoint1W77,outpoint2W77,inpointW77,meanerrW77,dellamW77)

colorC2=getcolor(5575.,66.,-0.04,4.51,0.1626,outpoint1C2,outpoint2C2,inpointC2,meanerrC2,dellamC2)
colorKep13=getcolor(7650.,250.,0.2,4.2,0.08047,outpoint1Kep13,outpoint2Kep13,inpointKep13,meanerrKep13,dellamKep13)
colorT3=getcolor(5514.,69.,-0.2,4.57,0.1619,outpoint1T3,outpoint2T3,inpointT3,meanerrT3,dellamT3)
colorW4=getcolor(5500.,100.,-0.03,4.5,0.1485,outpoint1W4,outpoint2W4,inpointW4,meanerrW4,dellamW4)
colorW12=getcolor(6118.,64.,0.07,4.14,0.115,outpoint1W12,outpoint2W12,inpointW12,meanerrW12,dellamW12)

colorW121v2=getcolor(6460.,140.,0.13,4.2,0.1245,outpoint1W121v2,outpoint2W121v2,inpointW121v2,meanerrW121v2,dellamW121v2)
colorW76v2=getcolor(6250.,100.,0.23,4.128,0.10873,outpoint1W76v2,outpoint2W76v2,inpointW76v2,meanerrW76v2,dellamW76v2)
colorW76v3=getcolor(6250.,100.,0.23,4.128,0.10873,outpoint1W76v3,outpoint2W76v3,inpointW76v3,meanerrW76v3,dellamW76v3)
colorKep13v2=getcolor(7650.,250.,0.2,4.2,0.08047,outpoint1Kep13v2,outpoint2Kep13v2,inpointKep13v2,meanerrKep13v2,dellamKep13v2)
colorK7v2=getcolor(6789.,50.,0.139,4.149,0.0888,outpoint1K7v2,outpoint2K7v2,inpointK7v2,meanerrK7v2,dellamK7v2)

####################### Adding Brown Dwarfs from Manjavacas et al. (2019) #########################################
bddatalist=glob.glob('./ManjavacasData/*.txt')
cbdlist=np.sort(bddatalist)
bdcolors=np.zeros((np.shape(bddatalist)[0],5))
extrabdinfo=np.loadtxt('ManjavacasBDdistances.txt')
distancelist=extrabdinfo[:,0] #distances to all the brown dwarfs in pc
radlist=extrabdinfo[:,1] #radii of brown dwarfs in Jupiter radii
counter=0
goodlist=[]
for file in cbdlist:
	bddata=np.loadtxt(file)
	bdfgoodunit=bddata[:,1]*10.**4*10.**10*((distancelist[counter]*3.086*10**16.)/(radlist[counter]*69.911*10**6.))**2. #unit erg/s/m^3
	bderrgoodunit=bddata[:,2]*10.**4*10.**10*((distancelist[counter]*3.086*10**16.)/(radlist[counter]*69.911*10**6.))**2. #unit erg/s/m^3
	# bdfgoodunit[bdfgoodunit<0]=0.
	# bderrgoodunit[bderrgoodunit<0]=0.
	bdwavesample=np.interp(masterwavegrid,bddata[:,0],bdfgoodunit)
	bderrsample=np.interp(masterwavegrid,bddata[:,0],bderrgoodunit)
	outpoint1bd=trapezoidint(masterwavegrid[outset1]*10.**-6.,bdwavesample[outset1])
	outpoint2bd=trapezoidint(masterwavegrid[outset2]*10.**-6.,bdwavesample[outset2])
	inpointbd=trapezoidint(masterwavegrid[inset]*10.**-6.,bdwavesample[inset])
	outerr1bd=trapezoidint(masterwavegrid[outset1]*10.**-6.,bderrsample[outset1])
	outerr2bd=trapezoidint(masterwavegrid[outset2]*10.**-6.,bderrsample[outset2])
	inerrbd=trapezoidint(masterwavegrid[inset]*10.**-6.,bderrsample[inset])
	#print(outpoint1bd,outpoint2bd,inpointbd)
	if not inpointbd<0:
		print(counter)
		goodlist.append(counter)
	# meanerrbd=np.mean(bderrgoodunit)
	# dellambd=np.mean(np.diff(bddata[:,0]))
	bdcolors[counter,:]=colormagbd(outpoint1bd,outpoint2bd,inpointbd,outerr1bd,outerr2bd,inerrbd)
	counter+=1

delTbd=np.array([20,6.5,4,0.5,0.5,0.5,5.5,0.5,0.5,0.5,0.5,0.5,0.5,5,5,0.5,0.5,0.5,0.5,8,0.5,\
	0.5,0.5,0.5,0.5,0.5,0.5,6.5,0.5,6.5,0.5,7,1.5,1,4,4,5,0.5,4,4.5,22.5,3.5,0.5,0.5,6.5,\
	0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,4])
# testfile=np.loadtxt('./data/WASP-43b.txt')
# testfgoodunit=testfile[:,1]*10.**4*10.**10
# testerrgoodunit=testfile[:,2]*10.**4*10.**10
# testwavesample=np.interp(masterwavegrid,testfile[:,0],testfgoodunit)
# outpoint1test=trapezoidint(masterwavegrid[outset1]*10.**-6.,testwavesample[outset1])
# outpoint2test=trapezoidint(masterwavegrid[outset2]*10.**-6.,testwavesample[outset2])
# inpointtest=trapezoidint(masterwavegrid[inset]*10.**-6.,testwavesample[inset])

#for Jake's JWST proposal
# K24=np.loadtxt('G395M_KELT-24b_Tad_scenario3.txt')
# K7down=np.interp(masterwavegrid,K7[:,0],K7[:,1]*10.**-2.)

# plt.figure()
# plt.errorbar([1.275,1.415,1.57],[Fpout1W33,FpinW33,Fpout2W33],yerr=[outerr1,inerr,outerr2],color='k',marker='.',linestyle='none',ms=15)
# plt.scatter(1.415,10**colorW33[1],color='r',s=150,marker='.')
# plt.show()

# plt.figure()
# plt.errorbar([1.275,1.415,1.57],[Fpout1W79,FpinW79,Fpout2W79],yerr=[outerr1,inerr,outerr2],color='k',marker='.',linestyle='none',ms=15)
# plt.scatter(1.415,10**colorW79[1],color='r',s=150,marker='.')
# plt.show()

# sp = S.Icat('k93models',6366.,0.204,4.122)	#Parameters go temp, metallicity, logg
# sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
# wave=sp.wave*10.**-10  #in meters
# fluxW76v2 = sp.flux*10.**4*10.**10 #in erg/m2/m/s
# interpW76v2=np.interp(masterwavegrid,wave*10**6.,fluxW76v2)
# starout1W76v2=trapezoidint(masterwavegrid[outset1]*10.**-6.,interpW76v2[outset1])	#unit erg/s/m^2
# starout2W76v2=trapezoidint(masterwavegrid[outset2]*10.**-6.,interpW76v2[outset2])
# starinW76v2=trapezoidint(masterwavegrid[inset]*10.**-6.,interpW76v2[inset])
# rprsW76v2=0.10873
# Fpout1W76v2=outpoint1W76v2*starout1W76v2/rprsW76v2**2.	#unit erg/s/m^2
# Fpout2W76v2=outpoint2W76v2*starout2W76v2/rprsW76v2**2.
# FpinW76v2=inpointW76v2*starinW76v2/rprsW76v2**2.
# meanerrW76v2=np.mean(W76v2[:,2]*10.**-6.)
# dellamW76v2=np.mean(np.diff(W76v2[:,0]))
# FsW76v2=np.array([starout1W76v2,starout2W76v2,starinW76v2])
# colorW76v2=colormagdata(Fpout1W76v2,Fpout2W76v2,FpinW76v2,meanerrW76v2,dellamW76v2,FsW76v2,0.10873)

# sp = S.Icat('k93models',6329.,0.366,4.196)	#Parameters go temp, metallicity, logg
# sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
# wave=sp.wave*10.**-10  #in meters
# fluxW76v3 = sp.flux*10.**4*10.**10 #in erg/m2/m/s
# interpW76v3=np.interp(masterwavegrid,wave*10**6.,fluxW76v3)
# starout1W76v3=trapezoidint(masterwavegrid[outset1]*10.**-6.,interpW76v3[outset1])	#unit erg/s/m^2
# starout2W76v3=trapezoidint(masterwavegrid[outset2]*10.**-6.,interpW76v3[outset2])
# starinW76v3=trapezoidint(masterwavegrid[inset]*10.**-6.,interpW76v3[inset])
# rprsW76v3=0.1062
# Fpout1W76v3=outpoint1W76v3*starout1W76v3/rprsW76v3**2.	#unit erg/s/m^2
# Fpout2W76v3=outpoint2W76v3*starout2W76v3/rprsW76v3**2.
# FpinW76v3=inpointW76v3*starinW76v3/rprsW76v3**2.
# meanerrW76v3=np.mean(W76v3[:,2]*10.**-6.)
# dellamW76v3=np.mean(np.diff(W76v3[:,0]))
# FsW76v3=np.array([starout1W76v3,starout2W76v3,starinW76v3])
# colorW76v3=colormagdata(Fpout1W76v3,Fpout2W76v3,FpinW76v3,meanerrW76v3,dellamW76v3,FsW76,0.1062)

# sp = S.Icat('k93models',6460.,0.13,4.2)	#Parameters go temp, metallicity, logg
# sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
# wave=sp.wave*10.**-10  #in meters
# fluxW121v2 = sp.flux*10.**4*10.**10 #in erg/m2/m/s
# interpW121v2=np.interp(masterwavegrid,wave*10**6.,fluxW121v2)
# starout1W121v2=trapezoidint(masterwavegrid[outset1]*10.**-6.,interpW121v2[outset1])	#unit erg/s/m^2
# starout2W121v2=trapezoidint(masterwavegrid[outset2]*10.**-6.,interpW121v2[outset2])
# starinW121v2=trapezoidint(masterwavegrid[inset]*10.**-6.,interpW121v2[inset])
# rprsW121v2=0.1245
# Fpout1W121v2=outpoint1W121v2*starout1W121v2/rprsW121v2**2.	#unit erg/s/m^2
# Fpout2W121v2=outpoint2W121v2*starout2W121v2/rprsW121v2**2.
# FpinW121v2=inpointW121v2*starinW121v2/rprsW121v2**2.
# meanerrW121v2=np.mean(W121v2[:,2]*10.**-6.)
# dellamW121v2=np.mean(np.diff(W121v2[:,0]))
# FsW121v2=np.array([starout1W121v2,starout2W121v2,starinW121v2])
# colorW121v2=colormagdata(Fpout1W121v2,Fpout2W121v2,FpinW121v2,meanerrW121v2,dellamW121v2,FsW121v2,rprsW121v2)

# Fpout1W103night=outpoint1W103night*starout1W103/rprsW103**2.	#unit erg/s/m^2
# Fpout2W103night=outpoint2W103night*starout2W103/rprsW103**2.
# FpinW103night=inpointW103night*starinW103/rprsW103**2.
# meanerrW103night=np.mean(W103night[:,2]*10.**-6.)
# dellamW103night=np.mean(np.diff(W103night[:,0]))
# FsW103=np.array([starout1W103,starout2W103,starinW103])
# colorW103night=colormagdata(Fpout1W103night,Fpout2W103night,FpinW103night,meanerrW103night,dellamW103night,FsW103,rprsW103)

# Fpout1W18night=outpoint1W18night*starout1W18/rprsW18**2.	#unit erg/s/m^2
# Fpout2W18night=outpoint2W18night*starout2W18/rprsW18**2.
# FpinW18night=inpointW18night*starinW18/rprsW18**2.
# meanerrW18night=np.mean(W18night[:,2]*10.**-6.)
# dellamW18night=np.mean(np.diff(W18night[:,0]))
# FsW18=np.array([starout1W103,starout2W18,starinW18])
# colorW18night=colormagdata(Fpout1W18night,Fpout2W18night,FpinW18night,meanerrW18night,dellamW18night,FsW18,rprsW18)

# Fpout1W43night=outpoint1W43night*starout1W43/rprsW43**2.	#unit erg/s/m^2
# Fpout2W43night=outpoint2W43night*starout2W43/rprsW43**2.
# FpinW43night=inpointW43night*starinW43/rprsW43**2.
# meanerrW43night=np.mean(W43night[:,2]*10.**-6.)
# dellamW43night=np.mean(np.diff(W43night[:,0]))
# FsW43=np.array([starout1W43,starout2W43,starinW43])
# colorW43night=colormagdata(Fpout1W43night,Fpout2W43night,FpinW43night,meanerrW43night,dellamW43night,FsW43,rprsW43)

# np.savez('HSTcolors.npz',color_fiducial=color_fiducial,color_lowCO=color_lowCO,color_highCO=color_highCO,\
# 	color_delayTiO1600=color_delayTiO1600,color_delayTiO1800=color_delayTiO1800,color_delayTiO2000=color_delayTiO2000,\
# 	color_delayTiO2800=color_delayTiO2800,color_delayTiO3600=color_delayTiO3600,color_grav20=color_grav20,\
# 	color_grav40=color_grav40,color_metneg15=color_metneg15,color_metpos15=color_metpos15,color_star3300=color_star3300,\
# 	color_star4300=color_star4300,color_star6300=color_star6300,color_star7200=color_star7200,color_star8200=color_star8200,\
# 	color_tint1percent=color_tint1percent,color_tintTF18=color_tintTF18,colorH7=colorH7,colorH32=colorH32,colorH41=colorH41,\
# 	colorHD189=colorHD189,colorHD209=colorHD209,colorK7=colorK7,colorW18=colorW18,colorW33=colorW33,colorW43=colorW43,\
# 	colorW74=colorW74,colorW76=colorW76,colorW79=colorW79,colorW103=colorW103,colorW121=colorW121)

############################################## ADD PROPOSED SURVEY PLANETS ########################
def addasim(Tstar,Tstarerr,metstar,loggstar,rprs,Tp,scale):
	sp = S.Icat('k93models',Tstar,metstar,loggstar)	#Parameters go temp, metallicity, logg
	sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave=sp.wave*10.**-10  #in meters
	flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
	interp=np.interp(masterwavegrid,wave*10**6.,flux) #star in erg/s/m^3
	starout1=trapezoidint(masterwavegrid[outset1]*10.**-6.,interp[outset1])	#unit erg/s/m^2
	starout2=trapezoidint(masterwavegrid[outset2]*10.**-6.,interp[outset2])
	starin=trapezoidint(masterwavegrid[inset]*10.**-6.,interp[inset])
	sp2=S.Icat('k93models',Tstar+Tstarerr,metstar,loggstar)
	sp2.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave2=sp2.wave*10.**-10  #in meters
	flux2 = sp2.flux*10.**4*10.**10 #in erg/m2/m/s
	interp2=np.interp(masterwavegrid,wave*10**6.,flux2) #star in erg/s/m^3
	starout12=trapezoidint(masterwavegrid[outset1]*10.**-6.,interp2[outset1])	#unit erg/s/m^2
	starout22=trapezoidint(masterwavegrid[outset2]*10.**-6.,interp2[outset2])
	starin2=trapezoidint(masterwavegrid[inset]*10.**-6.,interp2[inset])
	w77=fakedata_bbod(wave,Tp) #erg/s/m^2/m
	interpp=np.interp(masterwavegrid,wave*10**6.,w77)
	spectrum=interpp/interp*rprs**2.
	outpoint1=trapezoidint(masterwavegrid[outset1]*10.**-6.,interpp[outset1]) #unit erg/s/m^2
	outpoint2=trapezoidint(masterwavegrid[outset2]*10.**-6.,interpp[outset2]) #unit erg/s/m^2
	inpoint=trapezoidint(masterwavegrid[inset]*10.**-6.,interpp[inset]) #unit erg/s/m^2
	meanerr=meanerrW43*np.sqrt(5.)/scale 
	dellam=dellamW43
	Fs=np.array([starout1,starout2,starin])
	Fs2=np.array([starout12,starout22,starin2])
	# pdb.set_trace()
	color=colormagdata(outpoint1,outpoint2,inpoint,meanerr,dellam,Fs,Fs2,rprs)
	#creal=np.interp(color[1],color_tint1percent[:,1],color_tint1percent[:,2])
	creal=np.interp(color[1],color_fiducial[:,1],color_fiducial[:,2])
	creal2=np.interp(color[1],color_grav20[:,1],color_grav20[:,2])
	creal3=np.interp(color[1],color_grav40[:,1],color_grav40[:,2])
	return color,creal,creal2,creal3,spectrum

# colorW77,W77creal,W77creal2,W77creal3=addasim(5365.,-0.34,4.33,0.13012,1960.,1.392)#1913.
# colorK231,K231creal,K231creal2,K231creal3=addasim(5280.,0.080,4.60,0.135,1778.,0.792)
# # colorW95,W95creal=addasim(5830.,0.14,4.36,0.1005,1921.,0.629)
# colorK4,K4creal,K4creal2,K4creal3=addasim(6206.,-0.116,4.108,0.1089,2154.,1.128)
# colorH32f,H32fcreal,H32fcreal2,H32fcreal3=addasim(6001.,-0.16,4.22,0.14886,1935.,0.709)
# colorK3,K3creal,K3creal2,K3creal3=addasim(6304.,0.05,4.20,0.0939,2149.,0.789)
# colorH36,H36creal,H36creal2,H36creal3=addasim(5620.,0.26,4.37,0.1199,2108.,0.595)

#brian proposal
colorMA1,MA1creal,MA1creal2,MA1creal3,MA1spectrum=addasim(7554,138.,0.0,4.2,0.0735,3170.,4.4)#3025.,2.522)
colorK16,K16creal,K16creal2,K16creal3,K16spectrum=addasim(6236.,139.,0.0,4.2,0.1070,2920.,2.1)#2929.,1.780)
colorK20,K20creal,K20creal2,K20creal3,K20spectrum=addasim(8720.,131.,0.0,4.2,0.1144,2680.,5.1)#2695.,3.558)
colorW3,W3creal,W3creal2,W3creal3,W3spectrum=addasim(6400.,144.,0.04,4.42,0.1051,2360.,1.5)#2356.,1.232)
# colorW173,W173creal,W173creal2,W173creal3=addasim(5800.,0.0,4.2,0.1203,2230.,1.3)#2218.,1.104)
# colorW14,W14creal,W14creal2,W14creal3=addasim(6475.,0.0,4.07,0.0972,2210.,1.7)#2200.,0.927)
colorK11,K11creal,K11creal2,K11creal3,K11spectrum=addasim(5375.,25.,0.17,3.7,0.0514,2060.,2.2)#2005.,0.844)
colorK8,K8creal,K8creal2,K8creal3,K8spectrum=addasim(5754.,70.,0.27,4.08,0.1145,1990.,1.3)#1971.,0.823)
# colorW77,W77creal,W77creal2,W77creal3=addasim(5365.,-0.34,4.33,0.13012,1990.,2.1)
colorC2,C2creal,C2creal2,C2creal3,C2spectrum=addasim(5625.,49.,-0.09,4.73,0.161,1830.,0.9)#1799.,0.604)
# colorK23,K23creal,K23creal2,K23creal3=addasim(5899.,-0.105,4.417,0.1365,1561.,1.2)#1832.,0.864)
colorXO3,XO3creal,XO3creal2,XO3creal3,XO3spectrum=addasim(6429.,150.,-0.18,4.7,0.1005,2175.,0.9)
colorW140,W140creal,W140creal2,W140creal3,W140spectrum=addasim(5260.,78.,0.12,4.51,0.1664,1561.,0.7)

# #add for Kevin proposal
# colorLTT,LTTcreal,LTTcreal2,LTTcreal3,LTTspectrum=addasim(5443.,14.,0.27,4.35,0.0445,2336.,0.537)

################################################# MAKE PLOTS ###################################
# interpset=mikemodssorth209[:,4:8]
# biginterpset=np.zeros((np.shape(HD209[:,0])[0],1000))
# for i in np.arange(np.shape(HD209[:,0])[0]):
# 	biginterpset[i,:]=np.linspace(interpset[i,0],interpset[i,3],1000)
# #find which one has minimum error
# moderrors=np.zeros(1000)
# for i in np.arange(1000):
# 	leastsquares=(biginterpset[:,i]-tcHD209)**2.
# 	moderrors[i]=np.sum(leastsquares)

# #EXPLANATION PLOT
# W43mod=np.loadtxt('modelW43.txt')
# rc('axes',linewidth=2)
# fig,ax=plt.subplots(figsize=(9.5,7.5))
# plt.errorbar(W43[:,0],W43[:,1],yerr=W43[:,2],color='xkcd:blue',marker='.',linestyle='none',label='WASP-43b, Kreidberg+2014',zorder=3,markersize=15,linewidth=3)
# plt.plot(W43[:,0],make_bbod(W43[:,0],dellamW43,1769.),color='xkcd:slate gray',zorder=0,linewidth=3,marker='.',markeredgecolor='k',markersize=15,label='Blackbody')
# plt.plot(W43mod[:,0],W43mod[:,1],marker='.',color='xkcd:tan',linewidth=3,markeredgecolor='xkcd:light brown',markersize=15,zorder=1,label='Updated Fortney+2008 Model')
# plt.fill_between([1.35,1.48],[10.**10.,10.**10.],[10.**12.,10.**12.],color='xkcd:green',alpha=0.2,zorder=0,edgecolor='none',linewidth=0.0)
# plt.fill_between([1.22,1.33],[10.**10.,10.**10.],[10.**12.,10.**12.],color='xkcd:red',alpha=0.2,zorder=0,edgecolor='none',linewidth=0.0)
# plt.fill_between([1.53,1.61],[10.**10.,10.**10.],[10.**12.,10.**12.],color='xkcd:red',alpha=0.2,zorder=0,edgecolor='none',linewidth=0.0)
# plt.text(1.415,0.2*10.**11,'In-Band Flux',color='xkcd:green',fontsize=18,horizontalalignment='center')
# plt.text(1.275,0.2*10.**11,'Out-Of-\nBand Flux',color='xkcd:red',fontsize=18,horizontalalignment='center')
# plt.xlim((1.1,1.7))
# plt.ylim((1.5*10.**10,1.0*10.**11))
# plt.legend(fontsize=20,loc='upper right')
# plt.xlabel('Wavelength [$\mu$m]',fontsize=20)
# plt.ylabel('Planet Flux [erg/s/m$^{2}$]',fontsize=20)
# plt.tick_params(labelsize=20,axis="both",right=True,top=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="y",right=True,width=1.5,length=5,direction='in')
# t = ax.yaxis.get_offset_text()
# t.set_size(18)
# plt.show()

############################### MODELS SHOWING ALL THE DIFFERENT MODEL TRACKS ############################

###################### MAIN FIGURE START HERE ###################################################

# #testing fill between
# rc('axes',linewidth=2)
# fillhighCO=np.interp(color_lowCO[:,1],color_highCO[:,1],color_highCO[:,2])
# filllowCO=np.interp(color_highCO[:,1],color_lowCO[:,1],color_lowCO[:,2])
# filllowmet=np.interp(color_metpos15[:,1],color_metneg15[:,1],color_metneg15[:,2])
# fillhighmet=np.interp(color_metneg15[:,1],color_metpos15[:,1],color_metpos15[:,2])
# filldelaylow1=np.interp(color_grav20[:,1],color_delayTiO2800[:,1],color_delayTiO2800[:,2])
# filldelaylow2=np.interp(color_star8200[:,1],color_delayTiO2800[:,1],color_delayTiO2800[:,2])
# #filldelayhigh1=np.interp(color_delayTiO2800[:,1],color_grav20[:,1],color_grav20[:,2])
# filldelayhigh2=np.interp(color_delayTiO2800[:,1],color_star8200[:,1],color_star8200[:,2])
# fillthebottom1=np.interp(color_metneg15[:,1],color_star8200[:,1],color_star8200[:,2])
# fillthebottom2=np.interp(color_grav20[:,1],color_grav40[:,1],color_grav40[:,2])
# fillthebottom3=np.interp(color_tint1percent[:,1],color_metpos15[:,1],color_metpos15[:,2])
# fillstar=np.interp(color_star8200[:,1],color_star3300[:,1],color_star3300[:,2])

# #adding colormap for equilibrium temps
# equtemps=np.array([2241.,1898.,1935.,1216.,1451.,2047.,2550.,2383.,2780.,1448.,1921.,2169.,1867.,2504.,2358.])	#H7,H32,H41,HD189,HD209,K7,Kep13,W18,W33,W43,W74,W76,W79,W103,W121
# vmin=np.min(equtemps)
# vmax=np.max(equtemps)+200
# normequtemps=(equtemps-vmin)/np.max(equtemps-vmin)
# inferno = cm = plt.get_cmap('inferno') 
# cNorm  = mplcolors.Normalize(vmin=vmin, vmax=vmax)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=inferno)
# #print(scalarMap.get_clim())

# mycolor='xkcd:light gray'#'xkcd:beige' #'#dbedff' color I used for the fill_between on the proposal 
# edgecolor='k'#'xkcd:brown' #'xkcd:dark blue'
# linecolor='xkcd:slate gray'# 'xkcd:light brown'#'xkcd:cerulean'
# linecolor2='xkcd:light brown'
# linecolor3='xkcd:olive green'
# pointcolor2='xkcd:brown'
# pointcolor3='xkcd:dark green'

# fig,ax1=plt.subplots(figsize=(10,7))
# ax1.set_ylim((10.5,12.75))
# ax1.set_xlim((-0.2,0.65))
# # ax1.set_xlim((-0.2,0.75))
# ax1.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
# ax1.set_yticks(ticks=[10.537,10.779,11.174,11.336,11.481,11.611,11.834,11.930,12.018,12.100,12.244,12.308,12.368,12.424,12.526,12.572,12.616,12.658],minor=True)
# ax1.set_yticklabels([],minor=True)
# ax1.set_ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# ax1.set_xlabel('Water Feature Strength',fontsize=20)#log(Blackbody/In-Band Flux) Water Feature Strength
# # ax1.set_xlabel('Water Feature Strength',fontsize=20)#log(Blackbody/In-Band Flux) Water Feature Strength
# ax1.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

# ax1.axvline(x=0.0,color='k',zorder=1,linewidth=2)#'xkcd:slate gray'
# ax1.plot(color_fiducial[:,2],color_fiducial[:,1],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)#'xkcd:light brown','xkcd:brown'
# ax1.text(0.17,11.7,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
# # ax1.text(0.2,10.75,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
# # ax1.plot(color_lowCO[:,2],color_highCO[:,1],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
# # ax1.text(0.44,11.4,'[C/O]=0.01',color=linecolor2,fontsize=15,zorder=3,fontweight='bold')
# # ax1.plot(color_tintTF18[:,2],color_tintTF18[:,1],color=linecolor3,marker='.',markeredgecolor=pointcolor3,zorder=2,linewidth=2,markersize=10)
# # ax1.text(0.3,11.55,'TF18 Internal Heat',color=linecolor3,fontsize=15,zorder=3,fontweight='bold')

# # #ax1.plot(color_fiducialJHK[:,2],color_fiducialJHK[:,1],color='xkcd:olive green',marker='.',markeredgecolor='xkcd:dark green',zorder=2,linewidth=2,markersize=10)
# # ax1.plot(color_metpos15[:,2],color_metpos15[:,1],color='r',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_grav20[:,2],color_grav20[:,1],color='g',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # #ax1.plot(color_grav35[:,2],color_grav35[:,1],color='xkcd:tan',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_grav40[:,2],color_grav40[:,1],color='g',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_delayTiO1600[:,2],color_delayTiO1600[:,1],color='b',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_delayTiO1800[:,2],color_delayTiO1800[:,1],color='b',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_delayTiO2000[:,2],color_delayTiO2000[:,1],color='b',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_delayTiO2800[:,2],color_delayTiO2800[:,1],color='b',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_delayTiO3600[:,2],color_delayTiO3600[:,1],color='b',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_lowCO[:,2],color_lowCO[:,1],color='c',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_highCO[:,2],color_highCO[:,1],color='c',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_star3300[:,2],color_star3300[:,1],color='m',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_star4300[:,2],color_star4300[:,1],color='m',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_star6300[:,2],color_star6300[:,1],color='m',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_star7200[:,2],color_star7200[:,1],color='m',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_star8200[:,2],color_star8200[:,1],color='m',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_tint1percent[:,2],color_tint1percent[:,1],color='y',marker='.',markeredgecolor='xkcd:light brown',zorder=1)
# # ax1.plot(color_tintTF18[:,2],color_tintTF18[:,1],color='y',marker='.',markeredgecolor='xkcd:light brown',zorder=1)

# ax1.fill_betweenx(color_lowCO[:,1],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_highCO[:,1],color_highCO[:,2],filllowCO,color=mycolor,zorder=0)#'xkcd:beige'
# #ax1.fill_betweenx(color_metpos15[:,1],color_metpos15[:,2],filllowmet,color='xkcd:grey')
# ax1.fill_betweenx(color_metneg15[:,1],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=0)#'xkcd:beige'
# #ax1.fill_betweenx(color_grav25[:,1],color_grav25[:,2],filldelaylow1,color='xkcd:grey')
# ax1.fill_betweenx(color_star8200[:,1],color_star8200[:,2],filldelaylow2,color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_star8200[-3:,1],color_star8200[-3:,2],fillstar[-3:],color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_delayTiO2800[:,1],color_delayTiO2800[:,2],filldelayhigh2,color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_metneg15[:5,1],color_metneg15[:5,2],fillthebottom1[:5],color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_grav20[:,1],color_grav20[:,2],fillthebottom2,color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_tint1percent[:,1],color_tint1percent[:,2],fillthebottom3,color=mycolor,zorder=0)#'xkcd:beige'

# ax1.errorbar(colorH7[2],colorH7[1],xerr=colorH7[4],yerr=colorH7[3],mec=scalarMap.to_rgba(equtemps[0]),mfc=scalarMap.to_rgba(equtemps[0]),ecolor=scalarMap.to_rgba(equtemps[0]),marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
# ax1.text(colorH7[2]+0.01,colorH7[1]+0.05,'HAT-P-7b',color=scalarMap.to_rgba(equtemps[0]),fontsize=15,zorder=3)#'xkcd:kelly green'
# ax1.errorbar(colorH32[2],colorH32[1],xerr=colorH32[4],yerr=colorH32[3],mec=scalarMap.to_rgba(equtemps[1]),mfc=scalarMap.to_rgba(equtemps[1]),ecolor=scalarMap.to_rgba(equtemps[1]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorH32[2]-0.11,colorH32[1]-0.17,'HAT-P-32b',color=scalarMap.to_rgba(equtemps[1]),fontsize=15,zorder=3)#'xkcd:pink'
# ax1.errorbar(colorH41[2],colorH41[1],xerr=colorH41[4],yerr=colorH41[3],mec=scalarMap.to_rgba(equtemps[2]),mfc=scalarMap.to_rgba(equtemps[2]),ecolor=scalarMap.to_rgba(equtemps[2]),marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
# ax1.text(colorH41[2]+0.03,colorH41[1]-0.07,'HAT-P-41b',color=scalarMap.to_rgba(equtemps[2]),bbox=dict(facecolor='none',edgecolor='k'),fontsize=15,zorder=3)#'xkcd:light orange'
# # ax1.text(colorH41[2]+0.08,colorH41[1]-0.06,'HAT-P-41b',color=scalarMap.to_rgba(equtemps[2]),fontsize=15,zorder=3)#'xkcd:light orange'
# ax1.errorbar(colorHD189[2],colorHD189[1],xerr=colorHD189[4],yerr=colorHD189[3],mec=scalarMap.to_rgba(equtemps[3]),mfc=scalarMap.to_rgba(equtemps[3]),ecolor=scalarMap.to_rgba(equtemps[3]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorHD189[2]-0.15,colorHD189[1]+0.07,'HD 189733b',color=scalarMap.to_rgba(equtemps[3]),fontsize=15,zorder=3)#'xkcd:greenish grey'
# ax1.errorbar(colorHD209[2],colorHD209[1],xerr=colorHD209[4],yerr=colorHD209[3],mec=scalarMap.to_rgba(equtemps[4]),mfc=scalarMap.to_rgba(equtemps[4]),ecolor=scalarMap.to_rgba(equtemps[4]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorHD209[2]+0.01,colorHD209[1]+0.07,'HD 209458b',color=scalarMap.to_rgba(equtemps[4]),fontsize=15,zorder=3)#'xkcd:bright pink'
# # ax1.text(colorHD209[2]+0.1,colorHD209[1],'HD 209458b',color=scalarMap.to_rgba(equtemps[4]),fontsize=15,zorder=3)#'xkcd:bright pink'
# ax1.errorbar(colorK7[2],colorK7[1],xerr=colorK7[4],yerr=colorK7[3],mec=scalarMap.to_rgba(equtemps[5]),mfc=scalarMap.to_rgba(equtemps[5]),ecolor=scalarMap.to_rgba(equtemps[5]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorK7[2]+0.03,colorK7[1]+0.02,'KELT-7b',color=scalarMap.to_rgba(equtemps[5]),bbox=dict(facecolor='none',edgecolor='k'),fontsize=15,zorder=3)#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# # ax1.errorbar(colorK7v2[2],colorK7v2[1],xerr=colorK7v2[4],yerr=colorK7v2[3],mec='g',mfc='g',ecolor='g',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # ax1.text(colorK7[2]+0.03,colorK7[1]+0.02,'KELT-7b',color=scalarMap.to_rgba(equtemps[5]),fontsize=15,zorder=3)#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# # ax1.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],mec=scalarMap.to_rgba(equtemps[6]),mfc=scalarMap.to_rgba(equtemps[6]),ecolor=scalarMap.to_rgba(equtemps[6]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # ax1.text(colorKep13[2]+0.05,colorKep13[1]-0.03,'Kepler-13Ab',color=scalarMap.to_rgba(equtemps[6]),fontsize=15,zorder=3)#'xkcd:lilac'
# ax1.errorbar(colorW18[2],colorW18[1],xerr=colorW18[4],yerr=colorW18[3],mec=scalarMap.to_rgba(equtemps[7]),mfc=scalarMap.to_rgba(equtemps[7]),ecolor=scalarMap.to_rgba(equtemps[7]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW18[2]-0.15,colorW18[1]-0.05,'WASP-18b',color=scalarMap.to_rgba(equtemps[7]),fontsize=15,zorder=3)#xkcd:blue
# ax1.errorbar(colorW33[2],colorW33[1],xerr=colorW33[4],yerr=colorW33[3],mec=scalarMap.to_rgba(equtemps[8]),mfc=scalarMap.to_rgba(equtemps[8]),ecolor=scalarMap.to_rgba(equtemps[8]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW33[2]+0.04,colorW33[1]-0.02,'WASP-33b',color=scalarMap.to_rgba(equtemps[8]),fontsize=15,zorder=3)#xkcd:sky blue bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW43[2],colorW43[1],xerr=colorW43[4],yerr=colorW43[3],mec=scalarMap.to_rgba(equtemps[9]),mfc=scalarMap.to_rgba(equtemps[9]),ecolor=scalarMap.to_rgba(equtemps[9]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW43[2]-0.13,colorW43[1]-0.13,'WASP-43b',color=scalarMap.to_rgba(equtemps[9]),fontsize=15,zorder=3)#xkcd:red
# ax1.errorbar(colorW74[2],colorW74[1],xerr=colorW74[4],yerr=colorW74[3],mec=scalarMap.to_rgba(equtemps[10]),mfc=scalarMap.to_rgba(equtemps[10]),ecolor=scalarMap.to_rgba(equtemps[10]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW74[2]+0.01,colorW74[1]-0.10,'WASP-74b',color=scalarMap.to_rgba(equtemps[10]),bbox=dict(facecolor='none',edgecolor='k'),fontsize=15,zorder=3)#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# # ax1.text(colorW74[2]+0.04,colorW74[1]-0.08,'WASP-74b',color=scalarMap.to_rgba(equtemps[10]),fontsize=15,zorder=3)#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW76[2],colorW76[1],xerr=colorW76[4],yerr=colorW76[3],mec=scalarMap.to_rgba(equtemps[11]),mfc=scalarMap.to_rgba(equtemps[11]),ecolor=scalarMap.to_rgba(equtemps[11]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW76[2]-0.15,colorW76[1]-0.06,'WASP-76b',color=scalarMap.to_rgba(equtemps[11]),bbox=dict(facecolor='none',edgecolor='k'),fontsize=15,zorder=3)#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# # ax1.errorbar(colorW76v2[2],colorW76v2[1],xerr=colorW76v2[4],yerr=colorW76v2[3],mec='k',mfc='k',ecolor='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # ax1.errorbar(colorW76v3[2],colorW76v3[1],xerr=colorW76v3[4],yerr=colorW76v3[3],mec='r',mfc='r',ecolor='r',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # ax1.text(colorW76[2]+0.05,colorW76[1]-0.06,'WASP-76b',color=scalarMap.to_rgba(equtemps[11]),fontsize=15,zorder=3)#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW79[2],colorW79[1],xerr=colorW79[4],yerr=colorW79[3],mec=scalarMap.to_rgba(equtemps[12]),mfc=scalarMap.to_rgba(equtemps[12]),ecolor=scalarMap.to_rgba(equtemps[12]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW79[2]+0.04,colorW79[1]+0.1,'WASP-79b',color=scalarMap.to_rgba(equtemps[12]),bbox=dict(facecolor='none',edgecolor='k'),fontsize=15,zorder=3)#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# # ax1.text(colorW79[2]+0.01,colorW79[1]+0.05,'WASP-79b',color=scalarMap.to_rgba(equtemps[12]),fontsize=15,zorder=3)#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW103[2],colorW103[1],xerr=colorW103[4],yerr=colorW103[3],mec=scalarMap.to_rgba(equtemps[13]),mfc=scalarMap.to_rgba(equtemps[13]),ecolor=scalarMap.to_rgba(equtemps[13]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW103[2]-0.17,colorW103[1]+0.03,'WASP-103b',color=scalarMap.to_rgba(equtemps[13]),fontsize=15,zorder=3)#xkcd:violet
# ax1.errorbar(colorW121[2],colorW121[1],xerr=colorW121[4],yerr=colorW121[3],mec=scalarMap.to_rgba(equtemps[14]),mfc=scalarMap.to_rgba(equtemps[14]),ecolor=scalarMap.to_rgba(equtemps[14]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW121[2]-0.15,colorW121[1]-0.02,'WASP-121b',color=scalarMap.to_rgba(equtemps[14]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:light green
# # ax1.errorbar(colorW121v2[2],colorW121v2[1],xerr=colorW121v2[4],yerr=colorW121v2[3],mec='g',mfc='g',ecolor='g',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # ax1.text(colorW121v2[2]-0.15,colorW121v2[1]-0.02,'WASP-121b',color=scalarMap.to_rgba(equtemps[14]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:light green
# # ax1.text(colorW121[2]-0.16,colorW121[1]-0.02,'WASP-121b',color=scalarMap.to_rgba(equtemps[14]),fontsize=15,zorder=3)#xkcd:light green
# # ax1.set_xlabel('log(Blackbody/In-Band Flux) [erg/s/m$^{2}$]',fontsize=20)
# # ax1.set_ylabel('log(Blackbody Flux) [erg/s/m$^{2}$]',fontsize=20)
# # ax1.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# # ax1.set_ylim((10.3,12.8))
# # ax1.set_xlim((-0.2,0.65))
# # #plt.title('All Changes Combined')
# # #plt.legend()
# # ax2=ax1.twinx()
# # ax2.set_ylim((10.3,12.8))
# # ax2.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# # ax2.set_yticklabels(['1500','2000','2500','3000','3500'])
# # ax2.set_ylabel('Blackbody Temperature [K]',fontsize=20)
# # ax2.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')

# a = plt.axes([0.85,0.18,0.01,0.75], frameon=False)#plt.axes([0.76,0.20,0.01,0.75], frameon=False) 
# a.yaxis.set_visible(False)
# a.xaxis.set_visible(False)
# a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.inferno, aspect='auto', visible=False)
# cbar=plt.colorbar(a, fraction=3.0)
# cbar.ax.tick_params(labelsize=15,width=2,length=6)
# cbar.set_label('Equilibrium Temperature',fontsize=15)

# plt.tight_layout()
# plt.show()

###################### MAIN FIGURE END HERE ###################################################

###################### MAIN FIGURE FULLY ON TEMPERATURE AXIS ####################################
#testing fill between
rc('axes',linewidth=2)
fillhighCO=np.interp(color_lowCO[:,0],color_highCO[:,0],color_highCO[:,2])
filllowCO=np.interp(color_highCO[:,0],color_lowCO[:,0],color_lowCO[:,2])
filllowmet=np.interp(color_highCO[:,0],color_metpos15[:,0],color_metpos15[:,2])
fillhighmet=np.interp(color_metneg15[:,0],color_lowCO[:,0],color_lowCO[:,2])
fillheat=np.interp(color_tintTF18[:,0],color_delayTiO3000[:,0],color_delayTiO3000[:,2])
fillg=np.interp(color_grav40[:,0],color_highCO[:,0],color_highCO[:,2])
filltop=np.interp(color_lowCO[:,0],color_metpos15[:,0],color_metpos15[:,2])

fillbd=np.interp(color_bdlogg3[:,0],color_bdmetneg1[:,0],color_bdmetneg1[:,2])

#adding colormap for equilibrium temps
# equtemps=np.array([2241.,1898.,1935.,1216.,1451.,2047.,2550.,2383.,2780.,1448.,1921.,2169.,1867.,2504.,2358.])	#H7,H32,H41,HD189,HD209,K7,Kep13,W18,W33,W43,W74,W76,W79,W103,W121
equtemps=np.array([2241.,1898.,1935.,1216.,1451.,2047.,2550.,2383.,2780.,1448.,1921.,2169.,1867.,2504.,2358.,1522.,1590.,1666.,2550.,2498.,1701.])	#H7,H32,H41,HD189,HD209,K7,Kep13,W18,W33,W43,W74,W76,W79,W103,W121,C2,T3,W4,Kep13,W12, W77
vmin=np.min(equtemps)
vmax=np.max(equtemps)+200
normequtemps=(equtemps-vmin)/np.max(equtemps-vmin)
inferno = cm = plt.get_cmap('inferno') 
cNorm  = mplcolors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=inferno)
#print(scalarMap.get_clim())

mycolor='xkcd:light gray'#'xkcd:beige' #'#dbedff' color I used for the fill_between on the proposal 
edgecolor='k'#'xkcd:brown' #'xkcd:dark blue'
linecolor='xkcd:slate gray'# 'xkcd:light brown'#'xkcd:cerulean'
linecolor2='xkcd:light brown'
linecolor3='xkcd:olive green'
pointcolor2='xkcd:brown'
pointcolor3='xkcd:dark green'

fig,ax1=plt.subplots(figsize=(10,7))
ax1.set_ylim((1150,3550))
# ax1.set_ylim((1150,3400))
ax1.set_xlim((-0.2,0.65))
# ax1.set_yticks(ticks=[1500,2000,2500,3000])
# ax1.set_yticklabels(['1500','2000','2500','3000'])
ax1.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
ax1.set_yticks(ticks=[1200,1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax1.set_yticklabels([],minor=True)
ax1.set_ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
ax1.set_xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=20) #
# ax1.set_xlabel('log(Blackbody/In-Band Flux)',fontsize=20)
ax1.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

ax1.axvline(x=0.0,color='k',zorder=1,linewidth=2)#'xkcd:slate gray'
ax1.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)#'xkcd:light brown','xkcd:brown'
ax1.text(-0.18,3400,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
# ax1.text(0.13,1500,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
# ax1.plot(color_metpos15[:,2],color_metpos15[:,0],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
# ax1.text(0.24,2000,'[M/H]=1.5',color=linecolor2,fontsize=15,zorder=3,fontweight='bold')
# ax1.plot(color_tintTF18[:,2],color_tintTF18[:,0],color=linecolor3,marker='.',markeredgecolor=pointcolor3,zorder=2,linewidth=2,markersize=10)
# ax1.text(0.01,2600,'Thorngren & Fortney (2018) Internal Heat',color=linecolor3,fontsize=15,zorder=3,fontweight='bold')
ax1.plot(color_bd[:,2],color_bd[:,0],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
# ax1.plot(color_bdmetpos1[:,2],color_bdmetpos1[:,0],color=linecolor3,marker='.',markeredgecolor=pointcolor3,zorder=2,linewidth=2,markersize=10)
# ax1.plot(color_bdmetneg1[:,2],color_bdmetneg1[:,0],color=linecolor3,marker='.',markeredgecolor=pointcolor3,zorder=2,linewidth=2,markersize=10)
#ax1.plot(color_bdlogg3[:,2],color_bdlogg3[:,0],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
# ax1.plot(color_bdlogg4[:,2],color_bdlogg4[:,0],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
ax1.text(0.2,2800,'Self-Irradiated Bodies',color=linecolor2,fontsize=15,zorder=3,fontweight='bold')
# ax1.text(,,'Brown Dwarfs')
ax1.fill_betweenx(color_bdlogg3[:,0],fillbd,color_bdlogg3[:,2],color='xkcd:beige',zorder=0)

ax1.fill_betweenx(color_lowCO[:,0],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=1)#'xkcd:beige'
ax1.fill_betweenx(color_highCO[:,0],color_highCO[:,2],filllowCO,color=mycolor,zorder=1)#'xkcd:beige'
ax1.fill_betweenx(color_metneg15[:,0],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=1)#'xkcd:beige'
ax1.fill_betweenx(color_highCO[:,0],color_highCO[:,2],filllowmet,color=mycolor,zorder=1)
ax1.fill_betweenx(color_tintTF18[:,0],color_tintTF18[:,2],fillheat,color=mycolor,zorder=1)
ax1.fill_betweenx(color_grav40[:,0],color_grav40[:,2],fillg,color=mycolor,zorder=1)
ax1.fill_betweenx(color_lowCO[:,0],color_lowCO[:,2],filltop,color=mycolor,zorder=1)

ax1.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39.,mec=scalarMap.to_rgba(equtemps[0]),mfc=scalarMap.to_rgba(equtemps[0]),ecolor=scalarMap.to_rgba(equtemps[0]),marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
# ax1.errorbar(colorH7v2[2],colorH7v2[0],xerr=colorH7v2[4],yerr=38.,mec='b',mfc='b',ecolor='b',marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=3)
ax1.text(colorH7[2]+0.01,colorH7[0]+20,'HAT-P-7b',color=scalarMap.to_rgba(equtemps[0]),fontsize=15,zorder=3)#'xkcd:kelly green'
ax1.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59.,mec=scalarMap.to_rgba(equtemps[1]),mfc=scalarMap.to_rgba(equtemps[1]),ecolor=scalarMap.to_rgba(equtemps[1]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorH32[2]-0.11,colorH32[0]+60,'HAT-P-32b',color=scalarMap.to_rgba(equtemps[1]),fontsize=15,zorder=3)#'xkcd:pink'
ax1.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66.,mec=scalarMap.to_rgba(equtemps[2]),mfc=scalarMap.to_rgba(equtemps[2]),ecolor=scalarMap.to_rgba(equtemps[2]),marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
ax1.text(colorH41[2]+0.02,colorH41[0]-100,'HAT-P-41b',color=scalarMap.to_rgba(equtemps[2]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#'xkcd:light orange'
# ax1.errorbar(colorH41v2[2],colorH41v2[0],xerr=colorH41v2[4],yerr=0.,mec='g',mfc='g',ecolor='g',marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
# ax1.text(colorH41[2]+0.08,colorH41[1]-0.06,'HAT-P-41b',color=scalarMap.to_rgba(equtemps[2]),fontsize=15,zorder=3)#'xkcd:light orange'
ax1.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57.,mec=scalarMap.to_rgba(equtemps[3]),mfc=scalarMap.to_rgba(equtemps[3]),ecolor=scalarMap.to_rgba(equtemps[3]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorHD189[2]+0.02,colorHD189[0]-80,'HD 189733b',color=scalarMap.to_rgba(equtemps[3]),fontsize=15,zorder=3)#'xkcd:greenish grey'
ax1.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28.,mec=scalarMap.to_rgba(equtemps[4]),mfc=scalarMap.to_rgba(equtemps[4]),ecolor=scalarMap.to_rgba(equtemps[4]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorHD209[2]+0.01,colorHD209[0]-75,'HD 209458b',color=scalarMap.to_rgba(equtemps[4]),fontsize=15,zorder=3)#'xkcd:bright pink'
# ax1.text(colorHD209[2]+0.1,colorHD209[1],'HD 209458b',color=scalarMap.to_rgba(equtemps[4]),fontsize=15,zorder=3)#'xkcd:bright pink'
ax1.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54.,mec=scalarMap.to_rgba(equtemps[5]),mfc=scalarMap.to_rgba(equtemps[5]),ecolor=scalarMap.to_rgba(equtemps[5]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorK7[2]+0.005,colorK7[0]+130,'KELT-7b',color=scalarMap.to_rgba(equtemps[5]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorK7v2[2],colorK7v2[1],xerr=colorK7v2[4],yerr=colorK7v2[3],mec='g',mfc='g',ecolor='g',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorK7[2]+0.03,colorK7[1]+0.02,'KELT-7b',color=scalarMap.to_rgba(equtemps[5]),fontsize=15,zorder=3)#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20.,mec=scalarMap.to_rgba(equtemps[7]),mfc=scalarMap.to_rgba(equtemps[7]),ecolor=scalarMap.to_rgba(equtemps[7]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW18[2]-0.13,colorW18[0]-25,'WASP-18b',color=scalarMap.to_rgba(equtemps[7]),fontsize=15,zorder=3)#xkcd:blue
ax1.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26.,mec=scalarMap.to_rgba(equtemps[8]),mfc=scalarMap.to_rgba(equtemps[8]),ecolor=scalarMap.to_rgba(equtemps[8]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW33[2]+0.02,colorW33[0],'WASP-33b',color=scalarMap.to_rgba(equtemps[8]),fontsize=15,zorder=3)#xkcd:sky blue bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23.,mec=scalarMap.to_rgba(equtemps[9]),mfc=scalarMap.to_rgba(equtemps[9]),ecolor=scalarMap.to_rgba(equtemps[9]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW43[2]-0.13,colorW43[0]-75,'WASP-43b',color=scalarMap.to_rgba(equtemps[9]),fontsize=15,zorder=3)#xkcd:red
ax1.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48.,mec=scalarMap.to_rgba(equtemps[10]),mfc=scalarMap.to_rgba(equtemps[10]),ecolor=scalarMap.to_rgba(equtemps[10]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW74[2]+0.01,colorW74[0]-100,'WASP-74b',color=scalarMap.to_rgba(equtemps[10]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# ax1.text(colorW74[2]+0.04,colorW74[1]-0.08,'WASP-74b',color=scalarMap.to_rgba(equtemps[10]),fontsize=15,zorder=3)#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27.,mec=scalarMap.to_rgba(equtemps[11]),mfc=scalarMap.to_rgba(equtemps[11]),ecolor=scalarMap.to_rgba(equtemps[11]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW76[2]-0.14,colorW76[0]-50,'WASP-76b',color=scalarMap.to_rgba(equtemps[11]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW76v2[2],colorW76v2[1],xerr=colorW76v2[4],yerr=colorW76v2[3],mec='k',mfc='k',ecolor='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.errorbar(colorW76v3[2],colorW76v3[1],xerr=colorW76v3[4],yerr=colorW76v3[3],mec='r',mfc='r',ecolor='r',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW76[2]+0.05,colorW76[1]-0.06,'WASP-76b',color=scalarMap.to_rgba(equtemps[11]),fontsize=15,zorder=3)#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58.,mec=scalarMap.to_rgba(equtemps[12]),mfc=scalarMap.to_rgba(equtemps[12]),ecolor=scalarMap.to_rgba(equtemps[12]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW79[2]+0.02,colorW79[0]+65,'WASP-79b',color=scalarMap.to_rgba(equtemps[12]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.text(colorW79[2]+0.01,colorW79[1]+0.05,'WASP-79b',color=scalarMap.to_rgba(equtemps[12]),fontsize=15,zorder=3)#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50.,mec=scalarMap.to_rgba(equtemps[13]),mfc=scalarMap.to_rgba(equtemps[13]),ecolor=scalarMap.to_rgba(equtemps[13]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW103[2]-0.14,colorW103[0]+20,'WASP-103b',color=scalarMap.to_rgba(equtemps[13]),fontsize=15,zorder=3)#xkcd:violet
ax1.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39.,mec=scalarMap.to_rgba(equtemps[14]),mfc=scalarMap.to_rgba(equtemps[14]),ecolor=scalarMap.to_rgba(equtemps[14]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW121[2]-0.15,colorW121[0],'WASP-121b',color=scalarMap.to_rgba(equtemps[14]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:light green

#would need to update error bars
ax1.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,mec=scalarMap.to_rgba(equtemps[15]),mfc=scalarMap.to_rgba(equtemps[15]),ecolor=scalarMap.to_rgba(equtemps[15]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorC2[2]-0.10,colorC2[0]-140,'CoRoT-2b',color=scalarMap.to_rgba(equtemps[15]),fontsize=15,zorder=3)#'xkcd:lilac'
ax1.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,mec=scalarMap.to_rgba(equtemps[16]),mfc=scalarMap.to_rgba(equtemps[16]),ecolor=scalarMap.to_rgba(equtemps[16]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorT3[2]-0.18,colorT3[0]-80,'TrES-3b',color=scalarMap.to_rgba(equtemps[16]),fontsize=15,zorder=3)#'xkcd:lilac'
ax1.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,mec=scalarMap.to_rgba(equtemps[17]),mfc=scalarMap.to_rgba(equtemps[17]),ecolor=scalarMap.to_rgba(equtemps[17]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW4[2]+0.01,colorW4[0]-80,'WASP-4b',color=scalarMap.to_rgba(equtemps[17]),fontsize=15,zorder=3)#'xkcd:lilac'
ax1.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,mec=scalarMap.to_rgba(equtemps[18]),mfc=scalarMap.to_rgba(equtemps[18]),ecolor=scalarMap.to_rgba(equtemps[18]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorKep13[2]+0.03,colorKep13[0]-100,'Kepler-13Ab',color=scalarMap.to_rgba(equtemps[18]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#'xkcd:lilac'
ax1.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,mec=scalarMap.to_rgba(equtemps[19]),mfc=scalarMap.to_rgba(equtemps[19]),ecolor=scalarMap.to_rgba(equtemps[19]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW12[2]+0.0555,colorW12[0]-20,'WASP-12b',color=scalarMap.to_rgba(equtemps[19]),fontsize=15,zorder=3)#'xkcd:lilac'
# ax1.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,mec=scalarMap.to_rgba(equtemps[20]),mfc=scalarMap.to_rgba(equtemps[20]),ecolor=scalarMap.to_rgba(equtemps[20]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW77[2]+0.0555,colorW77[0]-20,'WASP-77Ab',color=scalarMap.to_rgba(equtemps[20]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#'xkcd:lilac'
# ax1.errorbar(LTTcreal,colorLTT[0],xerr=colorLTT[4],yerr=328.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(0.27,1940,'WASP-77Ab',color='r',fontsize=15,fontweight='bold')

####### adding brown dwarfs from Manjavacas ###########################
for i in goodlist:
	ax1.errorbar(bdcolors[i,2],bdcolors[i,0],xerr=bdcolors[i,4],yerr=delTbd[i],color=pointcolor2,marker='.',linestyle='none',linewidth=3,zorder=4,markersize=15)

a = plt.axes([0.85,0.21,0.01,0.75], frameon=False)#plt.axes([0.76,0.20,0.01,0.75], frameon=False) 
a.yaxis.set_visible(False)
a.xaxis.set_visible(False)
a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.inferno, aspect='auto', visible=False)
cbar=plt.colorbar(a, fraction=3.0)
cbar.ax.tick_params(labelsize=15,width=2,length=6)
cbar.set_label('Equilibrium Temperature',fontsize=15)

plt.tight_layout()
plt.show()
############### MAIN FIGURE TEMPERATURE END ##########################################

################# INDIVIDUAL MODEL TRACKS ON TEMPERATURE AXIS #########################

# colors=pl.cm.inferno(np.linspace(0,0.9,5))

# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
# plt.plot(color_lowCO[:,2],color_lowCO[:,0],color=colors[1],marker='.',label='C/O=0.01',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.plot(color_highCO[:,2],color_highCO[:,0],color=colors[3],marker='.',label='C/O=0.85',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
# plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25) #($S_{H_{2}O}$)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=20)
# plt.ylim((1150,3550))
# plt.xlim((-0.2,0.5))
# plt.title('Vary Planet C/O',fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
# # plt.plot(color_delayTiO3600[:,2],color_delayTiO3600[:,0],color=colors[4],marker='.',label='TiO Delayed to 3600 K',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# # plt.plot(color_delayTiO2800[:,2],color_delayTiO2800[:,0],color=colors[3],marker='.',label='TiO Delayed to 2800 K',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_delayTiO3000[:,2],color_delayTiO3000[:,0],color=colors[4],marker='.',label='TiO/VO Delayed to 3000 K',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.plot(color_delayTiO2500[:,2],color_delayTiO2500[:,0],color=colors[3],marker='.',label='TiO/VO Delayed to 2500 K',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.plot(color_delayTiO2000[:,2],color_delayTiO2000[:,0],color=colors[2],marker='.',label='TiO/VO Delayed to 2000 K',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# # plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
# plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=20)
# plt.ylim((1150,3550))
# plt.xlim((-0.2,0.5))
# plt.title('Delayed TiO/VO',fontsize=20)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
# plt.plot(color_grav20[:,2],color_grav20[:,0],color=colors[1],marker='.',label='log(g)=2.0',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# #plt.plot(color_grav35[:,2],color_grav35[:,1],color='xkcd:green',marker='.',label='log(g)=3.5',markeredgecolor='xkcd:dark green',zorder=2)
# plt.plot(color_grav40[:,2],color_grav40[:,0],color=colors[3],marker='.',label='log(g)=4.0',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# # plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
# # plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=20)
# plt.ylim((1150,3550))
# plt.xlim((-0.2,0.5))
# plt.title('Vary Planet Gravity',fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
# plt.plot(color_metneg15[:,2],color_metneg15[:,0],color=colors[1],marker='.',label='[M/H]=-1.5',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# #plt.plot(color_metpos1[:,2],color_metpos1[:,1],color='xkcd:green',marker='.',label='[M/H]=1.0',markeredgecolor='xkcd:dark green',zorder=2)
# plt.plot(color_metpos15[:,2],color_metpos15[:,0],color=colors[3],marker='.',label='[M/H]=1.5',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# # plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
# plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25) #($S_{H_{2}O}$)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=20)
# plt.ylim((1150,3550))
# plt.xlim((-0.2,0.5))
# plt.title('Vary Planet Metallicity',fontsize=20)
# plt.tight_layout()
# plt.show()

# # plt.figure(figsize=(10,7))
# # plt.axvline(x=0.0,color='k',zorder=0)
# # plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)
# # plt.plot(color_star3300[:,2],color_star3300[:,0],color=colors[0],marker='.',label='T$_{eff}$=3300',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# # plt.plot(color_star4300[:,2],color_star4300[:,0],color=colors[1],marker='.',label='T$_{eff}$=4300',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# # plt.plot(color_star6300[:,2],color_star6300[:,0],color=colors[2],marker='.',label='T$_{eff}$=6300',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# # plt.plot(color_star7200[:,2],color_star7200[:,0],color=colors[3],marker='.',label='T$_{eff}$=7200',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# # plt.plot(color_star8200[:,2],color_star8200[:,0],color=colors[4],marker='.',label='T$_{eff}$=8200',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# # plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=38,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=65,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=9,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # # plt.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=12,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=15,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=13,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=45,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=22,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=14,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# # ax=plt.gca()
# # ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# # ax.set_yticklabels([],minor=True)
# # plt.ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# # plt.xlabel('Water Feature Strength',fontsize=20)
# # plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# # plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# # plt.legend(fontsize=15)
# # plt.ylim((1150,3550))
# # plt.xlim((-0.2,0.5))
# # plt.title('Vary Stellar Temperature',fontsize=15)
# # plt.tight_layout()
# # plt.show()

# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# # plt.plot(color_tint1percent[:,2],color_tint1percent[:,0],color=colors[1],marker='.',label='1% Effective T$_{int}$',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_tintTF18[:,2],color_tintTF18[:,0],color=colors[3],marker='.',label='Thorngren et al. 2019',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
# # plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=20)
# plt.ylim((1150,3550))
# plt.xlim((-0.2,0.5))
# plt.title('Vary Internal Heating',fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# # plt.plot(color_quench[:,2],color_quench[:,0],color=colors[4],marker='.',label='No Clouds',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.plot(color_fsedlow[:,2],color_fsedlow[:,0],color=colors[3],marker='.',label=r'$f_{sed}=0.1$',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.plot(color_fsedhigh[:,2],color_fsedhigh[:,0],color=colors[1],marker='.',label=r'$f_{sed}=1.0$',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
# plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# # plt.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
# plt.errorbar(colorW77[2],colorW77[0],xerr=colorW77[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
# plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=20)
# plt.ylim((1150,3550))
# plt.xlim((-0.2,0.5))
# plt.title('Vary Clouds',fontsize=20)
# plt.tight_layout()
# plt.show()

##################### END INDIVIDUAL MODEL PLOTS ####################################################################

######################### PLOT ALL MODELS INDIVIDUALLY ####################################
# plt.figure(figsize=(10,7))
# plt.axvline(x=0.0,color='k',zorder=0)
# plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)
# plt.plot(color_lowCO[:,2],color_lowCO[:,0],marker='.',label='[C/O]=0.01',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_highCO[:,2],color_highCO[:,0],marker='.',label='[C/O]=0.85',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_tintTF18[:,2],color_tintTF18[:,0],marker='.',label='Thorngren & Fortney 2018',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_metneg15[:,2],color_metneg15[:,0],marker='.',label='[M/H]=-1.5',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_metpos15[:,2],color_metpos15[:,0],marker='.',label='[M/H]=1.5',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_grav20[:,2],color_grav20[:,0],marker='.',label='log(g)=2.0',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_grav40[:,2],color_grav40[:,0],marker='.',label='log(g)=4.0',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_delayTiO3000[:,2],color_delayTiO3000[:,0],marker='.',label='TiO Delayed to 3000 K',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_delayTiO2500[:,2],color_delayTiO2500[:,0],marker='.',label='TiO Delayed to 2500 K',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.plot(color_delayTiO2000[:,2],color_delayTiO2000[:,0],marker='.',label='TiO Delayed to 2000 K',markeredgecolor='k',zorder=3,linewidth=2,markersize=10)
# plt.yticks(ticks=[1500,2000,2500,3000],labels=['1500','2000','2500','3000'])
# ax=plt.gca()
# ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
# ax.set_yticklabels([],minor=True)
# plt.ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# plt.xlabel('Water Feature Strength',fontsize=20)
# plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.legend(fontsize=15)
# plt.ylim((1250,3400))
# plt.xlim((-0.2,0.5))
# plt.title('Vary Planet [C/O]',fontsize=15)
# plt.tight_layout()
# plt.show()
########################### END PLOT MODELS INDIVIDUALLY ########################################

############################# DATA IN BLACK FOR A PROPOSAL #################################
# rc('axes',linewidth=2)
# fillhighCO=np.interp(color_lowCO[:,0],color_highCO[:,0],color_highCO[:,2])
# filllowCO=np.interp(color_highCO[:,0],color_lowCO[:,0],color_lowCO[:,2])
# filllowmet=np.interp(color_highCO[:,0],color_metpos15[:,0],color_metpos15[:,2])
# fillhighmet=np.interp(color_metneg15[:,0],color_lowCO[:,0],color_lowCO[:,2])
# fillheat=np.interp(color_tintTF18[:,0],color_delayTiO3000[:,0],color_delayTiO3000[:,2])
# fillg=np.interp(color_grav40[:,0],color_highCO[:,0],color_highCO[:,2])
# filltop=np.interp(color_lowCO[:,0],color_metpos15[:,0],color_metpos15[:,2])

# mycolor='#dbedff'#'xkcd:beige' #'#dbedff' color I used for the fill_between on the proposal 
# edgecolor='xkcd:dark blue'#'xkcd:brown' #'xkcd:dark blue'
# linecolor='xkcd:cerulean'# 'xkcd:light brown'#'xkcd:cerulean'
# linecolor2='xkcd:light brown'
# linecolor3='xkcd:olive green'
# pointcolor2='xkcd:brown'
# pointcolor3='xkcd:dark green'

# fig,ax1=plt.subplots(figsize=(10,7))
# ax1.set_ylim((1000,3550))
# ax1.set_xlim((-0.2,0.65))
# ax1.set_yticks(ticks=[1500,2000,2500,3000,3500])
# ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
# ax1.set_yticks(ticks=[1100,1200,1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800,2900,3100,3200,3300,3400],minor=True)
# ax1.set_yticklabels([],minor=True)
# ax1.set_ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# ax1.set_xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=20) #
# ax1.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

# ax1.axvline(x=0.0,color='xkcd:slate gray',zorder=1,linewidth=2)#
# ax1.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)#'xkcd:light brown','xkcd:brown'
# ax1.text(-0.18,3400,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
# ax1.plot(color_metpos15[:,2],color_metpos15[:,0],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
# ax1.text(0.09,2200,'[M/H]=1.5',color=linecolor2,fontsize=15,zorder=3,fontweight='bold')
# ax1.plot(color_tintTF18[:,2],color_tintTF18[:,0],color=linecolor3,marker='.',markeredgecolor=pointcolor3,zorder=2,linewidth=2,markersize=10)
# ax1.text(-0.06,1570,'TF18 Internal Heat',color=linecolor3,fontsize=15,zorder=3,fontweight='bold')

# ax1.fill_betweenx(color_lowCO[:,0],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=1)#'xkcd:beige'
# ax1.fill_betweenx(color_highCO[:,0],color_highCO[:,2],filllowCO,color=mycolor,zorder=1)#'xkcd:beige'
# ax1.fill_betweenx(color_metneg15[:,0],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=1)#'xkcd:beige'
# ax1.fill_betweenx(color_highCO[:,0],color_highCO[:,2],filllowmet,color=mycolor,zorder=1)
# ax1.fill_betweenx(color_tintTF18[:,0],color_tintTF18[:,2],fillheat,color=mycolor,zorder=1)
# ax1.fill_betweenx(color_grav40[:,0],color_grav40[:,2],fillg,color=mycolor,zorder=1)
# ax1.fill_betweenx(color_lowCO[:,0],color_lowCO[:,2],filltop,color=mycolor,zorder=1)

# ax1.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
# #ax1.text(colorH7[2]+0.01,colorH7[0]+20,'HAT-P-7b',color=scalarMap.to_rgba(equtemps[0]),fontsize=15,zorder=3)#'xkcd:kelly green'
# ax1.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorH32[2]-0.11,colorH32[0]-80,'HAT-P-32b',color=scalarMap.to_rgba(equtemps[1]),fontsize=15,zorder=3)#'xkcd:pink'
# #ax1.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
# #ax1.text(colorH41[2]+0.02,colorH41[0]-100,'HAT-P-41b',color=scalarMap.to_rgba(equtemps[2]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#'xkcd:light orange'
# ax1.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorHD189[2]+0.02,colorHD189[0]+20,'HD 189733b',color=scalarMap.to_rgba(equtemps[3]),fontsize=15,zorder=3)#'xkcd:greenish grey'
# ax1.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorHD209[2]+0.01,colorHD209[0]+30,'HD 209458b',color=scalarMap.to_rgba(equtemps[4]),fontsize=15,zorder=3)#'xkcd:bright pink'
# ax1.errorbar(colorK7v2[2],colorK7v2[0],xerr=colorK7v2[4],yerr=54.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorK7[2]+0.005,colorK7[0]+130,'KELT-7b',color=scalarMap.to_rgba(equtemps[5]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW18[2]-0.13,colorW18[0]-25,'WASP-18b',color=scalarMap.to_rgba(equtemps[7]),fontsize=15,zorder=3)#xkcd:blue
# ax1.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW33[2]+0.02,colorW33[0],'WASP-33b',color=scalarMap.to_rgba(equtemps[8]),fontsize=15,zorder=3)#xkcd:sky blue bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW43[2]-0.13,colorW43[0]-75,'WASP-43b',color=scalarMap.to_rgba(equtemps[9]),fontsize=15,zorder=3)#xkcd:red
# # ax1.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW74[2]+0.01,colorW74[0]-100,'WASP-74b',color=scalarMap.to_rgba(equtemps[10]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW76v2[2],colorW76v2[0],xerr=colorW76v2[4],yerr=27.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW76[2]-0.14,colorW76[0]-50,'WASP-76b',color=scalarMap.to_rgba(equtemps[11]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# # ax1.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW79[2]+0.02,colorW79[0]+65,'WASP-79b',color=scalarMap.to_rgba(equtemps[12]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW103[2]-0.14,colorW103[0]+20,'WASP-103b',color=scalarMap.to_rgba(equtemps[13]),fontsize=15,zorder=3)#xkcd:violet
# ax1.errorbar(colorW121v2[2],colorW121v2[0],xerr=colorW121v2[4],yerr=39.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW121[2]-0.15,colorW121[0],'WASP-121b',color=scalarMap.to_rgba(equtemps[14]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:light green
# # ax1.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorC2[2]-0.12,colorC2[0]-100,'CoRoT-2b',color=scalarMap.to_rgba(equtemps[15]),fontsize=15,zorder=3)#'xkcd:lilac'
# # ax1.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # #ax1.text(colorT3[2]-0.16,colorT3[0]-80,'TrES-3b',color=scalarMap.to_rgba(equtemps[16]),fontsize=15,zorder=3)#'xkcd:lilac'
# # ax1.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # #ax1.text(colorW4[2]+0.01,colorW4[0]-80,'WASP-4b',color=scalarMap.to_rgba(equtemps[17]),fontsize=15,zorder=3)#'xkcd:lilac'
# # ax1.errorbar(colorKep13v2[2],colorKep13v2[0],xerr=colorKep13v2[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# # #ax1.text(colorKep13[2]+0.03,colorKep13[0]-100,'Kepler-13Ab',color=scalarMap.to_rgba(equtemps[18]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#'xkcd:lilac'
# # ax1.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# #ax1.text(colorW12[2]+0.0555,colorW12[0]-20,'WASP-12b',color=scalarMap.to_rgba(equtemps[19]),fontsize=15,zorder=3)#'xkcd:lilac'

# ax1.errorbar(MA1creal,colorMA1[0],xerr=colorMA1[4],yerr=32.5,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(0.03,3150,'MASCARA-1b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(K16creal,colorK16[0],xerr=colorK16[4]/np.sqrt(2.),yerr=27.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(-0.15,2870,'KELT-16b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(K20creal,colorK20[0],xerr=colorK20[4],yerr=17.5,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(-0.16,2660,'KELT-20b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(W3creal,colorW3[0],xerr=colorW3[4]/np.sqrt(2.),yerr=30.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(-0.17,2380,'WASP-3b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(XO3creal3,colorXO3[0],xerr=colorXO3[4]/np.sqrt(2.),yerr=57.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(-0.03,2220,'XO-3b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(K11creal2,colorK11[0],xerr=colorK11[4]/np.sqrt(2.),yerr=71.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(-0.18,2080,'KELT-11b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(K8creal2,colorK8[0],xerr=colorK8[4]/np.sqrt(2.),yerr=33.5,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(-0.15,1800,'KELT-8b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(C2creal,colorC2[0],xerr=colorC2[4]/np.sqrt(2.),yerr=30.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(0.227,1950,'CoRoT-2b',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(W140creal,colorW140[0],xerr=colorW140[4]/np.sqrt(2.),yerr=55.,color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(0.43,1600,'WASP-140b',color='r',fontsize=15,fontweight='bold')

# plt.tight_layout()
# plt.show()
############################# END SIMPLIFIED PLOT ########################################

#plot against model equilibrium temp instead of dayside temp (almost the same)
# fig,ax1=plt.subplots(figsize=(10,7))
# # ax1.set_ylim((10.5,12.75))
# ax1.set_xlim((-0.2,0.65))
# # ax1.set_xlim((-0.2,0.75))
# # ax1.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# # ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
# # ax1.set_yticks(ticks=[10.537,10.779,11.174,11.336,11.481,11.611,11.834,11.930,12.018,12.100,12.244,12.308,12.368,12.424,12.526,12.572,12.616,12.658],minor=True)
# # ax1.set_yticklabels([],minor=True)
# ax1.set_ylabel('Equilibrium Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# ax1.set_xlabel('Water Feature Strength',fontsize=20)#log(Blackbody/In-Band Flux) Water Feature Strength
# # ax1.set_xlabel('Water Feature Strength',fontsize=20)#log(Blackbody/In-Band Flux) Water Feature Strength
# ax1.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
# ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

# ax1.axvline(x=0.0,color='k',zorder=1,linewidth=2)#'xkcd:slate gray'
# ax1.plot(color_fiducial[:,2],templist,color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)#'xkcd:light brown','xkcd:brown'
# # ax1.text(0.17,11.7,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')
# plt.tight_layout()
# plt.show()

#Dayside/nightside changes for phase curves

# fig,ax1=plt.subplots(figsize=(10,7))
# ax1.axvline(x=0.0,color='k',zorder=1,linewidth=2)#'xkcd:slate gray'
# ax1.plot(color_fiducial[:,2],color_fiducial[:,1],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)#'xkcd:light brown','xkcd:brown'
# ax1.text(0.17,11.66,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
# ax1.fill_betweenx(color_lowCO[:,1],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_highCO[:,1],color_highCO[:,2],filllowCO,color=mycolor,zorder=0)#'xkcd:beige'
# #ax1.fill_betweenx(color_metpos15[:,1],color_metpos15[:,2],filllowmet,color='xkcd:grey')
# ax1.fill_betweenx(color_metneg15[:,1],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=0)#'xkcd:beige'
# #ax1.fill_betweenx(color_grav25[:,1],color_grav25[:,2],filldelaylow1,color='xkcd:grey')
# ax1.fill_betweenx(color_star8200[:,1],color_star8200[:,2],filldelaylow2,color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_star8200[-3:,1],color_star8200[-3:,2],fillstar[-3:],color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_delayTiO2800[:,1],color_delayTiO2800[:,2],filldelayhigh2,color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_metneg15[:5,1],color_metneg15[:5,2],fillthebottom1[:5],color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_grav20[:,1],color_grav20[:,2],fillthebottom2,color=mycolor,zorder=0)#'xkcd:beige'
# ax1.fill_betweenx(color_tint1percent[:,1],color_tint1percent[:,2],fillthebottom3,color=mycolor,zorder=0)#'xkcd:beige'

# ax1.errorbar(colorW103[2],colorW103[1],xerr=colorW103[4],yerr=colorW103[3],mec=scalarMap.to_rgba(equtemps[13]),mfc=scalarMap.to_rgba(equtemps[13]),ecolor=scalarMap.to_rgba(equtemps[13]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW103[2]-0.17,colorW103[1]+0.03,'WASP-103b',color=scalarMap.to_rgba(equtemps[13]),fontsize=15,zorder=3)#xkcd:violet
# ax1.errorbar(colorW103night[2],colorW103night[1],xerr=colorW103night[4],yerr=colorW103night[3],mec=scalarMap.to_rgba(equtemps[13]),mfc=scalarMap.to_rgba(equtemps[13]),ecolor=scalarMap.to_rgba(equtemps[13]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.plot([colorW103[2],colorW103night[2]],[colorW103[1],colorW103night[1]],linewidth=2,color=scalarMap.to_rgba(equtemps[13]),zorder=4)
# ax1.errorbar(colorW18[2],colorW18[1],xerr=colorW18[4],yerr=colorW18[3],mec=scalarMap.to_rgba(equtemps[7]),mfc=scalarMap.to_rgba(equtemps[7]),ecolor=scalarMap.to_rgba(equtemps[7]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW18[2]-0.15,colorW18[1]-0.05,'WASP-18b',color=scalarMap.to_rgba(equtemps[7]),fontsize=15,zorder=3)#xkcd:blue
# ax1.errorbar(colorW18night[2],colorW18night[1],xerr=colorW18night[4],yerr=colorW18night[3],mec=scalarMap.to_rgba(equtemps[7]),mfc=scalarMap.to_rgba(equtemps[7]),ecolor=scalarMap.to_rgba(equtemps[7]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.plot([colorW18[2],colorW18night[2]],[colorW18[1],colorW18night[1]],linewidth=2,color=scalarMap.to_rgba(equtemps[7]),zorder=4)
# ax1.errorbar(colorW43[2],colorW43[1],xerr=colorW43[4],yerr=colorW43[3],mec=scalarMap.to_rgba(equtemps[9]),mfc=scalarMap.to_rgba(equtemps[9]),ecolor=scalarMap.to_rgba(equtemps[9]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.text(colorW43[2]-0.13,colorW43[1]-0.13,'WASP-43b',color=scalarMap.to_rgba(equtemps[9]),fontsize=15,zorder=3)#xkcd:red
# ax1.errorbar(colorW43night[2],colorW43night[1],xerr=colorW43night[4],yerr=colorW43night[3],mec=scalarMap.to_rgba(equtemps[9]),mfc=scalarMap.to_rgba(equtemps[9]),ecolor=scalarMap.to_rgba(equtemps[9]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
# ax1.plot([colorW43[2],colorW43night[2]],[colorW43[1],colorW43night[1]],linewidth=2,color=scalarMap.to_rgba(equtemps[9]),zorder=4)

# ax1.set_xlabel('log(Blackbody/In-Band Flux) [erg/s/m$^{2}$]',fontsize=20)
# ax1.set_ylabel('log(Blackbody Flux) [erg/s/m$^{2}$]',fontsize=20)
# ax1.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# ax1.set_ylim((10.3,12.8))
# ax1.set_xlim((-0.2,0.65))

# a = plt.axes([0.85,0.20,0.01,0.75], frameon=False)#plt.axes([0.76,0.20,0.01,0.75], frameon=False) 
# a.yaxis.set_visible(False)
# a.xaxis.set_visible(False)
# a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.inferno, aspect='auto', visible=False)
# cbar=plt.colorbar(a, fraction=3.0)
# cbar.ax.tick_params(labelsize=15,width=2,length=6)
# cbar.set_label('Equilibrium Temperature',fontsize=15)

# plt.tight_layout()
# plt.show()
#The one for a proposal

#I broke the proposals somehow.
# greys = cm = plt.get_cmap('binary_r')
xerrors=np.array([colorH7[4],colorH32[4],colorH41[4],colorHD189[4],colorHD209[4],colorK7[4],\
	colorKep13[4],colorW18[4],colorW33[4],colorW43[4],colorW74[4],colorW76[4],colorW79[4],\
	colorW103[4],colorW121[4]])
# cNorm  = mplcolors.Normalize(vmin=np.min(xerrors), vmax=np.max(xerrors)+0.02)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=greys)

# fig,ax1=plt.subplots(figsize=(10,7))
# ax1.axvline(x=0.0,color='xkcd:slate gray',zorder=1,linewidth=2)
# ax1.plot(color_fiducial[:,2],color_fiducial[:,1],color='xkcd:light brown',marker='.',label='Fiducial',markeredgecolor='xkcd:brown',zorder=3,linewidth=2,markersize=10)
# ax1.text(0.32,10.7,'Fiducial',color='xkcd:light brown',fontsize=15,zorder=3,fontweight='bold')
# # ax1.plot(color_tintTF18[:,2],color_tintTF18[:,1],color='xkcd:sky blue',marker='.',markeredgecolor='xkcd:blue',zorder=2,linewidth=2,markersize=10)
# # ax1.text(0.37,11.1,'Tint=TF18',color='xkcd:sky blue',fontsize=15,zorder=3,fontweight='bold')
# # ax1.plot(color_lowCO[:,2],color_lowCO[:,1],color='xkcd:olive green',marker='.',markeredgecolor='xkcd:dark green',zorder=2,linewidth=2,markersize=10)
# # ax1.text(0.5,11.3,'[C/O]=0.01',color='xkcd:olive green',fontsize=15,zorder=3,fontweight='bold')

# ax1.fill_betweenx(color_lowCO[:,1],fillhighCO,color_lowCO[:,2],color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_highCO[:,1],color_highCO[:,2],filllowCO,color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_metneg15[:,1],color_metneg15[:,2],fillhighmet,color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_star8200[:,1],color_star8200[:,2],filldelaylow2,color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_star8200[-3:,1],color_star8200[-3:,2],fillstar[-3:],color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_delayTiO2800[:,1],color_delayTiO2800[:,2],filldelayhigh2,color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_metneg15[:5,1],color_metneg15[:5,2],fillthebottom1[:5],color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_grav20[:,1],color_grav20[:,2],fillthebottom2,color='xkcd:beige',zorder=0)
# ax1.fill_betweenx(color_tint1percent[:,1],color_tint1percent[:,2],fillthebottom3,color='xkcd:beige',zorder=0)

#calpha=1.0
#colorscale=np.zeros(np.shape(xerrors)[0])
#for i in np.arange(np.shape(xerrors)[0]):
tarray=np.log(xerrors)
colorscale=1.0-(tarray-np.min(tarray))/(np.max(tarray)-np.min(tarray)+1.)
# f=12
# #try greying with alphas instead
# ax1.errorbar(colorH7[2],colorH7[1],xerr=colorH7[4],yerr=colorH7[3],marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[0])#,mec=scalarMap.to_rgba(xerrors[0])[0],mfc=scalarMap.to_rgba(xerrors[0])[0],ecolor=scalarMap.to_rgba(xerrors[0])
# ax1.text(colorH7[2]+0.01,colorH7[1]+0.05,'HAT-P-7b',color='k',fontsize=f,zorder=3,alpha=colorscale[0])#'xkcd:kelly green'mec=,mfc=scalarMap.to_rgba(xerrors[0]),ecolor=scalarMap.to_rgba(xerrors[0])
# ax1.errorbar(colorH32[2],colorH32[1],xerr=colorH32[4],yerr=colorH32[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[1])#mec=scalarMap.to_rgba(xerrors[1])[0],mfc=scalarMap.to_rgba(xerrors[1])[0],ecolor=scalarMap.to_rgba(xerrors[1]),
# ax1.text(colorH32[2]-0.1,colorH32[1]-0.12,'HAT-P-32b',color='k',fontsize=f,zorder=3,alpha=colorscale[1])#'xkcd:pink'
# ax1.errorbar(colorH41[2],colorH41[1],xerr=colorH41[4],yerr=colorH41[3],marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[2])#mec=scalarMap.to_rgba(xerrors[2])[0],mfc=scalarMap.to_rgba(xerrors[2])[0],ecolor=scalarMap.to_rgba(xerrors[2]),
# ax1.text(colorH41[2]+0.02,colorH41[1]-0.1,'HAT-P-41b',color='k',fontsize=f,zorder=3,alpha=colorscale[2])#'xkcd:light orange'
# ax1.errorbar(colorHD189[2],colorHD189[1],xerr=colorHD189[4],yerr=colorHD189[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[3])#mec=scalarMap.to_rgba(xerrors[3])[0],mfc=scalarMap.to_rgba(xerrors[3])[0],ecolor=scalarMap.to_rgba(xerrors[3]),
# ax1.text(colorHD189[2]-0.13,colorHD189[1]+0.06,'HD 189733b',color='k',fontsize=f,zorder=3,alpha=colorscale[3])#'xkcd:greenish grey'
# ax1.errorbar(colorHD209[2],colorHD209[1],xerr=colorHD209[4],yerr=colorHD209[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[4])#,mec=scalarMap.to_rgba(xerrors[4])[0],mfc=scalarMap.to_rgba(xerrors[4])[0],ecolor=scalarMap.to_rgba(xerrors[4])
# ax1.text(colorHD209[2]+0.0,colorHD209[1]-0.08,'HD 209458b',color='k',fontsize=f,zorder=3,alpha=colorscale[4])#'xkcd:bright pink'
# ax1.errorbar(colorK7[2],colorK7[1],xerr=colorK7[4],yerr=colorK7[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[5])#mec=scalarMap.to_rgba(xerrors[5])[0],mfc=scalarMap.to_rgba(xerrors[5])[0],ecolor=scalarMap.to_rgba(xerrors[5]),
# ax1.text(colorK7[2]+0.03,colorK7[1]+0.02,'KELT-7b',color='k',fontsize=f,zorder=3,alpha=colorscale[5])#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[6])#mec=scalarMap.to_rgba(xerrors[6])[0],mfc=scalarMap.to_rgba(xerrors[6])[0],ecolor=scalarMap.to_rgba(xerrors[6]),
# ax1.text(colorKep13[2]+0.05,colorKep13[1]+0.01,'Kepler-13Ab',color='k',fontsize=f,zorder=3,alpha=colorscale[6])#'xkcd:lilac'
# ax1.errorbar(colorW18[2],colorW18[1],xerr=colorW18[4],yerr=colorW18[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[7])#mec=scalarMap.to_rgba(xerrors[7])[0],mfc=scalarMap.to_rgba(xerrors[7])[0],ecolor=scalarMap.to_rgba(xerrors[7]),
# ax1.text(colorW18[2]-0.11,colorW18[1]-0.05,'WASP-18b',color='k',fontsize=f,zorder=3,alpha=colorscale[7])#xkcd:blue
# ax1.errorbar(colorW33[2],colorW33[1],xerr=colorW33[4],yerr=colorW33[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[8])#mec=scalarMap.to_rgba(xerrors[8])[0],mfc=scalarMap.to_rgba(xerrors[8])[0],ecolor=scalarMap.to_rgba(xerrors[8]),
# ax1.text(colorW33[2]+0.02,colorW33[1]-0.03,'WASP-33b',color='k',fontsize=f,zorder=3,alpha=colorscale[8])#xkcd:sky blue
# ax1.errorbar(colorW43[2],colorW43[1],xerr=colorW43[4],yerr=colorW43[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[9])#mec=scalarMap.to_rgba(xerrors[9])[0],mfc=scalarMap.to_rgba(xerrors[9])[0],ecolor=scalarMap.to_rgba(xerrors[9]),
# ax1.text(colorW43[2]-0.1,colorW43[1]-0.11,'WASP-43b',color='k',fontsize=f,zorder=3,alpha=colorscale[9])#xkcd:red
# ax1.errorbar(colorW74[2],colorW74[1],xerr=colorW74[4],yerr=colorW74[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[10])#mec=scalarMap.to_rgba(xerrors[10])[0],mfc=scalarMap.to_rgba(xerrors[10])[0],ecolor=scalarMap.to_rgba(xerrors[10]),
# ax1.text(colorW74[2]+0.06,colorW74[1]-0.0,'WASP-74b',color='k',fontsize=f,zorder=3,alpha=colorscale[10])#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW76[2],colorW76[1],xerr=colorW76[4],yerr=colorW76[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[11])#mec=scalarMap.to_rgba(xerrors[11])[0],mfc=scalarMap.to_rgba(xerrors[11])[0],ecolor=scalarMap.to_rgba(xerrors[11]),
# ax1.text(colorW76[2]+0.05,colorW76[1]-0.06,'WASP-76b',color='k',fontsize=f,zorder=3,alpha=colorscale[11])#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW79[2],colorW79[1],xerr=colorW79[4],yerr=colorW79[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[12])#mec=scalarMap.to_rgba(xerrors[12])[0],mfc=scalarMap.to_rgba(xerrors[12])[0],ecolor=scalarMap.to_rgba(xerrors[12]),
# ax1.text(colorW79[2]+0.03,colorW79[1]+0.08,'WASP-79b',color='k',fontsize=f,zorder=3,alpha=colorscale[12])#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax1.errorbar(colorW103[2],colorW103[1],xerr=colorW103[4],yerr=colorW103[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[13])#mec=scalarMap.to_rgba(xerrors[13])[0],mfc=scalarMap.to_rgba(xerrors[13])[0],ecolor=scalarMap.to_rgba(xerrors[13]),
# ax1.text(colorW103[2]-0.13,colorW103[1]+0.03,'WASP-103b',color='k',fontsize=f,zorder=3,alpha=colorscale[13])#xkcd:violet
# ax1.errorbar(colorW121[2],colorW121[1],xerr=colorW121[4],yerr=colorW121[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[14])#mec=scalarMap.to_rgba(xerrors[14])[0],mfc=scalarMap.to_rgba(xerrors[14])[0],ecolor=scalarMap.to_rgba(xerrors[14]),
# ax1.text(colorW121[2]-0.15,colorW121[1]-0.02,'WASP-121b',color='k',fontsize=f,zorder=3,alpha=colorscale[14])#xkcd:light green
# ax1.errorbar(W77creal,colorW77[1],xerr=colorW77[4]/np.sqrt(3.),yerr=colorW77[3]/np.sqrt(3.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(0.36,11.6,'WASP-77Ab',color='r',fontsize=15,fontweight='bold')
# ax1.errorbar(W77creal2,colorW77[1],xerr=colorW77[4]/np.sqrt(3.),yerr=colorW77[3]/np.sqrt(3.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# #ax1.plot([W77creal,W77creal2],[colorW77[1],colorW77[1]],color='r',linestyle=':',linewidth=2)
# ax1.errorbar(W77creal3,colorW77[1],xerr=colorW77[4]/np.sqrt(3.),yerr=colorW77[3]/np.sqrt(3.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.plot([W77creal,W77creal3],[colorW77[1],colorW77[1]],color='r',linestyle=':',linewidth=2)
# # ax1.errorbar(K23creal,colorK23[1],xerr=colorK23[4]/np.sqrt(2.),yerr=colorK23[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(K23creal-0.1,colorK23[1]-0.05,'K23A',color='r',fontsize=15)
# # ax1.errorbar(K231creal,colorK231[1],xerr=colorK231[4]/np.sqrt(2.),yerr=colorK231[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(K231creal-0.1,colorK231[1]-0.05,'K2-31',color='r',fontsize=15)
# # ax1.errorbar(K3creal,colorK3[1],xerr=colorK3[4],yerr=colorK3[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(K3creal-0.1,colorK3[1]-0.05,'K3',color='r',fontsize=15)
# # ax1.errorbar(K4creal,colorK4[1],xerr=colorK4[4]/np.sqrt(2.),yerr=colorK4[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(K4creal-0.1,colorK4[1]-0.05,'K4A',color='r',fontsize=15)
# ax1.errorbar(W14creal,colorW14[1],xerr=colorW14[4],yerr=colorW14[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax1.text(W14creal-0.1,colorW14[1]-0.05,'W14',color='r',fontsize=15)
# # ax1.errorbar(H36creal,colorH36[1],xerr=colorH36[4],yerr=colorH36[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(H36creal-0.1,colorH36[1]-0.05,'H36',color='r',fontsize=15)
# # ax1.errorbar(H32fcreal,colorH32f[1],xerr=colorH32f[4],yerr=colorH32f[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(H32fcreal-0.1,colorH32f[1]-0.05,'H32',color='r',fontsize=15)
# # ax1.errorbar(W95creal,colorW95[1],xerr=colorW95[4],yerr=colorW95[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax1.text(W95creal-0.1,colorW95[1]-0.05,'W95',color='r',fontsize=15)
# ax1.set_xlabel('log(Blackbody/In-Band Flux)',fontsize=20)
# ax1.set_ylabel('log(Blackbody Flux) [erg/s/m$^{2}$]',fontsize=20)
# ax1.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# ax1.set_ylim((10.3,12.8))
# ax1.set_xlim((-0.2,0.65))
# ax2=ax1.twinx()

# ax2.set_ylim((10.3,12.8))
# ax2.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# ax2.set_yticklabels(['1500','2000','2500','3000','3500'])
# ax2.set_ylabel('Dayside Temperature [K]',fontsize=20)
# ax2.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# plt.tight_layout()
# plt.show()

# #colormag plot with only temperatures plotted
# fig,ax1=plt.subplots(figsize=(10,7))
# ax1.set_ylim((10.5,12.7))
# ax1.set_xlim((-0.2,0.65))
# ax1.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
# ax1.set_yticks(ticks=[10.537,10.779,11.174,11.336,11.481,11.611,11.834,11.930,12.018,12.100,12.244,12.308,12.368,12.424,12.526,12.572,12.616,12.658],minor=True)
# ax1.set_yticklabels([],minor=True)
# ax1.set_ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# ax1.set_xlabel('Water Feature Strength',fontsize=20)#log(Blackbody/In-Band Flux)
# ax1.tick_params(which='major',labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

# ax2=ax1.twinx()
# ax2.axvline(x=0.0,color='k',zorder=1,linewidth=2)
# ax2.plot(color_fiducial[:,2],color_fiducial[:,1],color='xkcd:cerulean',marker='.',label='Fiducial',markeredgecolor='xkcd:dark blue',zorder=3,linewidth=2,markersize=10)
# ax2.text(0.32,10.7,'Fiducial',color='xkcd:dark blue',fontsize=15,zorder=3)
# ax2.plot(color_tintTF18[:,2],color_tintTF18[:,1],color='xkcd:light brown',marker='.',markeredgecolor='xkcd:brown',zorder=3,linewidth=2,markersize=10)
# ax2.text(0.1,11.1,'TF18 Internal Heat',color='xkcd:light brown',fontsize=15,zorder=3)
# ax2.plot(color_lowCO[:,2],color_lowCO[:,1],color='xkcd:olive green',marker='.',markeredgecolor='xkcd:dark green',zorder=3,linewidth=2,markersize=10)
# ax2.text(0.5,11.3,'[C/O]=0.01',color='xkcd:olive green',fontsize=15,zorder=3)


# mycolor='#dbedff'#'xkcd:carolina blue'#'#e6e1de'#'xkcd:light grey'#'xkcd:pale lavender'
# fillalpha=1.0
# ax2.fill_betweenx(color_lowCO[:,1],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_highCO[:,1],color_highCO[:,2],filllowCO,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_metneg15[:,1],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_star8200[:,1],color_star8200[:,2],filldelaylow2,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_star8200[-3:,1],color_star8200[-3:,2],fillstar[-3:],color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_delayTiO2800[:,1],color_delayTiO2800[:,2],filldelayhigh2,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_metneg15[:5,1],color_metneg15[:5,2],fillthebottom1[:5],color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_grav20[:,1],color_grav20[:,2],fillthebottom2,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_tint1percent[:,1],color_tint1percent[:,2],fillthebottom3,color=mycolor,zorder=0,alpha=fillalpha)

# pointcolor='k'#'xkcd:cobalt blue'
# ax2.errorbar(colorH7[2],colorH7[1],xerr=colorH7[4],yerr=colorH7[3],marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[0])#,mec=scalarMap.to_rgba(xerrors[0])[0],mfc=scalarMap.to_rgba(xerrors[0])[0],ecolor=scalarMap.to_rgba(xerrors[0])
# ax2.text(colorH7[2]+0.01,colorH7[1]+0.05,'HAT-P-7b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[0])#'xkcd:kelly green'mec=,mfc=scalarMap.to_rgba(xerrors[0]),ecolor=scalarMap.to_rgba(xerrors[0])
# ax2.errorbar(colorH32[2],colorH32[1],xerr=colorH32[4],yerr=colorH32[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[1])#mec=scalarMap.to_rgba(xerrors[1])[0],mfc=scalarMap.to_rgba(xerrors[1])[0],ecolor=scalarMap.to_rgba(xerrors[1]),
# ax2.text(colorH32[2]-0.1,colorH32[1]-0.12,'HAT-P-32b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[1])#'xkcd:pink'
# ax2.errorbar(colorH41[2],colorH41[1],xerr=colorH41[4],yerr=colorH41[3],marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[2])#mec=scalarMap.to_rgba(xerrors[2])[0],mfc=scalarMap.to_rgba(xerrors[2])[0],ecolor=scalarMap.to_rgba(xerrors[2]),
# ax2.text(colorH41[2]+0.02,colorH41[1]-0.1,'HAT-P-41b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[2])#'xkcd:light orange'
# ax2.errorbar(colorHD189[2],colorHD189[1],xerr=colorHD189[4],yerr=colorHD189[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[3])#mec=scalarMap.to_rgba(xerrors[3])[0],mfc=scalarMap.to_rgba(xerrors[3])[0],ecolor=scalarMap.to_rgba(xerrors[3]),
# ax2.text(colorHD189[2]-0.13,colorHD189[1]+0.06,'HD 189733b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[3])#'xkcd:greenish grey'
# ax2.errorbar(colorHD209[2],colorHD209[1],xerr=colorHD209[4],yerr=colorHD209[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[4])#,mec=scalarMap.to_rgba(xerrors[4])[0],mfc=scalarMap.to_rgba(xerrors[4])[0],ecolor=scalarMap.to_rgba(xerrors[4])
# ax2.text(colorHD209[2]+0.0,colorHD209[1]-0.08,'HD 209458b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[4])#'xkcd:bright pink'
# ax2.errorbar(colorK7[2],colorK7[1],xerr=colorK7[4],yerr=colorK7[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[5])#mec=scalarMap.to_rgba(xerrors[5])[0],mfc=scalarMap.to_rgba(xerrors[5])[0],ecolor=scalarMap.to_rgba(xerrors[5]),
# ax2.text(colorK7[2]+0.03,colorK7[1]+0.02,'KELT-7b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[5])#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# # ax2.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[6])#mec=scalarMap.to_rgba(xerrors[6])[0],mfc=scalarMap.to_rgba(xerrors[6])[0],ecolor=scalarMap.to_rgba(xerrors[6]),
# # ax2.text(colorKep13[2]+0.05,colorKep13[1]+0.01,'Kepler-13Ab',color='k',fontsize=f,zorder=3,alpha=colorscale[6])#'xkcd:lilac'
# ax2.errorbar(colorW18[2],colorW18[1],xerr=colorW18[4],yerr=colorW18[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[7])#mec=scalarMap.to_rgba(xerrors[7])[0],mfc=scalarMap.to_rgba(xerrors[7])[0],ecolor=scalarMap.to_rgba(xerrors[7]),
# ax2.text(colorW18[2]-0.11,colorW18[1]-0.05,'WASP-18b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[7])#xkcd:blue
# ax2.errorbar(colorW33[2],colorW33[1],xerr=colorW33[4],yerr=colorW33[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[8])#mec=scalarMap.to_rgba(xerrors[8])[0],mfc=scalarMap.to_rgba(xerrors[8])[0],ecolor=scalarMap.to_rgba(xerrors[8]),
# ax2.text(colorW33[2]+0.02,colorW33[1]-0.03,'WASP-33b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[8])#xkcd:sky blue
# ax2.errorbar(colorW43[2],colorW43[1],xerr=colorW43[4],yerr=colorW43[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[9])#mec=scalarMap.to_rgba(xerrors[9])[0],mfc=scalarMap.to_rgba(xerrors[9])[0],ecolor=scalarMap.to_rgba(xerrors[9]),
# ax2.text(colorW43[2]-0.1,colorW43[1]-0.11,'WASP-43b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[9])#xkcd:red
# ax2.errorbar(colorW74[2],colorW74[1],xerr=colorW74[4],yerr=colorW74[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[10])#mec=scalarMap.to_rgba(xerrors[10])[0],mfc=scalarMap.to_rgba(xerrors[10])[0],ecolor=scalarMap.to_rgba(xerrors[10]),
# ax2.text(colorW74[2]+0.06,colorW74[1]-0.0,'WASP-74b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[10])#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# ax2.errorbar(colorW76[2],colorW76[1],xerr=colorW76[4],yerr=colorW76[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[11])#mec=scalarMap.to_rgba(xerrors[11])[0],mfc=scalarMap.to_rgba(xerrors[11])[0],ecolor=scalarMap.to_rgba(xerrors[11]),
# ax2.text(colorW76[2]+0.05,colorW76[1]-0.06,'WASP-76b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[11])#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax2.errorbar(colorW79[2],colorW79[1],xerr=colorW79[4],yerr=colorW79[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[12])#mec=scalarMap.to_rgba(xerrors[12])[0],mfc=scalarMap.to_rgba(xerrors[12])[0],ecolor=scalarMap.to_rgba(xerrors[12]),
# ax2.text(colorW79[2]+0.03,colorW79[1]+0.08,'WASP-79b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[12])#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax2.errorbar(colorW103[2],colorW103[1],xerr=colorW103[4],yerr=colorW103[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[13])#mec=scalarMap.to_rgba(xerrors[13])[0],mfc=scalarMap.to_rgba(xerrors[13])[0],ecolor=scalarMap.to_rgba(xerrors[13]),
# ax2.text(colorW103[2]-0.13,colorW103[1]+0.03,'WASP-103b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[13])#xkcd:violet
# ax2.errorbar(colorW121[2],colorW121[1],xerr=colorW121[4],yerr=colorW121[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[14])#mec=scalarMap.to_rgba(xerrors[14])[0],mfc=scalarMap.to_rgba(xerrors[14])[0],ecolor=scalarMap.to_rgba(xerrors[14]),
# ax2.text(colorW121[2]-0.15,colorW121[1]-0.02,'WASP-121b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[14])#xkcd:light green
# ax2.errorbar(W77creal,colorW77[1],xerr=colorW77[4]/np.sqrt(2.),yerr=colorW77[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.text(0.3,11.65,'WASP-77Ab',color='r',fontsize=15,fontweight='bold')
# ax2.errorbar(W77creal2,colorW77[1],xerr=colorW77[4]/np.sqrt(2.),yerr=colorW77[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(W77creal3,colorW77[1],xerr=colorW77[4]/np.sqrt(2.),yerr=colorW77[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.plot([W77creal,W77creal3],[colorW77[1],colorW77[1]],color='r',linestyle=':',linewidth=2)

# ax2.set_ylim((10.5,12.7))
# ax2.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# ax2.set_yticklabels([])
# ax2.set_yticks(ticks=[10.537,10.779,11.174,11.336,11.481,11.611,11.834,11.930,12.018,12.100,12.244,12.308,12.368,12.424,12.526,12.572,12.616,12.658],minor=True)
# ax2.set_yticklabels([],minor=True)
# ax2.tick_params(which='major',labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# ax2.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.tight_layout()
# plt.show()

# fig,ax1=plt.subplots(figsize=(10,7))
# ax1.set_ylim((10.5,12.7))
# ax1.set_xlim((-0.2,0.65))
# ax1.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
# ax1.set_yticks(ticks=[10.537,10.779,11.174,11.336,11.481,11.611,11.834,11.930,12.018,12.100,12.244,12.308,12.368,12.424,12.526,12.572,12.616,12.658],minor=True)
# ax1.set_yticklabels([],minor=True)
# ax1.set_ylabel('Dayside Temperature [K]',fontsize=20)#Blackbody Temperature [K]
# ax1.set_xlabel('Water Feature Strength',fontsize=20)#log(Blackbody/In-Band Flux)
# ax1.tick_params(which='major',labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

# ax2=ax1.twinx()
# ax2.axvline(x=0.0,color='k',zorder=1,linewidth=2)
# ax2.plot(color_fiducial[:,2],color_fiducial[:,1],color='xkcd:cerulean',marker='.',label='Fiducial',markeredgecolor='xkcd:dark blue',zorder=3,linewidth=2,markersize=10)
# ax2.text(0.32,10.7,'Fiducial',color='xkcd:dark blue',fontsize=15,zorder=3)
# ax2.plot(color_tintTF18[:,2],color_tintTF18[:,1],color='xkcd:light brown',marker='.',markeredgecolor='xkcd:brown',zorder=3,linewidth=2,markersize=10)
# ax2.text(0.1,11.15,'TF18 Internal Heat',color='xkcd:light brown',fontsize=15,zorder=3)
# ax2.plot(color_lowCO[:,2],color_lowCO[:,1],color='xkcd:olive green',marker='.',markeredgecolor='xkcd:dark green',zorder=3,linewidth=2,markersize=10)
# ax2.text(0.5,11.3,'[C/O]=0.01',color='xkcd:olive green',fontsize=15,zorder=3)

# mycolor='#dbedff'#'xkcd:carolina blue'#'#e6e1de'#'xkcd:light grey'#'xkcd:pale lavender'
# fillalpha=1.0
# ax2.fill_betweenx(color_lowCO[:,1],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_highCO[:,1],color_highCO[:,2],filllowCO,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_metneg15[:,1],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_star8200[:,1],color_star8200[:,2],filldelaylow2,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_star8200[-3:,1],color_star8200[-3:,2],fillstar[-3:],color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_delayTiO2800[:,1],color_delayTiO2800[:,2],filldelayhigh2,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_metneg15[:5,1],color_metneg15[:5,2],fillthebottom1[:5],color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_grav20[:,1],color_grav20[:,2],fillthebottom2,color=mycolor,zorder=0,alpha=fillalpha)
# ax2.fill_betweenx(color_tint1percent[:,1],color_tint1percent[:,2],fillthebottom3,color=mycolor,zorder=0,alpha=fillalpha)

# pointcolor='k'#'xkcd:cobalt blue'
# ax2.errorbar(colorH7[2],colorH7[1],xerr=colorH7[4],yerr=colorH7[3],marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#,mec=scalarMap.to_rgba(xerrors[0])[0],mfc=scalarMap.to_rgba(xerrors[0])[0],ecolor=scalarMap.to_rgba(xerrors[0])
# ax2.text(colorH7[2]+0.01,colorH7[1]+0.05,'HAT-P-7b',color=pointcolor,fontsize=f,zorder=3)#'xkcd:kelly green'mec=,mfc=scalarMap.to_rgba(xerrors[0]),ecolor=scalarMap.to_rgba(xerrors[0])
# ax2.errorbar(colorH32[2],colorH32[1],xerr=colorH32[4],yerr=colorH32[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[1])[0],mfc=scalarMap.to_rgba(xerrors[1])[0],ecolor=scalarMap.to_rgba(xerrors[1]),
# ax2.text(colorH32[2]-0.1,colorH32[1]-0.12,'HAT-P-32b',color=pointcolor,fontsize=f,zorder=3)#'xkcd:pink'
# #ax2.errorbar(colorH41[2],colorH41[1],xerr=colorH41[4],yerr=colorH41[3],marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[2])#mec=scalarMap.to_rgba(xerrors[2])[0],mfc=scalarMap.to_rgba(xerrors[2])[0],ecolor=scalarMap.to_rgba(xerrors[2]),
# #ax2.text(colorH41[2]+0.02,colorH41[1]-0.1,'HAT-P-41b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[2])#'xkcd:light orange'
# ax2.errorbar(colorHD189[2],colorHD189[1],xerr=colorHD189[4],yerr=colorHD189[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[3])[0],mfc=scalarMap.to_rgba(xerrors[3])[0],ecolor=scalarMap.to_rgba(xerrors[3]),
# ax2.text(colorHD189[2]-0.13,colorHD189[1]+0.06,'HD 189733b',color=pointcolor,fontsize=f,zorder=3)#'xkcd:greenish grey'
# ax2.errorbar(colorHD209[2],colorHD209[1],xerr=colorHD209[4],yerr=colorHD209[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#,mec=scalarMap.to_rgba(xerrors[4])[0],mfc=scalarMap.to_rgba(xerrors[4])[0],ecolor=scalarMap.to_rgba(xerrors[4])
# ax2.text(colorHD209[2]+0.0,colorHD209[1]-0.08,'HD 209458b',color=pointcolor,fontsize=f,zorder=3)#'xkcd:bright pink'
# #ax2.errorbar(colorK7[2],colorK7[1],xerr=colorK7[4],yerr=colorK7[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[5])#mec=scalarMap.to_rgba(xerrors[5])[0],mfc=scalarMap.to_rgba(xerrors[5])[0],ecolor=scalarMap.to_rgba(xerrors[5]),
# #ax2.text(colorK7[2]+0.03,colorK7[1]+0.02,'KELT-7b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[5])#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
# # ax2.errorbar(colorKep13[2],colorKep13[1],xerr=colorKep13[4],yerr=colorKep13[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc='k',ecolor='k',alpha=colorscale[6])#mec=scalarMap.to_rgba(xerrors[6])[0],mfc=scalarMap.to_rgba(xerrors[6])[0],ecolor=scalarMap.to_rgba(xerrors[6]),
# # ax2.text(colorKep13[2]+0.05,colorKep13[1]+0.01,'Kepler-13Ab',color='k',fontsize=f,zorder=3,alpha=colorscale[6])#'xkcd:lilac'
# ax2.errorbar(colorW18[2],colorW18[1],xerr=colorW18[4],yerr=colorW18[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[7])[0],mfc=scalarMap.to_rgba(xerrors[7])[0],ecolor=scalarMap.to_rgba(xerrors[7]),
# ax2.text(colorW18[2]-0.11,colorW18[1]-0.05,'WASP-18b',color=pointcolor,fontsize=f,zorder=3)#xkcd:blue
# ax2.errorbar(colorW33[2],colorW33[1],xerr=colorW33[4],yerr=colorW33[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[8])[0],mfc=scalarMap.to_rgba(xerrors[8])[0],ecolor=scalarMap.to_rgba(xerrors[8]),
# ax2.text(colorW33[2]+0.02,colorW33[1]-0.03,'WASP-33b',color=pointcolor,fontsize=f,zorder=3)#xkcd:sky blue
# ax2.errorbar(colorW43[2],colorW43[1],xerr=colorW43[4],yerr=colorW43[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[9])[0],mfc=scalarMap.to_rgba(xerrors[9])[0],ecolor=scalarMap.to_rgba(xerrors[9]),
# ax2.text(colorW43[2]-0.1,colorW43[1]-0.11,'WASP-43b',color=pointcolor,fontsize=f,zorder=3)#xkcd:red
# #ax2.errorbar(colorW74[2],colorW74[1],xerr=colorW74[4],yerr=colorW74[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[10])#mec=scalarMap.to_rgba(xerrors[10])[0],mfc=scalarMap.to_rgba(xerrors[10])[0],ecolor=scalarMap.to_rgba(xerrors[10]),
# #ax2.text(colorW74[2]+0.06,colorW74[1]-0.0,'WASP-74b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[10])#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
# #ax2.errorbar(colorW76[2],colorW76[1],xerr=colorW76[4],yerr=colorW76[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[11])#mec=scalarMap.to_rgba(xerrors[11])[0],mfc=scalarMap.to_rgba(xerrors[11])[0],ecolor=scalarMap.to_rgba(xerrors[11]),
# #ax2.text(colorW76[2]+0.05,colorW76[1]-0.06,'WASP-76b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[11])#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# #ax2.errorbar(colorW79[2],colorW79[1],xerr=colorW79[4],yerr=colorW79[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor,alpha=colorscale[12])#mec=scalarMap.to_rgba(xerrors[12])[0],mfc=scalarMap.to_rgba(xerrors[12])[0],ecolor=scalarMap.to_rgba(xerrors[12]),
# #ax2.text(colorW79[2]+0.03,colorW79[1]+0.08,'WASP-79b',color=pointcolor,fontsize=f,zorder=3,alpha=colorscale[12])#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
# ax2.errorbar(colorW103[2],colorW103[1],xerr=colorW103[4],yerr=colorW103[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[13])[0],mfc=scalarMap.to_rgba(xerrors[13])[0],ecolor=scalarMap.to_rgba(xerrors[13]),
# ax2.text(colorW103[2]-0.13,colorW103[1]+0.03,'WASP-103b',color=pointcolor,fontsize=f,zorder=3)#xkcd:violet
# ax2.errorbar(colorW121[2],colorW121[1],xerr=colorW121[4],yerr=colorW121[3],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4,mec='k',mfc=pointcolor,ecolor=pointcolor)#mec=scalarMap.to_rgba(xerrors[14])[0],mfc=scalarMap.to_rgba(xerrors[14])[0],ecolor=scalarMap.to_rgba(xerrors[14]),
# ax2.text(colorW121[2]-0.15,colorW121[1]-0.02,'WASP-121b',color=pointcolor,fontsize=f,zorder=3)#xkcd:light green
# ax2.errorbar(MA1creal2,colorMA1[1],xerr=colorMA1[4],yerr=colorMA1[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(K16creal2,colorK16[1],xerr=colorK16[4]/np.sqrt(2.),yerr=colorK16[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(K20creal2,colorK20[1],xerr=colorK20[4],yerr=colorK20[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(W3creal2,colorW3[1],xerr=colorW3[4]/np.sqrt(2.),yerr=colorW3[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# #ax2.errorbar(W173creal2,colorW173[1],xerr=colorW173[4]/np.sqrt(2.),yerr=colorW173[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# #ax2.errorbar(W14creal2,colorW14[1],xerr=colorW14[4]/np.sqrt(2.),yerr=colorW14[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(K11creal2,colorK11[1],xerr=colorK11[4],yerr=colorK11[3],color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(K8creal2,colorK8[1],xerr=colorK8[4]/np.sqrt(2.),yerr=colorK8[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(W77creal2,colorW77[1],xerr=colorW77[4]/np.sqrt(2.),yerr=colorW77[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(C2creal2,colorC2[1],xerr=colorC2[4]/np.sqrt(2.),yerr=colorC2[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(W140creal2,colorW140[1],xerr=colorW140[4]/np.sqrt(2.),yerr=colorW140[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# ax2.errorbar(XO3creal2,colorXO3[1],xerr=colorXO3[4]/np.sqrt(2.),yerr=colorXO3[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# #ax2.text(0.36,11.6,'WASP-77Ab',color='r',fontsize=15,fontweight='bold')
# # ax2.errorbar(W77creal2,colorW77[1],xerr=colorW77[4]/np.sqrt(2.),yerr=colorW77[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax2.errorbar(W77creal3,colorW77[1],xerr=colorW77[4]/np.sqrt(2.),yerr=colorW77[3]/np.sqrt(2.),color='r',mec='k',ecolor='r',marker='*',markersize=20,linestyle='none',linewidth=3,zorder=5)
# # ax2.plot([W77creal,W77creal3],[colorW77[1],colorW77[1]],color='r',linestyle=':',linewidth=2)

# ax2.set_ylim((10.5,12.7))
# ax2.set_yticks(ticks=[10.989,11.727,12.174,12.477,12.697])
# ax2.set_yticklabels([])
# ax2.set_yticks(ticks=[10.537,10.779,11.174,11.336,11.481,11.611,11.834,11.930,12.018,12.100,12.244,12.308,12.368,12.424,12.526,12.572,12.616,12.658],minor=True)
# ax2.set_yticklabels([],minor=True)
# ax2.tick_params(which='major',labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# ax2.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
# plt.tight_layout()
# plt.show()

#plot temps versus each other
# bbodtemps=np.array([2762.,1935.,2415.,1442.,1705.,2445.,3468.\
# 	,2968.,3116.,1769.,2201.,2768.,2075.,3007.,2662.])


############### FIT TIRR VS TDAY ###########################################
# def fittemps(p,tirr,tday):
# 	y=p[0]*tirr+p[1]
# 	return np.array(tday-y)

# #H7,H32,H41,HD189,HD209,K7,Kep13,W18,W33,W43,W74,W76,W79,W103,W121
# tday=np.array([colorH7[0],colorH32[0],colorH41[0],colorHD189[0],colorHD209[0],colorK7[0],colorW18[0],\
# 	colorW33[0],colorW43[0],colorW74[0],colorW76[0],colorW79[0],colorW103[0],colorW121[0]])
# tdayerr=np.array([38.,58.,65.,9.,26.,54.,12.,15.,13.,45.,22.,57.,20.,14.])
# tirr=np.array([3169.,2685.,2737.,1719.,2052.,2895.,3370.,3932.,2048.,2717.,3095.,2489.,3541.,3334.])

# params0=[0.8,-90.]
# tfit=leastsq(fittemps,params0,args=(tirr,tday))

# def lnprob(theta,y,yerr,tirr):
# 	lp=lnprior(theta)
# 	return lp+lnlike(theta,y,yerr,tirr)

# def lnlike(theta,y,yerr,tirr):
# 	model = theta[0]*tirr+theta[1]
# 	resid=y-model
# 	chi2=np.sum((resid/yerr)**2)
# 	dof=np.shape(y)[0]-1.
# 	chi2red=chi2/dof
# 	ln_likelihood=-0.5*(np.sum((resid/yerr)**2 + np.log(2.0*np.pi*(yerr)**2)))
# 	return ln_likelihood

# def lnprior(theta):
# 	lnpriorprob=0.
# 	return lnpriorprob

# ndim=2
# nwalkers=100
# theta=tfit[0]

# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (tday,tdayerr,tirr), pool=mp.Pool())
# pos = [theta + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

# bestfit=np.zeros(ndim+1)
# bestfit[0]=10.**8
# for i, result in enumerate(sampler.sample(pos, iterations=1000)):
# 	print("step", i)
# 	if i>200:
# 		for guy in np.arange(nwalkers):
# 			resid=fittemps(result.coords[guy],tirr,tday)
# 			chi2val=np.sum((resid//tdayerr)**2.)
# 			if chi2val<bestfit[0]:
# 				bestfit[0]=chi2val
# 				bestfit[1:]=result.coords[guy]

# samples = sampler.chain[:, 200:, :].reshape((-1, ndim))
# def quantile(x, q):
# 	return np.percentile(x, [100. * qi for qi in q])
# bestfit=0.807*np.linspace(1500,4000,100)+71

# plt.figure()
# plt.errorbar(tirr,tday,tdayerr,color='k',marker='.',linestyle='none')
# plt.plot(np.linspace(1500,4000,100),bestfit,color='r')
# plt.show()

# #ttemp=1900.
def getflux(ttemp):
	toutpoint1=make_bbod(np.array([1.275]),np.array([0.11]),ttemp)
	toutpoint2=make_bbod(np.array([1.57]),np.array([0.08]),ttemp)
	tinpoint=make_bbod(np.array([1.415]),np.array([0.13]),ttemp)
	testtemp,testflux,trash=colormagmod(toutpoint1[0],toutpoint2[0],tinpoint[0])
	#print(testflux)
	return testflux

# #plot showing three spectra and blackbody fits
# #just doing ratio of blackbodies here because that's good enough
# fpfsbW43=(np.exp(h*c/kb/(masterwavegrid*10**-6.)/4520.)-1)/(np.exp(h*c/kb/(masterwavegrid*10**-6.)/1683.)-1)*rprsW43**2.
# fpfsbW121=(np.exp(h*c/kb/(masterwavegrid*10**-6.)/6460.)-1)/(np.exp(h*c/kb/(masterwavegrid*10**-6.)/2550.)-1)*rprsW121**2.
# fpfsbW103=(np.exp(h*c/kb/(masterwavegrid*10**-6.)/6110.)-1)/(np.exp(h*c/kb/(masterwavegrid*10**-6.)/2932.)-1)*rprsW103**2.
# fpfsbH7=(np.exp(h*c/kb/(masterwavegrid*10**-6.)/6441.)-1)/(np.exp(h*c/kb/(masterwavegrid*10**-6.)/2693.)-1)*rprsH7**2.
# fpfsbH32=(np.exp(h*c/kb/(masterwavegrid*10**-6.)/6207.)-1)/(np.exp(h*c/kb/(masterwavegrid*10**-6.)/1843.)-1)*rprsH32**2.

# #plt.figure(figsize=(10,7))
# fig,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True)
# ax1.errorbar(W43[:,0],W43[:,1],yerr=W43[:,2],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=0,mec='k',mfc='k',ecolor='k',label='WASP-43b (Kreidberg et al. 2014)')
# ax2.errorbar(W121[:,0],W121[:,1],yerr=W121[:,2],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=0,mec='k',mfc='k',ecolor='k',label='WASP-121b (Evans et al. 2017)')
# #ax3.errorbar(W103[:,0],W103[:,1],yerr=W103[:,2],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=0,mec='b',mfc='b',ecolor='b',label='WASP-103b (Kreidberg et al. 2018)')
# ax3.errorbar(H7[:,0],H7[:,1]*10**4.,yerr=H7[:,2]*10**4.,marker='.',markersize=15,linestyle='none',linewidth=3,zorder=0,mec='k',mfc='k',ecolor='k',label='HAT-P-7b')
# #ax2.errorbar(H32[:,0],H32[:,1],yerr=H32[:,2],marker='.',markersize=15,linestyle='none',linewidth=3,zorder=0,mec='b',mfc='b',ecolor='b',label='HAT-P-32b')
# ax1.plot(masterwavegrid,fpfsbW43*10**6.,color='k',linewidth=2)
# ax2.plot(masterwavegrid,fpfsbW121*10**6.,color='k',linewidth=2)
# #ax3.plot(masterwavegrid,fpfsbW103*10**6.,color='b',linewidth=2)
# ax3.plot(masterwavegrid,fpfsbH7*10**6.,color='k',linewidth=2)
# #ax2.plot(masterwavegrid,fpfsbH32*10**6.,color='b',linewidth=2)
# plt.xlim((1.1,1.7))
# #plt.legend(fontsize=20,loc='upper right')
# plt.xlabel('Wavelength [$\mu$m]',fontsize=20)
# ax2.set_ylabel('Fp/Fs [ppm]',fontsize=20)
# ax1.set_title('WASP-43b (Kreidberg et al. 2014)',fontsize=15)
# ax2.set_title('WASP-121b (Evans et al. 2017)',fontsize=15)
# ax3.set_title('HAT-P-7b (Mansfield et al. 2018)',fontsize=15)
# ax1.tick_params(labelsize=20,axis="both",right=True,top=True,width=2,length=8,direction='in')
# ax2.tick_params(labelsize=20,axis="both",right=True,top=True,width=2,length=8,direction='in')
# ax3.tick_params(labelsize=20,axis="both",right=True,top=True,width=2,length=8,direction='in')
# plt.show()

##########################Interpolate Mike's fiducial models to make the explanation plot

# datwave=W43[:,0]
# datflux=W43[:,1]*10**-6.
# daterr=W43[:,2]*10**-6.
# diff=np.diff(W43[:,0])[0]
# plwave=waveW43
# plflux=fluxW43
# plrprs=rprsW43

# tempwavegrid=masterwavegrid*10**-6.

# def indplanetbbod(pltemp,plwave,startemp,starmet,starlogg,rprs):
# 	plfinebbod=fakedata_bbod(tempwavegrid,pltemp) #erg/s/m^2/m
# 	sp = S.Icat('k93models',startemp,starmet,starlogg)	#Parameters go temp, metallicity, logg
# 	sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
# 	wave=sp.wave*10.**-10  #in meters
# 	flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
# 	starfinebbod=np.interp(tempwavegrid,wave,flux)
# 	# starfinebbod=fakedata_bbod(masterwavegrid,startemp)
# 	diff=np.mean(np.diff(plwave))*10**-6.

# 	newminwave=np.min(plwave)*10**-6.
# 	newmaxwave=np.max(plwave)*10**-6.
# 	smallgrid=tempwavegrid[(tempwavegrid>newminwave-diff/2.)&(tempwavegrid<newmaxwave+diff/2.)]
# 	smallplflux=plfinebbod[(tempwavegrid>newminwave-diff/2.)&(tempwavegrid<newmaxwave+diff/2.)]
# 	smallstarflux=starfinebbod[(tempwavegrid>newminwave-diff/2.)&(tempwavegrid<newmaxwave+diff/2.)]
# 	tgrid=tempwavegrid[(tempwavegrid>newminwave)&(tempwavegrid<newmaxwave)]
# 	modelbbod=np.zeros(np.shape(tgrid)[0])
# 	for i in np.arange(np.shape(tgrid)[0]):
# 		wave1=tgrid[i]-diff/2.
# 		wave2=tgrid[i]+diff/2.
# 		plint=trapezoidint(smallgrid[(smallgrid>wave1)&(smallgrid<wave2)],smallplflux[(smallgrid>wave1)&(smallgrid<wave2)])
# 		starint=trapezoidint(smallgrid[(smallgrid>wave1)&(smallgrid<wave2)],smallstarflux[(smallgrid>wave1)&(smallgrid<wave2)])
# 		modelbbod[i]=(plint/starint*rprs**2.)*10**6. #in ppm
# 	return tgrid*10**6.,modelbbod

# tgridW43,BmodelW43=indplanetbbod(1775.,datwave,4520.,-0.01,4.645,0.1558)
# plotBbod=np.interp(datwave,tgridW43,BmodelW43)

# W43mod=np.loadtxt('modelW43.txt')
# interpmodel=np.interp(datwave,W43mod[:,0],W43mod[:,1]*10**6.)
# plt.figure(figsize=(8.5,6))
# # plt.errorbar(datwave,datinpunits,yerr=errinpunits,color='b',marker='.',markersize=15,linestyle='none',linewidth=3)
# # plt.plot(datwave,binneddownmod[:,63],marker='.',markersize=15,linewidth=3,markeredgecolor='xkcd:brown',color='xkcd:light brown')
# # plt.plot(datwave,planetbbod,marker='.',markersize=15,linewidth=3,markeredgecolor='k',color='xkcd:slate gray')
# # plt.errorbar(datwave,datflux*10**6.,yerr=daterr*10**6.,color='b',marker='.',markersize=15,linestyle='none',linewidth=3,label='WASP-43b, Kreidberg et al. (2014)',zorder=2)
# plt.plot(datwave,interpmodel,marker='.',markersize=15,linewidth=3,markeredgecolor='k',color='xkcd:slate gray',label='Fiducial Model',zorder=1)
# plt.plot(datwave,plotBbod,marker='d',markersize=9,linewidth=3,markeredgecolor='k',color='xkcd:slate gray',label='Blackbody',zorder=1)#slate gray
# plt.fill_between([1.22,1.33],[200,200],[800,800],color='xkcd:orange',alpha=0.3,zorder=0,edgecolor='none')
# plt.fill_between([1.53,1.61],[200,200],[800,800],color='xkcd:orange',alpha=0.3,zorder=0,edgecolor='none')
# plt.fill_between([1.35,1.48],[200,200],[800,800],color='g',alpha=0.3,zorder=0,edgecolor='none')
# plt.text(1.415,250,'In-Band',fontsize=20,color='g',horizontalalignment='center')
# plt.text(1.275,250,'Out-of-\nBand',fontsize=20,color='xkcd:orange',horizontalalignment='center')
# plt.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
# plt.xlabel('Wavelength [$\mu$m]',fontsize=20)
# plt.ylabel('F$_{p}$/F$_{s}$ [ppm]',fontsize=20)
# plt.legend(fontsize=15,loc='upper center')
# plt.xlim((1.1,1.7))
# plt.ylim((225,800))
# plt.tight_layout()
# plt.show()

#################### CALCULATE CHI-SQUARED FOR DATA FITTING EACH MODEL IN COLOR-MAG SPACE #########################

# xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorH32[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW18[2],colorW103[2],colorW33[2]])
# ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorH32[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW18[1],colorW103[1],colorW33[1]])
# xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorH32[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW18[4],colorW103[4],colorW33[4]])
# yerrs=np.array([colorHD189[3],colorHD209[3],colorW43[3],colorH32[3],colorW79[3],colorW74[3],colorH41[3],colorK7[3],colorW76[3],colorW121[3],colorH7[3],colorW18[3],colorW103[3],colorW33[3]])

#with stare mode
xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorC2[2],colorT3[2],colorH32[2],colorW4[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2],colorW18[2],colorW103[2],colorW33[2],colorKep13[2]])
ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorC2[1],colorT3[1],colorH32[1],colorW4[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1],colorW18[1],colorW103[1],colorW33[1],colorKep13[1]])
xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorC2[4],colorT3[4],colorH32[4],colorW4[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4],colorW18[4],colorW103[4],colorW33[4],colorKep13[4]])
yerrs=np.array([colorHD189[3],colorHD209[3],colorW43[3],colorC2[3],colorT3[3],colorH32[3],colorW4[3],colorW79[3],colorW74[3],colorH41[3],colorK7[3],colorW76[3],colorW121[3],colorH7[3],colorW12[3],colorW18[3],colorW103[3],colorW33[3],colorKep13[3]])

#with W77
# xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorC2[2],colorT3[2],colorH32[2],colorW77[2],colorW4[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2],colorW18[2],colorW103[2],colorW33[2],colorKep13[2]])
# ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorC2[1],colorT3[1],colorH32[1],colorW77[1],colorW4[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1],colorW18[1],colorW103[1],colorW33[1],colorKep13[1]])
# xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorC2[4],colorT3[4],colorH32[4],colorW77[4],colorW4[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4],colorW18[4],colorW103[4],colorW33[4],colorKep13[4]])
# yerrs=np.array([colorHD189[3],colorHD209[3],colorW43[3],colorC2[3],colorT3[3],colorH32[3],colorW77[3],colorW4[3],colorW79[3],colorW74[3],colorH41[3],colorK7[3],colorW76[3],colorW121[3],colorH7[3],colorW12[3],colorW18[3],colorW103[3],colorW33[3],colorKep13[3]])

numpoints=19.
# xlist=np.array([colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2],colorW18[2],colorW103[2],colorW33[2],colorKep13[2]])
# ylist=np.array([colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1],colorW18[1],colorW103[1],colorW33[1],colorKep13[1]])
# xerrs=np.array([colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4],colorW18[4],colorW103[4],colorW33[4],colorKep13[4]])
# yerrs=np.array([colorW74[3],colorH41[3],colorK7[3],colorW76[3],colorW121[3],colorH7[3],colorW12[3],colorW18[3],colorW103[3],colorW33[3],colorKep13[3]])
# xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorC2[2],colorT3[2],colorH32[2],colorW4[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2]])
# ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorC2[1],colorT3[1],colorH32[1],colorW4[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1]])
# xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorC2[4],colorT3[4],colorH32[4],colorW4[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4]])
# yerrs=np.array([colorHD189[3],colorHD209[3],colorW43[3],colorC2[3],colorT3[3],colorH32[3],colorW4[3],colorW79[3],colorW74[3],colorH41[3],colorK7[3],colorW76[3],colorW121[3],colorH7[3],colorW12[3]])

xvals_fiducial=np.interp(ylist,color_fiducial[:,1],color_fiducial[:,2])
chi2_fiducial=np.sum(((xvals_fiducial-xlist)/xerrs)**2.)
chi2red_fiducial=chi2_fiducial/numpoints
signif_fiducial=stats.chi2.sf(chi2_fiducial,numpoints)
sigma_fiducial=special.erfinv(1-signif_fiducial)*np.sqrt(2.)

xvals_lowCO=np.interp(ylist,color_lowCO[:,1],color_lowCO[:,2])
chi2_lowCO=np.sum(((xvals_lowCO-xlist)/xerrs)**2.)
chi2red_lowCO=chi2_lowCO/numpoints
signif_lowCO=stats.chi2.sf(chi2_lowCO,numpoints)
sigma_lowCO=special.erfinv(1-signif_lowCO)*np.sqrt(2.)

xvals_highCO=np.interp(ylist,color_highCO[:,1],color_highCO[:,2])
chi2_highCO=np.sum(((xvals_highCO-xlist)/xerrs)**2.)
chi2red_highCO=chi2_highCO/numpoints
signif_highCO=stats.chi2.sf(chi2_highCO,numpoints)
sigma_highCO=special.erfinv(1-signif_highCO)*np.sqrt(2.)

# xvals_delayTiO1600=np.interp(ylist,color_delayTiO1600[:,1],color_delayTiO1600[:,2])
# chi2_delayTiO1600=np.sum(((xvals_delayTiO1600-xlist)/xerrs)**2.)
# chi2red_delayTiO1600=chi2_delayTiO1600/14.
# signif_delayTiO1600=stats.chi2.sf(chi2_delayTiO1600,14)
# sigma_delayTiO1600=special.erfinv(1-signif_delayTiO1600)*np.sqrt(2.)

# xvals_delayTiO1800=np.interp(ylist,color_delayTiO1800[:,1],color_delayTiO1800[:,2])
# chi2_delayTiO1800=np.sum(((xvals_delayTiO1800-xlist)/xerrs)**2.)
# chi2red_delayTiO1800=chi2_delayTiO1800/14.
# signif_delayTiO1800=stats.chi2.sf(chi2_delayTiO1800,14)
# sigma_delayTiO1800=special.erfinv(1-signif_delayTiO1800)*np.sqrt(2.)

xvals_delayTiO2000=np.interp(ylist,color_delayTiO2000[:,1],color_delayTiO2000[:,2])
chi2_delayTiO2000=np.sum(((xvals_delayTiO2000-xlist)/xerrs)**2.)
chi2red_delayTiO2000=chi2_delayTiO2000/numpoints
signif_delayTiO2000=stats.chi2.sf(chi2_delayTiO2000,numpoints)
sigma_delayTiO2000=special.erfinv(1-signif_delayTiO2000)*np.sqrt(2.)

xvals_delayTiO2500=np.interp(ylist,color_delayTiO2500[:,1],color_delayTiO2500[:,2])
chi2_delayTiO2500=np.sum(((xvals_delayTiO2500-xlist)/xerrs)**2.)
chi2red_delayTiO2500=chi2_delayTiO2500/numpoints
signif_delayTiO2500=stats.chi2.sf(chi2_delayTiO2500,numpoints)
sigma_delayTiO2500=special.erfinv(1-signif_delayTiO2500)*np.sqrt(2.)

xvals_delayTiO3000=np.interp(ylist,color_delayTiO3000[:,1],color_delayTiO3000[:,2])
chi2_delayTiO3000=np.sum(((xvals_delayTiO3000-xlist)/xerrs)**2.)
chi2red_delayTiO3000=chi2_delayTiO3000/numpoints
signif_delayTiO3000=stats.chi2.sf(chi2_delayTiO3000,numpoints)
sigma_delayTiO3000=special.erfinv(1-signif_delayTiO3000)*np.sqrt(2.)

xvals_grav20=np.interp(ylist,color_grav20[:,1],color_grav20[:,2])
chi2_grav20=np.sum(((xvals_grav20-xlist)/xerrs)**2.)
chi2red_grav20=chi2_grav20/numpoints
signif_grav20=stats.chi2.sf(chi2_grav20,numpoints)
sigma_grav20=special.erfinv(1-signif_grav20)*np.sqrt(2.)

xvals_grav40=np.interp(ylist,color_grav40[:,1],color_grav40[:,2])
chi2_grav40=np.sum(((xvals_grav40-xlist)/xerrs)**2.)
chi2red_grav40=chi2_grav40/numpoints
signif_grav40=stats.chi2.sf(chi2_grav40,numpoints)
sigma_grav40=special.erfinv(1-signif_grav40)*np.sqrt(2.)

xvals_metneg15=np.interp(ylist,color_metneg15[:,1],color_metneg15[:,2])
chi2_metneg15=np.sum(((xvals_metneg15-xlist)/xerrs)**2.)
chi2red_metneg15=chi2_metneg15/numpoints
signif_metneg15=stats.chi2.sf(chi2_metneg15,numpoints)
sigma_metneg15=special.erfinv(1-signif_metneg15)*np.sqrt(2.)

xvals_metpos15=np.interp(ylist,color_metpos15[:,1],color_metpos15[:,2])
chi2_metpos15=np.sum(((xvals_metpos15-xlist)/xerrs)**2.)
chi2red_metpos15=chi2_metpos15/numpoints
signif_metpos15=stats.chi2.sf(chi2_metpos15,numpoints)
sigma_metpos15=special.erfinv(1-signif_metpos15)*np.sqrt(2.)

# xvals_star3300=np.interp(ylist,color_star3300[:,1],color_star3300[:,2])
# chi2_star3300=np.sum(((xvals_star3300-xlist)/xerrs)**2.)
# chi2red_star3300=chi2_star3300/14.
# signif_star3300=stats.chi2.sf(chi2_star3300,14)
# sigma_star3300=special.erfinv(1-signif_star3300)*np.sqrt(2.)

# xvals_star4300=np.interp(ylist,color_star4300[:,1],color_star4300[:,2])
# chi2_star4300=np.sum(((xvals_star4300-xlist)/xerrs)**2.)
# chi2red_star4300=chi2_star4300/14.
# signif_star4300=stats.chi2.sf(chi2_star4300,14)
# sigma_star4300=special.erfinv(1-signif_star4300)*np.sqrt(2.)

# xvals_star6300=np.interp(ylist,color_star6300[:,1],color_star6300[:,2])
# chi2_star6300=np.sum(((xvals_star6300-xlist)/xerrs)**2.)
# chi2red_star6300=chi2_star6300/14.
# signif_star6300=stats.chi2.sf(chi2_star6300,14)
# sigma_star6300=special.erfinv(1-signif_star6300)*np.sqrt(2.)

# xvals_star7200=np.interp(ylist,color_star7200[:,1],color_star7200[:,2])
# chi2_star7200=np.sum(((xvals_star7200-xlist)/xerrs)**2.)
# chi2red_star7200=chi2_star7200/14.
# signif_star7200=stats.chi2.sf(chi2_star7200,14)
# sigma_star7200=special.erfinv(1-signif_star7200)*np.sqrt(2.)

# xvals_star8200=np.interp(ylist,color_star8200[:,1],color_star8200[:,2])
# chi2_star8200=np.sum(((xvals_star8200-xlist)/xerrs)**2.)
# chi2red_star8200=chi2_star8200/14.
# signif_star8200=stats.chi2.sf(chi2_star8200,14)
# sigma_star8200=special.erfinv(1-signif_star8200)*np.sqrt(2.)

# xvals_tint1percent=np.interp(ylist,color_tint1percent[:,1],color_tint1percent[:,2])
# chi2_tint1percent=np.sum(((xvals_tint1percent-xlist)/xerrs)**2.)
# chi2red_tint1percent=chi2_tint1percent/14.
# signif_tint1percent=stats.chi2.sf(chi2_tint1percent,14)
# sigma_tint1percent=special.erfinv(1-signif_tint1percent)*np.sqrt(2.)

xvals_tintTF18=np.interp(ylist,color_tintTF18[:,1],color_tintTF18[:,2])
chi2_tintTF18=np.sum(((xvals_tintTF18-xlist)/xerrs)**2.)
chi2red_tintTF18=chi2_tintTF18/numpoints
signif_tintTF18=stats.chi2.sf(chi2_tintTF18,numpoints)
sigma_tintTF18=special.erfinv(1-signif_tintTF18)*np.sqrt(2.)

#Giving each planet an appropriate gravity compared to the fiducial model:
#With W77
# xvals_changegrav=np.array([xvals_fiducial[0],xvals_fiducial[1],xvals_grav40[2],xvals_fiducial[3],\
# 	xvals_fiducial[4],xvals_grav20[5],xvals_fiducial[6],xvals_fiducial[7],xvals_fiducial[8],xvals_fiducial[9],\
# 	xvals_fiducial[10],xvals_fiducial[11],xvals_grav40[12],xvals_fiducial[13],xvals_fiducial[14],\
# 	xvals_fiducial[15],xvals_grav40[16],xvals_fiducial[17],xvals_fiducial[18],xvals_grav40[19]])

#without
xvals_changegrav=np.array([xvals_fiducial[0],xvals_fiducial[1],xvals_grav40[2],xvals_fiducial[3],\
	xvals_fiducial[4],xvals_grav20[5],xvals_fiducial[6],xvals_fiducial[7],xvals_fiducial[8],\
	xvals_fiducial[9],xvals_fiducial[10],xvals_grav40[11],xvals_fiducial[12],xvals_fiducial[13],\
	xvals_fiducial[14],xvals_grav40[15],xvals_fiducial[16],xvals_fiducial[17],xvals_grav40[18]])

# xvals_changegrav=np.array([xvals_fiducial[0],xvals_fiducial[1],xvals_grav40[2],xvals_fiducial[3],\
# 	xvals_fiducial[4],xvals_grav20[5],xvals_fiducial[6],xvals_fiducial[7],\
# 	xvals_fiducial[8],xvals_fiducial[9],xvals_grav40[10],xvals_fiducial[11],xvals_fiducial[12],\
# 	xvals_fiducial[13],xvals_grav40[14],xvals_fiducial[15],xvals_fiducial[16],xvals_grav40[17]])

# lowgravs=xvals_grav20[0,2,3,4,5,7]
# midgravs=xvals_fiducial[6,9,11]
# highgravs=xvals_grav40[1,8,10,12]
chi2_changegrav=np.sum(((xvals_changegrav-xlist)/xerrs)**2.)
chi2red_changegrav=chi2_changegrav/numpoints
signif_changegrav=stats.chi2.sf(chi2_changegrav,numpoints)
sigma_changegrav=special.erfinv(1-signif_changegrav)*np.sqrt(2.)

xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorC2[2],colorT3[2],colorH32[2],colorW77[2],colorW4[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2],colorW18[2]])
ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorC2[1],colorT3[1],colorH32[1],colorW77[1],colorW4[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1],colorW18[1]])
xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorC2[4],colorT3[4],colorH32[4],colorW77[4],colorW4[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4],colorW18[4]])
yerrs=np.array([colorHD189[3],colorHD209[3],colorW43[3],colorC2[3],colorT3[3],colorH32[3],colorW77[3],colorW4[3],colorW79[3],colorW74[3],colorH41[3],colorK7[3],colorW76[3],colorW121[3],colorH7[3],colorW12[3],colorW18[3]])

numpoints=16.

xvals_bd=np.interp(ylist,color_bd[:,1],color_bd[:,2])
chi2_bd=np.sum(((xvals_bd-xlist)/xerrs)**2.)
chi2red_bd=chi2_bd/numpoints
signif_bd=stats.chi2.sf(chi2_bd,numpoints)
sigma_bd=special.erfinv(1-signif_bd)*np.sqrt(2.)

xvals_bdlogg3=np.interp(ylist,color_bdlogg3[:,1],color_bdlogg3[:,2])
chi2_bdlogg3=np.sum(((xvals_bdlogg3-xlist)/xerrs)**2.)
chi2red_bdlogg3=chi2_bdlogg3/numpoints
signif_bdlogg3=stats.chi2.sf(chi2_bdlogg3,numpoints)
sigma_bdlogg3=special.erfinv(1-signif_bdlogg3)*np.sqrt(2.)

xvals_bdlogg4=np.interp(ylist,color_bdlogg4[:,1],color_bdlogg4[:,2])
chi2_bdlogg4=np.sum(((xvals_bdlogg4-xlist)/xerrs)**2.)
chi2red_bdlogg4=chi2_bdlogg4/numpoints
signif_bdlogg4=stats.chi2.sf(chi2_bdlogg4,numpoints)
sigma_bdlogg4=special.erfinv(1-signif_bdlogg4)*np.sqrt(2.)

xvals_bdmetneg1=np.interp(ylist,color_bdmetneg1[:,1],color_bdmetneg1[:,2])
chi2_bdmetneg1=np.sum(((xvals_bdmetneg1-xlist)/xerrs)**2.)
chi2red_bdmetneg1=chi2_bdmetneg1/numpoints
signif_bdmetneg1=stats.chi2.sf(chi2_bdmetneg1,numpoints)
sigma_bdmetneg1=special.erfinv(1-signif_bdmetneg1)*np.sqrt(2.)

xvals_bdmetpos1=np.interp(ylist,color_bdmetpos1[:,1],color_bdmetpos1[:,2])
chi2_bdmetpos1=np.sum(((xvals_bdmetpos1-xlist)/xerrs)**2.)
chi2red_bdmetpos1=chi2_bdmetpos1/numpoints
signif_bdmetpos1=stats.chi2.sf(chi2_bdmetpos1,numpoints)
sigma_bdmetpos1=special.erfinv(1-signif_bdmetpos1)*np.sqrt(2.)






