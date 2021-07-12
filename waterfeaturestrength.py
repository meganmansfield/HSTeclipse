#UP TO DATE COPY 7/12/21
import pysynphot as S
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
import glob
from matplotlib import rc
import matplotlib.pylab as pl
import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
from scipy import stats
from scipy import special

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

def colormagmod(flist,Tlist,bd=False):
	fpmods=np.zeros((np.shape(masterwavegrid)[0],np.shape(flist)[0]))
	color=np.zeros((np.shape(flist)[0],3))
	for f in flist:
		thefile=np.loadtxt(f)
		s1=f.split('_')
		if bd==False:
			inds1=np.where(np.array(s1)=='Tirr')[0]
		else:
			inds1=np.where(np.array(s1)=='Teff')[0]
		placer=np.argmin(abs(float(s1[inds1[0]+1])-Tlist))
		filewaves=thefile[:,0]	#microns
		filefluxpl=thefile[:,1]*10**7.	#erg/s/m^2/m
		fpmods[:,placer]=np.interp(masterwavegrid,filewaves,filefluxpl)
	for i in np.arange(np.shape(flist)[0]):
		outpoint1=trapezoidint(masterwavegrid[outset1]*10**-6.,fpmods[:,i][outset1])
		outpoint2=trapezoidint(masterwavegrid[outset2]*10**-6.,fpmods[:,i][outset2])
		inpoint=trapezoidint(masterwavegrid[inset]*10**-6.,fpmods[:,i][inset])
		outflux=np.array([outpoint1,outpoint2])
		wave=np.array([(edgewave2+edgewave1)/2.,(edgewave6+edgewave5)/2.])
		diff2=np.array([(edgewave2-edgewave1),(edgewave6-edgewave5)])
		params0=np.array([1000.])
		mpfit=leastsq(bbod2point,params0,args=(wave,diff2,outflux))
		fakeinflux=make_bbod(np.array([(edgewave4+edgewave3)/2.]),np.array([(edgewave4-edgewave3)]),mpfit[0][0])
		color[i]= mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(inpoint))
	return fpmods,color

def colormagdata(datfile,planetparams):
	#downsample through interpolation to the master wavelength grid
	downsamplewave=np.interp(masterwavegrid,datfile[:,0],datfile[:,1]*10**-6.)
	outpoint1=np.mean(downsamplewave[outset1])
	outpoint2=np.mean(downsamplewave[outset2])
	inpoint=np.mean(downsamplewave[inset])
	meanerr=np.mean(datfile[:,2]*10.**-6.)
	dellam=np.mean(np.diff(datfile[:,0]))
	sp = S.Icat('k93models',planetparams[0],planetparams[2],planetparams[3])
	sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave=sp.wave*10.**-10  #in meters
	flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
	interp=np.interp(masterwavegrid,wave*10**6.,flux)
	starout1=trapezoidint(masterwavegrid[outset1]*10.**-6.,interp[outset1])	#unit erg/s/m^2
	starout2=trapezoidint(masterwavegrid[outset2]*10.**-6.,interp[outset2])
	starin=trapezoidint(masterwavegrid[inset]*10.**-6.,interp[inset])
	sp2 = S.Icat('k93models',planetparams[0]+planetparams[1],planetparams[2],planetparams[3])
	sp2.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave2=sp2.wave*10.**-10  #in meters
	flux2 = sp2.flux*10.**4*10.**10 #in erg/m2/m/s
	interp2=np.interp(masterwavegrid,wave2*10**6.,flux2)
	starout12=trapezoidint(masterwavegrid[outset1]*10.**-6.,interp2[outset1])	#unit erg/s/m^2
	starout22=trapezoidint(masterwavegrid[outset2]*10.**-6.,interp2[outset2])
	starin2=trapezoidint(masterwavegrid[inset]*10.**-6.,interp2[inset])
	Fpout1=outpoint1*starout1/planetparams[4]**2.	#unit erg/s/m^2
	Fpout2=outpoint2*starout2/planetparams[4]**2.
	Fpin=inpoint*starin/planetparams[4]**2.
	outflux=np.array([Fpout1,Fpout2])
	wave=np.array([(edgewave2+edgewave1)/2.,(edgewave6+edgewave5)/2.])
	diff2=np.array([(edgewave2-edgewave1),(edgewave6-edgewave5)])
	params0=np.array([1000.])
	mpfit=leastsq(bbod2point,params0,args=(wave,diff2,outflux))
	fakeinflux=make_bbod(np.array([(edgewave4+edgewave3)/2.]),np.array([(edgewave4-edgewave3)]),mpfit[0][0])
	outerr1=meanerr*starout1/planetparams[4]**2.*np.sqrt(dellam/(edgewave2-edgewave1))
	temp=Fpout1*(starout1-starout12)/starout1
	couterr1=np.sqrt(outerr1**2.+temp**2.)
	outerr2=meanerr*starout2/planetparams[4]**2.*np.sqrt(dellam/(edgewave6-edgewave5))
	temp=Fpout2*(starout2-starout22)/starout2
	couterr2=np.sqrt(outerr2**2.+temp**2.)
	inerr=meanerr*starin/planetparams[4]**2.*np.sqrt(dellam/(edgewave4-edgewave3))
	temp=Fpin*(starin-starin2)/starin
	cinerr=np.sqrt(inerr**2.+temp**2.)
	netoerr=np.sqrt(couterr1**2.+couterr2**2.)
	magerr=np.log10((netoerr+fakeinflux)/fakeinflux)
	colorerr=np.sqrt(netoerr**2./(fakeinflux**2.*np.log(10.)**2.)+cinerr**2./(Fpin**2.*np.log(10.)**2.))
	return mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(Fpin)),magerr,colorerr

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
	return mpfit[0][0],np.log10(fakeinflux),(np.log10(fakeinflux)-np.log10(inpoint)),magerr,colorerr

#define the small spacing grid in wavelength
masterwavegrid=np.linspace(1.0,2.0,2000) #microns

diff=np.diff(masterwavegrid)[0]
outset1=np.where((masterwavegrid>edgewave1)&(masterwavegrid<edgewave2))
outset2=np.where((masterwavegrid>edgewave5)&(masterwavegrid<edgewave6))
inset=np.where((masterwavegrid>edgewave3)&(masterwavegrid<edgewave4))

############################################## MIKE MODELS #######################################
#reading in all Sc-CHIMERA models - base units W/m^2
templist=np.array([500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1650,1700,1750,1800,1850,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3200,3400,3600])
flist_fiducial=glob.glob('./AllHotJupiterModels/Fiducial/*spec.txt')
flist_lowCO=glob.glob('./AllHotJupiterModels/CtoO/*0.01_spec.txt')
flist_highCO=glob.glob('./AllHotJupiterModels/CtoO/*0.85_spec.txt')
flist_delayTiO2000=glob.glob('./AllHotJupiterModels/TiO_VO_Cold_Trap/*DELAY_TiO_VO_2000*spec.txt')
flist_delayTiO2500=glob.glob('./AllHotJupiterModels/TiO_VO_Cold_Trap/*DELAY_TiO_VO_2500*spec.txt')
flist_delayTiO3000=glob.glob('./AllHotJupiterModels/TiO_VO_Cold_Trap/*DELAY_TiO_VO_3000*spec.txt')
flist_grav20=glob.glob('./AllHotJupiterModels/LogG/LOGG_2.0_*spec.txt')
flist_grav40=glob.glob('./AllHotJupiterModels/LogG/LOGG_4.0_*spec.txt')
flist_metneg15=glob.glob('./AllHotJupiterModels/Metallicity/*logZ_-1.5*spec.txt')
flist_metpos15=glob.glob('./AllHotJupiterModels/Metallicity/*logZ_+1.5*spec.txt')
flist_tintTF18=glob.glob('./AllHotJupiterModels/Tint/*TF18_TINT*spec.txt')
flist_fsedlow=glob.glob('./AllHotJupiterModels/Clouds/*fsed_0.1*spec.txt')
flist_fsedhigh=glob.glob('./AllHotJupiterModels/Clouds/*fsed_1.0*spec.txt')
flist_star3300=glob.glob('./AllHotJupiterModels/Stellar_Teff/*TSTAR_3300*spec.txt')
flist_star4300=glob.glob('./AllHotJupiterModels/Stellar_Teff/*TSTAR_4300*spec.txt')
flist_star6300=glob.glob('./AllHotJupiterModels/Stellar_Teff/*TSTAR_6300*spec.txt')
flist_star7200=glob.glob('./AllHotJupiterModels/Stellar_Teff/*TSTAR_7200*spec.txt')
flist_star8200=glob.glob('./AllHotJupiterModels/Stellar_Teff/*TSTAR_8200*spec.txt')

#Self-luminous object Sc-CHIMERA models
flist_bd=glob.glob('./AllSelfLuminousModels/Fiducial/*spec.txt')
flist_bdmetpos1=glob.glob('./AllSelfLuminousModels/LogMet+1/*spec.txt')
flist_bdmetneg1=glob.glob('./AllSelfLuminousModels/LogMet-1/*spec.txt')
flist_bdlogg3=glob.glob('./AllSelfLuminousModels/LogG_3/*spec.txt')
flist_bdlogg4=glob.glob('./AllSelfLuminousModels/LogG_4/*spec.txt')
bdtemplist=np.array([1000.,1200.,1400.,1600.,1800.,2000.,2200.,2400.,2600.,2800.])

fpmods_fiducial,color_fiducial=colormagmod(flist_fiducial,templist)
fpmods_lowCO,color_lowCO=colormagmod(flist_lowCO,templist)
fpmods_highCO,color_highCO=colormagmod(flist_highCO,templist)
fpmods_delayTiO2000,color_delayTiO2000=colormagmod(flist_delayTiO2000,templist)
fpmods_delayTiO2500,color_delayTiO2500=colormagmod(flist_delayTiO2500,templist)
fpmods_delayTiO3000,color_delayTiO3000=colormagmod(flist_delayTiO3000,templist)
fpmods_grav20,color_grav20=colormagmod(flist_grav20,templist)
fpmods_grav40,color_grav40=colormagmod(flist_grav40,templist)
fpmods_metneg15,color_metneg15=colormagmod(flist_metneg15,templist)
fpmods_metpos15,color_metpos15=colormagmod(flist_metpos15,templist)
fpmods_star3300,color_star3300=colormagmod(flist_star3300,templist)
fpmods_star4300,color_star4300=colormagmod(flist_star4300,templist)
fpmods_star6300,color_star6300=colormagmod(flist_star6300,templist)
fpmods_star7200,color_star7200=colormagmod(flist_star7200,templist)
fpmods_star8200,color_star8200=colormagmod(flist_star8200,templist)
fpmods_tintTF18,color_tintTF18=colormagmod(flist_tintTF18,templist)
fpmods_fsedlow,color_fsedlow=colormagmod(flist_fsedlow,templist)
fpmods_fsedhigh,color_fsedhigh=colormagmod(flist_fsedhigh,templist)

fpmods_bd,color_bd=colormagmod(flist_bd,bdtemplist,bd=True)
fpmods_bdmetpos1,color_bdmetpos1=colormagmod(flist_bdmetpos1,bdtemplist,bd=True)
fpmods_bdmetneg1,color_bdmetneg1=colormagmod(flist_bdmetneg1,bdtemplist,bd=True)
fpmods_bdlogg3,color_bdlogg3=colormagmod(flist_bdlogg3,bdtemplist,bd=True)
fpmods_bdlogg4,color_bdlogg4=colormagmod(flist_bdlogg4,bdtemplist,bd=True)

##################### FIGURE 2: Plotting fiducial model grid ###################################
tplist_fiducial=glob.glob('./AllHotJupiterModels/Fiducial/*TP_GAS.txt')
mikemods=np.zeros((1406,np.shape(flist_fiducial)[0]))
miketp=np.zeros((69,np.shape(flist_fiducial)[0]))
for f in flist_fiducial:
	thefile=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	mikewaves=thefile[:,0]	#microns
	mikemods[:,placer]=thefile[:,1]*10**7.	#erg/s/m^2/m
for f in tplist_fiducial:
	thefile=np.loadtxt(f)
	s1=f.split('_')
	inds1=np.where(np.array(s1)=='Tirr')[0]
	placer=np.argmin(abs(float(s1[inds1[0]+1])-templist))
	miketp[:,placer]=thefile[:,1]
	mikepressures=thefile[:,0]

n=np.shape(templist)[0]
colors=pl.cm.plasma(np.linspace(0,0.9,n))

rc('axes',linewidth=2)
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,21))
for i in np.arange(np.shape(templist)[0]/2.)*2.:
	ax2.plot(mikewaves,mikemods[:,int(i)],color=colors[int(i)],linewidth=3,zorder=1)
ax2.set_xlim((1.0,1.8))
ax2.axvspan(1.22,1.33,color='k',alpha=0.2,zorder=0)
ax2.axvspan(1.53,1.61,color='k',alpha=0.2,zorder=0)
ax2.axvspan(1.35,1.48,color='k',alpha=0.2,zorder=0)
ax2.set_yscale('log')
ax2.set_ylim((1.*10.**13.,7.*10**19.))
ax2.set_xlabel('Wavelength [$\mu$m]',fontsize=20)
ax2.set_ylabel('Planet Flux [erg/s/m$^{3}$]',fontsize=20)
ax2.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax2.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

for i in np.arange(np.shape(templist)[0]/2.)*2.:
	ax1.plot(miketp[:,int(i)],mikepressures,color=colors[int(i)],linewidth=3,zorder=1)
ax1.set_xlim((500,4000))
ax1.set_yscale('log')
ax1.set_ylim((2*10.**2,5*10.**-4.))
ax1.set_xlabel('Temperature [K]',fontsize=20)
ax1.set_ylabel('Pressure [bar]',fontsize=20)
ax1.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

opacityfile=np.loadtxt('opacity.txt')

ax3.plot(opacityfile[:,1]/opacityfile[:,2],opacityfile[:,0],color='k',linewidth=3,zorder=1)
ax3.axvline(x=1,color='k',linestyle=':',linewidth=3,zorder=0)
ax3.set_xscale('log')
ax3.set_xlabel('$K_{J}$/$K_{B}$',fontsize=20)
ax3.set_ylabel('T$_{eq}$ [K]',fontsize=20)
ax3.invert_xaxis()
ax3.set_xticks([0.01,0.1,1])
ax3.set_xticklabels(['0.01','0.1','1'])
ax3.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax3.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

plt.tight_layout()
plt.savefig('models.png',dpi=300)
plt.show()

################################################## DATA SETS ##############################
#read in the data - data go wavelength, flux, flux err
C2=np.loadtxt('./Spectra/CoRoT2b.txt')
H7=np.loadtxt('./Spectra/HAT7b.txt')
H32=np.loadtxt('./Spectra/HAT32Ab.txt')
H41=np.loadtxt('./Spectra/HAT41b.txt')
HD189=np.loadtxt('./Spectra/HD189733b.txt')
HD209=np.loadtxt('./Spectra/HD209458b.txt')
K7=np.loadtxt('./Spectra/KELT7b.txt')
Kep13=np.loadtxt('./Spectra/Kepler13Ab.txt')
T3=np.loadtxt('./Spectra/TrES3b.txt')
W4=np.loadtxt('./Spectra/WASP4b.txt')
W12=np.loadtxt('./Spectra/WASP12b.txt')
W18=np.loadtxt('./Spectra/WASP18b.txt')
W33=np.loadtxt('./Spectra/WASP33b.txt')
W43=np.loadtxt('./Spectra/WASP43b.txt')
W74=np.loadtxt('./Spectra/WASP74b.txt')
W76=np.loadtxt('./Spectra/WASP76b.txt')
W79=np.loadtxt('./Spectra/WASP79b.txt')
W103=np.loadtxt('./Spectra/WASP103b.txt')
W121=np.loadtxt('./Spectra/WASP121b.txt')

allplanetparams=np.loadtxt('planetparams.txt')

colorC2=colormagdata(C2,allplanetparams[0])
colorH7=colormagdata(H7,allplanetparams[1])
colorH32=colormagdata(H32,allplanetparams[2])
colorH41=colormagdata(H41,allplanetparams[3])
colorHD189=colormagdata(HD189,allplanetparams[4])
colorHD209=colormagdata(HD209,allplanetparams[5])
colorK7=colormagdata(K7,allplanetparams[6])
colorKep13=colormagdata(Kep13,allplanetparams[7])
colorT3=colormagdata(T3,allplanetparams[8])
colorW4=colormagdata(W4,allplanetparams[9])
colorW12=colormagdata(W12,allplanetparams[10])
colorW18=colormagdata(W18,allplanetparams[11])
colorW33=colormagdata(W33,allplanetparams[12])
colorW43=colormagdata(W43,allplanetparams[13])
colorW74=colormagdata(W74,allplanetparams[14])
colorW76=colormagdata(W76,allplanetparams[15])
colorW79=colormagdata(W79,allplanetparams[16])
colorW103=colormagdata(W103,allplanetparams[17])
colorW121=colormagdata(W121,allplanetparams[18])

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
	bdwavesample=np.interp(masterwavegrid,bddata[:,0],bdfgoodunit)
	bderrsample=np.interp(masterwavegrid,bddata[:,0],bderrgoodunit)
	outpoint1bd=trapezoidint(masterwavegrid[outset1]*10.**-6.,bdwavesample[outset1])
	outpoint2bd=trapezoidint(masterwavegrid[outset2]*10.**-6.,bdwavesample[outset2])
	inpointbd=trapezoidint(masterwavegrid[inset]*10.**-6.,bdwavesample[inset])
	outerr1bd=trapezoidint(masterwavegrid[outset1]*10.**-6.,bderrsample[outset1])
	outerr2bd=trapezoidint(masterwavegrid[outset2]*10.**-6.,bderrsample[outset2])
	inerrbd=trapezoidint(masterwavegrid[inset]*10.**-6.,bderrsample[inset])
	if not inpointbd<0:
		goodlist.append(counter)
	bdcolors[counter,:]=colormagbd(outpoint1bd,outpoint2bd,inpointbd,outerr1bd,outerr2bd,inerrbd)
	counter+=1

delTbd=np.array([20,6.5,4,0.5,0.5,0.5,5.5,0.5,0.5,0.5,0.5,0.5,0.5,5,5,0.5,0.5,0.5,0.5,8,0.5,\
	0.5,0.5,0.5,0.5,0.5,0.5,6.5,0.5,6.5,0.5,7,1.5,1,4,4,5,0.5,4,4.5,22.5,3.5,0.5,0.5,6.5,\
	0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,4])

###################### FIGURE 3 ####################################
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
equtemps=np.array([2241.,1898.,1935.,1216.,1451.,2047.,2550.,2383.,2780.,1448.,1921.,2169.,1867.,2504.,2358.,1522.,1590.,1666.,2550.,2498.])	#H7,H32,H41,HD189,HD209,K7,Kep13,W18,W33,W43,W74,W76,W79,W103,W121,C2,T3,W4,Kep13,W12
vmin=np.min(equtemps)
vmax=np.max(equtemps)+200
normequtemps=(equtemps-vmin)/np.max(equtemps-vmin)
inferno = cm = plt.get_cmap('inferno') 
cNorm  = mplcolors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=inferno)

mycolor='xkcd:light gray'
edgecolor='k'
linecolor='xkcd:slate gray'
linecolor2='xkcd:light brown'
linecolor3='xkcd:olive green'
pointcolor2='xkcd:brown'
pointcolor3='xkcd:dark green'

fig,ax1=plt.subplots(figsize=(10,7))
ax1.set_ylim((1150,3550))
ax1.set_xlim((-0.2,0.65))
ax1.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax1.set_yticklabels(['1500','2000','2500','3000','3500'])
ax1.set_yticks(ticks=[1200,1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax1.set_yticklabels([],minor=True)
ax1.set_ylabel('Dayside Temperature [K]',fontsize=20)
ax1.set_xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=20) #
ax1.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')

ax1.axvline(x=0.0,color='k',zorder=1,linewidth=2)#'xkcd:slate gray'
ax1.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=3,linewidth=2,markersize=10)#'xkcd:light brown','xkcd:brown'
ax1.text(-0.18,3400,'Fiducial Model',color=linecolor,fontsize=15,zorder=3,fontweight='bold')#Fiducial 'xkcd:light brown'
ax1.plot(color_bd[:,2],color_bd[:,0],color=linecolor2,marker='.',markeredgecolor=pointcolor2,zorder=2,linewidth=2,markersize=10)
ax1.text(0.2,2800,'Self-Irradiated Bodies',color=linecolor2,fontsize=15,zorder=3,fontweight='bold')
ax1.fill_betweenx(color_bdlogg3[:,0],fillbd,color_bdlogg3[:,2],color='xkcd:beige',zorder=0)

ax1.fill_betweenx(color_lowCO[:,0],fillhighCO,color_lowCO[:,2],color=mycolor,zorder=1)#'xkcd:beige'
ax1.fill_betweenx(color_highCO[:,0],color_highCO[:,2],filllowCO,color=mycolor,zorder=1)#'xkcd:beige'
ax1.fill_betweenx(color_metneg15[:,0],color_metneg15[:,2],fillhighmet,color=mycolor,zorder=1)#'xkcd:beige'
ax1.fill_betweenx(color_highCO[:,0],color_highCO[:,2],filllowmet,color=mycolor,zorder=1)
ax1.fill_betweenx(color_tintTF18[:,0],color_tintTF18[:,2],fillheat,color=mycolor,zorder=1)
ax1.fill_betweenx(color_grav40[:,0],color_grav40[:,2],fillg,color=mycolor,zorder=1)
ax1.fill_betweenx(color_lowCO[:,0],color_lowCO[:,2],filltop,color=mycolor,zorder=1)

ax1.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39.,mec=scalarMap.to_rgba(equtemps[0]),mfc=scalarMap.to_rgba(equtemps[0]),ecolor=scalarMap.to_rgba(equtemps[0]),marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
ax1.text(colorH7[2]+0.01,colorH7[0]+20,'HAT-P-7b',color=scalarMap.to_rgba(equtemps[0]),fontsize=15,zorder=3)#'xkcd:kelly green'
ax1.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59.,mec=scalarMap.to_rgba(equtemps[1]),mfc=scalarMap.to_rgba(equtemps[1]),ecolor=scalarMap.to_rgba(equtemps[1]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorH32[2]-0.11,colorH32[0]+60,'HAT-P-32b',color=scalarMap.to_rgba(equtemps[1]),fontsize=15,zorder=3)#'xkcd:pink'
ax1.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66.,mec=scalarMap.to_rgba(equtemps[2]),mfc=scalarMap.to_rgba(equtemps[2]),ecolor=scalarMap.to_rgba(equtemps[2]),marker='.',markersize=15,linestyle='none',linewidth=3,label='Observed Planets',zorder=4)
ax1.text(colorH41[2]+0.02,colorH41[0]-100,'HAT-P-41b',color=scalarMap.to_rgba(equtemps[2]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#'xkcd:light orange'
ax1.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57.,mec=scalarMap.to_rgba(equtemps[3]),mfc=scalarMap.to_rgba(equtemps[3]),ecolor=scalarMap.to_rgba(equtemps[3]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorHD189[2]+0.02,colorHD189[0]-80,'HD 189733b',color=scalarMap.to_rgba(equtemps[3]),fontsize=15,zorder=3)#'xkcd:greenish grey'
ax1.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28.,mec=scalarMap.to_rgba(equtemps[4]),mfc=scalarMap.to_rgba(equtemps[4]),ecolor=scalarMap.to_rgba(equtemps[4]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorHD209[2]+0.01,colorHD209[0]-75,'HD 209458b',color=scalarMap.to_rgba(equtemps[4]),fontsize=15,zorder=3)#'xkcd:bright pink'
ax1.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54.,mec=scalarMap.to_rgba(equtemps[5]),mfc=scalarMap.to_rgba(equtemps[5]),ecolor=scalarMap.to_rgba(equtemps[5]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorK7[2]+0.005,colorK7[0]+130,'KELT-7b',color=scalarMap.to_rgba(equtemps[5]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:orange,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20.,mec=scalarMap.to_rgba(equtemps[7]),mfc=scalarMap.to_rgba(equtemps[7]),ecolor=scalarMap.to_rgba(equtemps[7]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW18[2]-0.13,colorW18[0]-25,'WASP-18b',color=scalarMap.to_rgba(equtemps[7]),fontsize=15,zorder=3)#xkcd:blue
ax1.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26.,mec=scalarMap.to_rgba(equtemps[8]),mfc=scalarMap.to_rgba(equtemps[8]),ecolor=scalarMap.to_rgba(equtemps[8]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW33[2]+0.02,colorW33[0],'WASP-33b',color=scalarMap.to_rgba(equtemps[8]),fontsize=15,zorder=3)#xkcd:sky blue bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23.,mec=scalarMap.to_rgba(equtemps[9]),mfc=scalarMap.to_rgba(equtemps[9]),ecolor=scalarMap.to_rgba(equtemps[9]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW43[2]-0.13,colorW43[0]-75,'WASP-43b',color=scalarMap.to_rgba(equtemps[9]),fontsize=15,zorder=3)#xkcd:red
ax1.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48.,mec=scalarMap.to_rgba(equtemps[10]),mfc=scalarMap.to_rgba(equtemps[10]),ecolor=scalarMap.to_rgba(equtemps[10]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW74[2]+0.01,colorW74[0]-100,'WASP-74b',color=scalarMap.to_rgba(equtemps[10]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:magenta,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27.,mec=scalarMap.to_rgba(equtemps[11]),mfc=scalarMap.to_rgba(equtemps[11]),ecolor=scalarMap.to_rgba(equtemps[11]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW76[2]-0.14,colorW76[0]-50,'WASP-76b',color=scalarMap.to_rgba(equtemps[11]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58.,mec=scalarMap.to_rgba(equtemps[12]),mfc=scalarMap.to_rgba(equtemps[12]),ecolor=scalarMap.to_rgba(equtemps[12]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW79[2]+0.02,colorW79[0]+65,'WASP-79b',color=scalarMap.to_rgba(equtemps[12]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:aquamarine,bbox=dict(facecolor='none',edgecolor='k')
ax1.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50.,mec=scalarMap.to_rgba(equtemps[13]),mfc=scalarMap.to_rgba(equtemps[13]),ecolor=scalarMap.to_rgba(equtemps[13]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW103[2]-0.14,colorW103[0]+20,'WASP-103b',color=scalarMap.to_rgba(equtemps[13]),fontsize=15,zorder=3)#xkcd:violet
ax1.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39.,mec=scalarMap.to_rgba(equtemps[14]),mfc=scalarMap.to_rgba(equtemps[14]),ecolor=scalarMap.to_rgba(equtemps[14]),marker='.',markersize=15,linestyle='none',linewidth=3,zorder=4)
ax1.text(colorW121[2]-0.15,colorW121[0],'WASP-121b',color=scalarMap.to_rgba(equtemps[14]),fontsize=15,zorder=3,bbox=dict(facecolor='none',edgecolor='k'))#xkcd:light green

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
plt.savefig('Fig3.png',dpi=300)
plt.show()
############### FIGURE 3 END ##########################################

################# SUPPLEMENTARY FIGURE 4 #########################

colors=pl.cm.inferno(np.linspace(0,0.9,5))

tempx=[x for _,x in sorted(zip(color_star3300[:,0],color_star3300[:,2]))]
tempy=np.sort(color_star3300[:,0])

plt.figure(figsize=(10,7))
plt.axvline(x=0.0,color='k',zorder=0)
plt.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
plt.plot(color_fsedlow[:,2],color_fsedlow[:,0],color=colors[3],marker='.',label=r'$f_{sed}=0.1$',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
plt.plot(color_fsedhigh[:,2],color_fsedhigh[:,0],color=colors[1],marker='.',label=r'$f_{sed}=1.0$',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
plt.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
plt.yticks(ticks=[1500,2000,2500,3000,3500],labels=['1500','2000','2500','3000','3500'])
ax=plt.gca()
ax.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax.set_yticklabels([],minor=True)
plt.ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
plt.xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
plt.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
plt.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
plt.legend(fontsize=20)
plt.ylim((1150,3550))
plt.xlim((-0.2,0.5))
plt.title('Vary Clouds',fontsize=20)
plt.tight_layout()
plt.savefig('clouds.png',dpi=300)
plt.show()


########################## FIGURE 4 ###########################
fig,((ax1,ax2),(ax3,ax4),(ax5,ax6))=plt.subplots(3,2,figsize=(21,20))

ax1.axvline(x=0.0,color='k',zorder=0)
ax1.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
ax1.plot(color_lowCO[:,2],color_lowCO[:,0],color=colors[1],marker='.',label='C/O=0.01',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax1.plot(color_highCO[:,2],color_highCO[:,0],color=colors[3],marker='.',label='C/O=0.85',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax1.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax1.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax1.set_yticklabels(labels=['1500','2000','2500','3000','3500'])
ax1.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800,2900,3100,3200,3300,3400],minor=True)
ax1.set_yticklabels(labels=['','','','','','','','','','','','','','','','','',''],minor=True)
ax1.set_ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
ax1.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax1.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
ax1.legend(fontsize=20)
ax1.set_ylim((1150,3550))
ax1.set_xlim((-0.2,0.5))
ax1.set_title('Vary Planet C/O',fontsize=20)

ax2.axvline(x=0.0,color='k',zorder=0)
ax2.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
ax2.plot(color_metneg15[:,2],color_metneg15[:,0],color=colors[1],marker='.',label='[M/H]=-1.5',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax2.plot(color_metpos15[:,2],color_metpos15[:,0],color=colors[3],marker='.',label='[M/H]=1.5',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax2.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax2.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax2.set_yticklabels(labels=['1500','2000','2500','3000','3500'])
ax2.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800,2900,3100,3200,3300,3400],minor=True)
ax2.set_yticklabels(labels=['','','','','','','','','','','','','','','','','',''],minor=True)
ax2.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax2.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
ax2.legend(fontsize=20)
ax2.set_ylim((1150,3550))
ax2.set_xlim((-0.2,0.5))
ax2.set_title('Vary Planet Metallicity',fontsize=20)

ax3.axvline(x=0.0,color='k',zorder=0)
ax3.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
ax3.plot(tempx,tempy,color=colors[0],marker='.',label='T$_{eff}$=3300',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax3.plot(color_star4300[:,2],color_star4300[:,0],color=colors[1],marker='.',label='T$_{eff}$=4300',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax3.plot(color_star6300[:,2],color_star6300[:,0],color=colors[2],marker='.',label='T$_{eff}$=6300',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax3.plot(color_star7200[:,2],color_star7200[:,0],color=colors[3],marker='.',label='T$_{eff}$=7200',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax3.plot(color_star8200[:,2],color_star8200[:,0],color=colors[4],marker='.',label='T$_{eff}$=8200',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax3.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=11,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax3.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax3.set_yticklabels(labels=['1500','2000','2500','3000','3500'])
ax3.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax3.set_yticklabels(labels=['','','','','','','','','','','','','','','','','',''],minor=True)
ax3.set_ylabel('Dayside Temperature [K]',fontsize=25)#Blackbody Temperature [K]
ax3.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax3.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
ax3.legend(fontsize=20)
ax3.set_ylim((1250,3550))
ax3.set_xlim((-0.2,0.5))
ax3.set_title('Vary Stellar Temperature',fontsize=20)

ax4.axvline(x=0.0,color='k',zorder=0)
ax4.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
ax4.plot(color_grav20[:,2],color_grav20[:,0],color=colors[1],marker='.',label='log(g)=2.0',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax4.plot(color_grav40[:,2],color_grav40[:,0],color=colors[3],marker='.',label='log(g)=4.0',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax4.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax4.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax4.set_yticklabels(labels=['1500','2000','2500','3000','3500'])
ax4.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax4.set_yticklabels([],minor=True)
ax4.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax4.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
ax4.legend(fontsize=20)
ax4.set_ylim((1150,3550))
ax4.set_xlim((-0.2,0.5))
ax4.set_title('Vary Planet Gravity',fontsize=20)

ax5.axvline(x=0.0,color='k',zorder=0)
ax5.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax5.plot(color_tintTF18[:,2],color_tintTF18[:,0],color=colors[3],marker='.',label='Thorngren et al. 2019',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax5.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax5.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax5.set_yticklabels(labels=['1500','2000','2500','3000','3500'])
ax5.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax5.set_yticklabels([],minor=True)
ax5.set_ylabel('Dayside Temperature [K]',fontsize=25)
ax5.set_xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
ax5.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax5.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
ax5.legend(fontsize=20)
ax5.set_ylim((1150,3550))
ax5.set_xlim((-0.2,0.5))
ax5.set_title('Vary Internal Heating',fontsize=20)

ax6.axvline(x=0.0,color='k',zorder=0)
ax6.plot(color_fiducial[:,2],color_fiducial[:,0],color=linecolor,marker='.',label='Fiducial',markeredgecolor=edgecolor,zorder=4,linewidth=3,markersize=15)
ax6.plot(color_delayTiO3000[:,2],color_delayTiO3000[:,0],color=colors[4],marker='.',label='TiO/VO Delayed to 3000 K',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax6.plot(color_delayTiO2500[:,2],color_delayTiO2500[:,0],color=colors[3],marker='.',label='TiO/VO Delayed to 2500 K',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax6.plot(color_delayTiO2000[:,2],color_delayTiO2000[:,0],color=colors[2],marker='.',label='TiO/VO Delayed to 2000 K',markeredgecolor='k',zorder=4,linewidth=3,markersize=15)
ax6.errorbar(colorH7[2],colorH7[0],xerr=colorH7[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorH32[2],colorH32[0],xerr=colorH32[4],yerr=59,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorH41[2],colorH41[0],xerr=colorH41[4],yerr=66,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorHD189[2],colorHD189[0],xerr=colorHD189[4],yerr=57,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorHD209[2],colorHD209[0],xerr=colorHD209[4],yerr=28,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorK7[2],colorK7[0],xerr=colorK7[4],yerr=54,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW18[2],colorW18[0],xerr=colorW18[4],yerr=20,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW33[2],colorW33[0],xerr=colorW33[4],yerr=26,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW43[2],colorW43[0],xerr=colorW43[4],yerr=23,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW74[2],colorW74[0],xerr=colorW74[4],yerr=48,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW76[2],colorW76[0],xerr=colorW76[4],yerr=27,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW79[2],colorW79[0],xerr=colorW79[4],yerr=58,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW103[2],colorW103[0],xerr=colorW103[4],yerr=50,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW121[2],colorW121[0],xerr=colorW121[4],yerr=39,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorC2[2],colorC2[0],xerr=colorC2[4],yerr=42.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorT3[2],colorT3[0],xerr=colorT3[4],yerr=97.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW4[2],colorW4[0],xerr=colorW4[4],yerr=62.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorKep13[2],colorKep13[0],xerr=colorKep13[4],yerr=107.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.errorbar(colorW12[2],colorW12[0],xerr=colorW12[4],yerr=70.,color='k',marker='.',markersize=15,linestyle='none',linewidth=3,zorder=3)
ax6.set_yticks(ticks=[1500,2000,2500,3000,3500])
ax6.set_yticklabels(labels=['1500','2000','2500','3000','3500'])
ax6.set_yticks(ticks=[1300,1400,1600,1700,1800,1900,2100,2200,2300,2400,2600,2700,2800.2900,3100,3200,3300,3400],minor=True)
ax6.set_yticklabels([],minor=True)
ax6.set_xlabel('Water Feature Strength ($S_{H_{2}O}$)',fontsize=25)
ax6.tick_params(which='major',labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
ax6.tick_params(which='minor',axis="both",right=True,width=1,length=4,direction='in')
ax6.legend(fontsize=20)
ax6.set_ylim((1150,3550))
ax6.set_xlim((-0.2,0.5))
ax6.set_title('Delayed TiO/VO',fontsize=20)

plt.tight_layout()
plt.savefig('individualmodels.png',dpi=300)
plt.show()

########################## SUPPLEMENTARY FIGURE 1 ###########################

datwave=W43[:,0]
datflux=W43[:,1]*10**-6.
daterr=W43[:,2]*10**-6.
diff=np.diff(W43[:,0])[0]

tempwavegrid=masterwavegrid*10**-6.

def indplanetbbod(pltemp,plwave,startemp,starmet,starlogg,rprs):
	plfinebbod=fakedata_bbod(tempwavegrid,pltemp) #erg/s/m^2/m
	sp = S.Icat('k93models',startemp,starmet,starlogg)
	sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave=sp.wave*10.**-10  #in meters
	flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
	starfinebbod=np.interp(tempwavegrid,wave,flux)
	diff=np.mean(np.diff(plwave))*10**-6.

	newminwave=np.min(plwave)*10**-6.
	newmaxwave=np.max(plwave)*10**-6.
	smallgrid=tempwavegrid[(tempwavegrid>newminwave-diff/2.)&(tempwavegrid<newmaxwave+diff/2.)]
	smallplflux=plfinebbod[(tempwavegrid>newminwave-diff/2.)&(tempwavegrid<newmaxwave+diff/2.)]
	smallstarflux=starfinebbod[(tempwavegrid>newminwave-diff/2.)&(tempwavegrid<newmaxwave+diff/2.)]
	tgrid=tempwavegrid[(tempwavegrid>newminwave)&(tempwavegrid<newmaxwave)]
	modelbbod=np.zeros(np.shape(tgrid)[0])
	for i in np.arange(np.shape(tgrid)[0]):
		wave1=tgrid[i]-diff/2.
		wave2=tgrid[i]+diff/2.
		plint=trapezoidint(smallgrid[(smallgrid>wave1)&(smallgrid<wave2)],smallplflux[(smallgrid>wave1)&(smallgrid<wave2)])
		starint=trapezoidint(smallgrid[(smallgrid>wave1)&(smallgrid<wave2)],smallstarflux[(smallgrid>wave1)&(smallgrid<wave2)])
		modelbbod[i]=(plint/starint*rprs**2.)*10**6. #in ppm
	return tgrid*10**6.,modelbbod

tgridW43,BmodelW43=indplanetbbod(1775.,datwave,4520.,-0.01,4.645,0.1558)
plotBbod=np.interp(datwave,tgridW43,BmodelW43)

W43mod=np.loadtxt('./FiducialModels/modelfidW43.txt')
interpmodel=np.interp(datwave,W43mod[:,0],W43mod[:,1]*10**6.)
plt.figure(figsize=(8.5,6))
plt.errorbar(datwave,datflux*10**6.,yerr=daterr*10**6.,color='b',marker='.',markersize=15,linestyle='none',linewidth=3,label='WASP-43b, Kreidberg et al. (2014)',zorder=2)
plt.plot(datwave,interpmodel,marker='.',markersize=15,linewidth=3,markeredgecolor='k',color='xkcd:slate gray',label='Fiducial Model',zorder=1)
plt.plot(datwave,plotBbod,marker='d',markersize=9,linewidth=3,markeredgecolor='k',color='xkcd:slate gray',label='Blackbody',zorder=1)#slate gray
plt.fill_between([1.22,1.33],[200,200],[800,800],color='xkcd:orange',alpha=0.3,zorder=0,edgecolor='none')
plt.fill_between([1.53,1.61],[200,200],[800,800],color='xkcd:orange',alpha=0.3,zorder=0,edgecolor='none')
plt.fill_between([1.35,1.48],[200,200],[800,800],color='g',alpha=0.3,zorder=0,edgecolor='none')
plt.text(1.415,250,'In-Band',fontsize=20,color='g',horizontalalignment='center')
plt.text(1.275,250,'Out-of-\nBand',fontsize=20,color='xkcd:orange',horizontalalignment='center')
plt.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
plt.xlabel('Wavelength [$\mu$m]',fontsize=20)
plt.ylabel('F$_{p}$/F$_{s}$ [ppm]',fontsize=20)
plt.legend(fontsize=15,loc='upper center')
plt.xlim((1.1,1.7))
plt.ylim((225,800))
plt.tight_layout()
plt.savefig('explanation.png',dpi=300)
plt.show()

#################### CALCULATE CHI-SQUARED FOR DATA FITTING EACH MODEL #########################

xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorC2[2],colorT3[2],colorH32[2],colorW4[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2],colorW18[2],colorW103[2],colorW33[2],colorKep13[2]])
ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorC2[1],colorT3[1],colorH32[1],colorW4[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1],colorW18[1],colorW103[1],colorW33[1],colorKep13[1]])
xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorC2[4],colorT3[4],colorH32[4],colorW4[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4],colorW18[4],colorW103[4],colorW33[4],colorKep13[4]])

numpoints=19.

def chisqcalc(xlist,ylist,xerrs,modelcolors):
	xvals=np.interp(ylist,modelcolors[:,1],modelcolors[:,2])
	chi2=np.sum(((xvals-xlist)/xerrs)**2.)
	chi2red=chi2/numpoints
	signif=stats.chi2.sf(chi2,numpoints)
	sigma=special.erfinv(1-signif)*np.sqrt(2.)
	return sigma

sigma_fiducial=chisqcalc(xlist,ylist,xerrs,color_fiducial)
sigma_lowCO=chisqcalc(xlist,ylist,xerrs,color_lowCO)
sigma_highCO=chisqcalc(xlist,ylist,xerrs,color_highCO)
sigma_delayTiO2000=chisqcalc(xlist,ylist,xerrs,color_delayTiO2000)
sigma_delayTiO2500=chisqcalc(xlist,ylist,xerrs,color_delayTiO2500)
sigma_delayTiO3000=chisqcalc(xlist,ylist,xerrs,color_delayTiO3000)
sigma_grav20=chisqcalc(xlist,ylist,xerrs,color_grav20)
sigma_grav40=chisqcalc(xlist,ylist,xerrs,color_grav40)
sigma_metneg15=chisqcalc(xlist,ylist,xerrs,color_metneg15)
sigma_metpos15=chisqcalc(xlist,ylist,xerrs,color_metpos15)
sigma_tintTF18=chisqcalc(xlist,ylist,xerrs,color_tintTF18)

#Giving each planet an appropriate gravity compared to the fiducial model:
# xvals_changegrav=np.array([xvals_fiducial[0],xvals_fiducial[1],xvals_grav40[2],xvals_fiducial[3],\
# 	xvals_fiducial[4],xvals_grav20[5],xvals_fiducial[6],xvals_fiducial[7],xvals_fiducial[8],\
# 	xvals_fiducial[9],xvals_fiducial[10],xvals_grav40[11],xvals_fiducial[12],xvals_fiducial[13],\
# 	xvals_fiducial[14],xvals_grav40[15],xvals_fiducial[16],xvals_fiducial[17],xvals_grav40[18]])

# chi2_changegrav=np.sum(((xvals_changegrav-xlist)/xerrs)**2.)
# chi2red_changegrav=chi2_changegrav/numpoints
# signif_changegrav=stats.chi2.sf(chi2_changegrav,numpoints)
# sigma_changegrav=special.erfinv(1-signif_changegrav)*np.sqrt(2.)

#For comparison to brown dwarf models, which don't go all the way to higher temperatures. This only includes planets with Tday<~3000 K
xlist=np.array([colorHD189[2],colorHD209[2],colorW43[2],colorC2[2],colorT3[2],colorH32[2],colorW4[2],colorW79[2],colorW74[2],colorH41[2],colorK7[2],colorW76[2],colorW121[2],colorH7[2],colorW12[2],colorW18[2]])
ylist=np.array([colorHD189[1],colorHD209[1],colorW43[1],colorC2[1],colorT3[1],colorH32[1],colorW4[1],colorW79[1],colorW74[1],colorH41[1],colorK7[1],colorW76[1],colorW121[1],colorH7[1],colorW12[1],colorW18[1]])
xerrs=np.array([colorHD189[4],colorHD209[4],colorW43[4],colorC2[4],colorT3[4],colorH32[4],colorW4[4],colorW79[4],colorW74[4],colorH41[4],colorK7[4],colorW76[4],colorW121[4],colorH7[4],colorW12[4],colorW18[4]])

numpoints=16.

sigma_bd=chisqcalc(xlist,ylist,xerrs,color_bd)
sigma_bdlogg3=chisqcalc(xlist,ylist,xerrs,color_bdlogg3)
sigma_bdlogg4=chisqcalc(xlist,ylist,xerrs,color_bdlogg4)
sigma_bdmetneg1=chisqcalc(xlist,ylist,xerrs,color_bdmetneg1)
sigma_bdmetpos1=chisqcalc(xlist,ylist,xerrs,color_bdmetpos1)


