#plot spectra of newly analyzed planets
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc
import smooth
import pysynphot as S

h=6.626*10**-34 #planck constant
c=2.998*10**8 #speed of light
kb=1.381*10**-23 #boltzmann constant

H41=np.loadtxt('./Spectra/HAT41b.txt')
K7=np.loadtxt('./Spectra/KELT7b.txt')
W74=np.loadtxt('./Spectra/WASP74b.txt')
W76=np.loadtxt('./Spectra/WASP76b.txt')
W79=np.loadtxt('./Spectra/WASP79b.txt')
W121=np.loadtxt('./Spectra/WASP121b.txt')
Kep13=np.loadtxt('./Spectra/Kepler13Ab.txt')

W121v2=np.loadtxt('./Spectra/WASP121b_MikalEvans.txt')
W76v2=np.loadtxt('./Spectra/WASP76b_Edwards.txt')
W76v3=np.loadtxt('./Spectra/WASP76b_Fu.txt')
K7v2=np.loadtxt('./Spectra/KELT7b_Pluriel.txt')
Kep13v2=np.loadtxt('./Spectra/Kepler13Ab_Beatty.txt')

def fakedata_bbod(wave,temp):
	#output units: erg/s/m^2/m
	Bbod=np.zeros(np.shape(wave)[0])
	for x in np.arange(np.shape(wave)[0]):
		#factor of pi to remove sterradians; factor of 10^7 converts from J to erg
		Bbod[x]=2*h*c**2./wave[x]**5./(np.exp(h*c/(wave[x]*kb*temp))-1.)*np.pi*10.**7

	return Bbod

def trapezoidint(xvals,yvals):
	total=0
	for i in np.arange(np.shape(xvals)[0]-1):
		total+=(yvals[i]+yvals[i+1])*(xvals[i+1]-xvals[i])*0.5
	return total

def planetstarflux(planetT,Startemp,Starm,Starlogg,diff,rp):
	outputwave=np.linspace(1.0*10**-6,2.0*10**-6,2000)
	outputthermal=np.zeros(np.shape(outputwave)[0])
	outputstar=np.zeros(np.shape(outputwave)[0])
	planetbbod=fakedata_bbod(outputwave,planetT)

	sp = S.Icat('k93models',Startemp,Starm,Starlogg)
	sp.convert('photlam')
	wave=sp.wave*10.**-10  #in meters
	photflux = sp.flux*10.**4*10.**10 #in photons/s/m^2/m
	fluxmks = np.zeros(np.shape(photflux)[0])
	for k in np.arange(np.shape(photflux)[0]):
		Ephoton=h*c/wave[k]
		fluxmks[k]=photflux[k]*Ephoton*10**7.	#in erg/s/m^2/m
	starset=np.interp(outputwave,wave,fluxmks)
	
	for i in np.arange(np.shape(outputwave)[0]):
		wave1=outputwave[i]-diff/2.
		wave2=outputwave[i]+diff/2.
		outputthermal[i]=trapezoidint(outputwave[(outputwave>wave1)&(outputwave<wave2)],planetbbod[(outputwave>wave1)&(outputwave<wave2)])
		outputstar[i]=trapezoidint(outputwave[(outputwave>wave1)&(outputwave<wave2)],starset[(outputwave>wave1)&(outputwave<wave2)])
	finalratio=outputthermal/outputstar*rp**2.
	smoothed=smooth.smooth(finalratio)
	return smoothed

waves=np.linspace(1.0*10**-6,2.0*10**-6,2000)
allplanetparams=np.loadtxt('planetparams.txt')

smoothedH41=planetstarflux(allplanetparams[3,5],allplanetparams[3,0],allplanetparams[3,2],allplanetparams[3,3],np.mean(np.diff(H41[:,0]))*10**-6.,allplanetparams[3,4])
smoothedK7=planetstarflux(allplanetparams[6,5],allplanetparams[6,0],allplanetparams[6,2],allplanetparams[6,3],np.mean(np.diff(K7[:,0]))*10**-6.,allplanetparams[6,4])
smoothedW74=planetstarflux(allplanetparams[14,5],allplanetparams[14,0],allplanetparams[14,2],allplanetparams[14,3],np.mean(np.diff(W74[:,0]))*10**-6.,allplanetparams[14,4])
smoothedW76=planetstarflux(allplanetparams[15,5],allplanetparams[15,0],allplanetparams[15,2],allplanetparams[15,3],np.mean(np.diff(W76[:,0]))*10**-6.,allplanetparams[15,4])
smoothedW79=planetstarflux(allplanetparams[16,5],allplanetparams[16,0],allplanetparams[16,2],allplanetparams[16,3],np.mean(np.diff(W79[:,0]))*10**-6.,allplanetparams[16,4])
smoothedW121=planetstarflux(allplanetparams[18,5],allplanetparams[18,0],allplanetparams[18,2],allplanetparams[18,3],np.mean(np.diff(W121[:,0]))*10**-6.,allplanetparams[18,4])
smoothedKep13=planetstarflux(allplanetparams[7,5],allplanetparams[7,0],allplanetparams[7,2],allplanetparams[7,3],np.mean(np.diff(Kep13[:,0]))*10**-6.,allplanetparams[7,4])

rc('axes',linewidth=2)

fig,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8))=plt.subplots(4,2,figsize=(15,14))
fig.delaxes(ax8)
[label.set_visible(False) for label in ax1.get_xticklabels()]
[label.set_visible(False) for label in ax2.get_xticklabels()]
[label.set_visible(False) for label in ax3.get_xticklabels()]
[label.set_visible(False) for label in ax4.get_xticklabels()]
[label.set_visible(False) for label in ax5.get_xticklabels()]

ax3.errorbar(H41[:,0],H41[:,1],yerr=H41[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',label='HAT-P-41b',zorder=1)
ax4.errorbar(K7[:,0],K7[:,1],yerr=K7[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',zorder=2)
ax4.errorbar(K7v2[:,0],K7v2[:,1],yerr=K7v2[:,2],color='r',linewidth=3,marker='.',markersize=15,linestyle='none',label='Pluriel+20',zorder=1)
ax2.errorbar(W74[:,0],W74[:,1],yerr=W74[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',label='WASP-74b',zorder=1)
ax5.errorbar(W76[:,0],W76[:,1],yerr=W76[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',zorder=2)
ax5.errorbar(W76v2[:,0],W76v2[:,1],yerr=W76v2[:,2],color='r',linewidth=3,marker='.',markersize=15,linestyle='none',label='Edwards+20',zorder=1)
ax5.errorbar(W76v3[:,0],W76v3[:,1],yerr=W76v3[:,2],color='b',linewidth=3,marker='d',markersize=8,linestyle='none',label='Fu+20',zorder=1)
ax1.errorbar(W79[:,0],W79[:,1],yerr=W79[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',label='WASP-79b',zorder=1)
ax6.errorbar(W121[:,0],W121[:,1],yerr=W121[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',zorder=2)
ax6.errorbar(W121v2[:,0],W121v2[:,1],yerr=W121v2[:,2],color='r',linewidth=3,marker='.',markersize=15,linestyle='none',label='Mikal-Evans+20',zorder=1)
ax7.errorbar(Kep13[:,0],Kep13[:,1],yerr=Kep13[:,2],color='k',linewidth=3,marker='.',markersize=15,linestyle='none',zorder=1)
ax7.errorbar(Kep13v2[:,0],Kep13v2[:,1],yerr=Kep13v2[:,2],color='r',linewidth=3,marker='.',markersize=15,linestyle='none',label='Beatty+2017',zorder=1)

ax3.plot(waves*10**6.,smoothedH41*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')
ax4.plot(waves*10**6.,smoothedK7*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')
ax2.plot(waves*10**6.,smoothedW74*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')
ax5.plot(waves*10**6.,smoothedW76*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')
ax1.plot(waves*10**6.,smoothedW79*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')
ax6.plot(waves*10**6.,smoothedW121*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')
ax7.plot(waves*10**6.,smoothedKep13*10**6.,color='k',linewidth=3,zorder=0,linestyle='--')

fig.text(0.03,0.5,r'F$_{\mathrm{p}}$/F$_{\mathrm{s}}$ [ppm]',fontsize=20,verticalalignment='center',rotation='vertical')
ax6.set_xlabel(r'Wavelength [$\mu$m]',fontsize=20)
ax7.set_xlabel(r'Wavelength [$\mu$m]',fontsize=20)
ax3.set_title(r'HAT-P-41b, $T_{day}=2411$ K',fontsize=15)
ax4.set_title(r'KELT-7b, $T_{day}=2424$ K',fontsize=15)
ax2.set_title(r'WASP-74b, $T_{day}=2269$ K',fontsize=15)
ax5.set_title(r'WASP-76b, $T_{day}=2560$ K',fontsize=15)
ax1.set_title(r'WASP-79b, $T_{day}=1886$ K',fontsize=15)
ax6.set_title(r'WASP-121b, $T_{day}=2662$ K',fontsize=15)
ax7.set_title(r'Kepler-13Ab, $T_{day}=3385$ K',fontsize=15)
ax1.set_xlim(1.1,1.75)
ax2.set_xlim(1.1,1.75)
ax3.set_xlim(1.1,1.75)
ax4.set_xlim(1.1,1.75)
ax6.set_xlim(1.1,1.75)
ax5.set_xlim(1.1,1.75)
ax7.set_xlim(1.1,1.75)
ax1.set_ylim(-0.005*10**4.,0.04*10**4.)
ax2.set_ylim(0.02*10**4.,0.08*10**4.)
ax3.set_ylim(0.0*10**4.,0.09*10**4.)
ax4.set_ylim(0.02*10**4.,0.06*10**4.)
ax6.set_ylim(0.08*10**4.,0.15*10**4.)
ax5.set_ylim(0.03*10**4.,0.14*10**4.)
ax7.set_ylim(400,1550)
ax4.legend(fontsize=15)
ax5.legend(fontsize=15)
ax6.legend(fontsize=15)
ax7.legend(fontsize=15)
ax1.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
ax2.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
ax3.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
ax4.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
ax6.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
ax5.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
ax7.tick_params(labelsize=20,axis="both",top=True,width=2,length=8,direction='in')
plt.savefig('SupplementaryFig2.png')
plt.show()






