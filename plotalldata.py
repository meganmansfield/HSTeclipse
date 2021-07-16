import pysynphot as S
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx

#important constants
h=6.626*10.**-34
c=2.998*10.**8
kb=1.381*10.**-23

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


masterwavegrid=np.linspace(1.0*10**-6.,2.0*10**-6.,2000)

def trapezoidint(xvals,yvals):
	total=0
	for i in np.arange(np.shape(xvals)[0]-1):
		total+=(yvals[i]+yvals[i+1])*(xvals[i+1]-xvals[i])*0.5
	return total

def fakedata_bbod(wave,temp):
	#output units: erg/s/m^2/m
	Bbod=np.zeros(np.shape(wave)[0])
	for x in np.arange(np.shape(wave)[0]):
		#factor of pi to remove sterradians; factor of 10^7 converts from J to erg
		Bbod[x]=2*h*c**2./wave[x]**5./(np.exp(h*c/(wave[x]*kb*temp))-1.)*np.pi*10.**7

	return Bbod

############################# ADDING BLACKBODIES FOR EACH PLANET INDIVIDUALLY #########################
def indplanetbbod(pltemp,plwave,startemp,starmet,starlogg,rprs,short=False):
	plfinebbod=fakedata_bbod(masterwavegrid,pltemp) #erg/s/m^2/m
	sp = S.Icat('k93models',startemp,starmet,starlogg)
	sp.convert('flam') ## initial units erg/cm^2/s/Angstrom
	wave=sp.wave*10.**-10  #in meters
	flux = sp.flux*10.**4*10.**10 #in erg/m2/m/s
	starfinebbod=np.interp(masterwavegrid,wave,flux)
	diff=np.mean(np.diff(plwave))*10**-6.

	newminwave=1.1*10**-6.
	newmaxwave=1.85*10**-6.
	if short:
		newminwave=1.1*10**-6.
		newmaxwave=1.65*10**-6.
	smallgrid=masterwavegrid[(masterwavegrid>newminwave-diff/2.)&(masterwavegrid<newmaxwave+diff/2.)]
	smallplflux=plfinebbod[(masterwavegrid>newminwave-diff/2.)&(masterwavegrid<newmaxwave+diff/2.)]
	smallstarflux=starfinebbod[(masterwavegrid>newminwave-diff/2.)&(masterwavegrid<newmaxwave+diff/2.)]
	tgrid=masterwavegrid[(masterwavegrid>newminwave)&(masterwavegrid<newmaxwave)]
	modelbbod=np.zeros(np.shape(tgrid)[0])
	for i in np.arange(np.shape(tgrid)[0]):
		wave1=tgrid[i]-diff/2.
		wave2=tgrid[i]+diff/2.
		plint=trapezoidint(smallgrid[(smallgrid>wave1)&(smallgrid<wave2)],smallplflux[(smallgrid>wave1)&(smallgrid<wave2)])
		starint=trapezoidint(smallgrid[(smallgrid>wave1)&(smallgrid<wave2)],smallstarflux[(smallgrid>wave1)&(smallgrid<wave2)])
		modelbbod[i]=(plint/starint*rprs**2.)*10**6. #in ppm
	return tgrid*10**6.,modelbbod

allplanetparams=np.loadtxt('planetparams.txt')

tgridC2,BmodelC2=indplanetbbod(allplanetparams[0,5],C2[:,0],allplanetparams[0,0],allplanetparams[0,2],allplanetparams[0,3],allplanetparams[0,4],short=True)
tgridH7,BmodelH7=indplanetbbod(allplanetparams[1,5],H7[:,0],allplanetparams[1,0],allplanetparams[1,2],allplanetparams[1,3],allplanetparams[1,4])
tgridH32,BmodelH32=indplanetbbod(allplanetparams[2,5],H32[:,0],allplanetparams[2,0],allplanetparams[2,2],allplanetparams[2,3],allplanetparams[2,4],short=True)
tgridH41,BmodelH41=indplanetbbod(allplanetparams[3,5],H41[:,0],allplanetparams[3,0],allplanetparams[3,2],allplanetparams[3,3],allplanetparams[3,4])
tgridHD189,BmodelHD189=indplanetbbod(allplanetparams[4,5],HD189[:,0],allplanetparams[4,0],allplanetparams[4,2],allplanetparams[4,3],allplanetparams[4,4])
tgridHD209,BmodelHD209=indplanetbbod(allplanetparams[5,5],HD209[:,0],allplanetparams[5,0],allplanetparams[5,2],allplanetparams[5,3],allplanetparams[5,4])
tgridK7,BmodelK7=indplanetbbod(allplanetparams[6,5],K7[:,0],allplanetparams[6,0],allplanetparams[6,2],allplanetparams[6,3],allplanetparams[6,4])
tgridKep13,BmodelKep13=indplanetbbod(allplanetparams[7,5],Kep13[:,0],allplanetparams[7,0],allplanetparams[7,2],allplanetparams[7,3],allplanetparams[7,4])
tgridT3,BmodelT3=indplanetbbod(allplanetparams[8,5],T3[:,0],allplanetparams[8,0],allplanetparams[8,2],allplanetparams[8,3],allplanetparams[8,4],short=True)
tgridW4,BmodelW4=indplanetbbod(allplanetparams[9,5],W4[:,0],allplanetparams[9,0],allplanetparams[9,2],allplanetparams[9,3],allplanetparams[9,4])
tgridW12,BmodelW12=indplanetbbod(allplanetparams[10,5],W12[:,0],allplanetparams[10,0],allplanetparams[10,2],allplanetparams[10,3],allplanetparams[10,4])
tgridW18,BmodelW18=indplanetbbod(allplanetparams[11,5],W18[:,0],allplanetparams[11,0],allplanetparams[11,2],allplanetparams[11,3],allplanetparams[11,4])
tgridW33,BmodelW33=indplanetbbod(allplanetparams[12,5],W33[:,0],allplanetparams[12,0],allplanetparams[12,2],allplanetparams[12,3],allplanetparams[12,4])
tgridW43,BmodelW43=indplanetbbod(allplanetparams[13,5],W43[:,0],allplanetparams[13,0],allplanetparams[13,2],allplanetparams[13,3],allplanetparams[13,4])
tgridW74,BmodelW74=indplanetbbod(allplanetparams[14,5],W74[:,0],allplanetparams[14,0],allplanetparams[14,2],allplanetparams[14,3],allplanetparams[14,4])
tgridW76,BmodelW76=indplanetbbod(allplanetparams[15,5],W76[:,0],allplanetparams[15,0],allplanetparams[15,2],allplanetparams[15,3],allplanetparams[15,4])
tgridW79,BmodelW79=indplanetbbod(allplanetparams[16,5],W79[:,0],allplanetparams[16,0],allplanetparams[16,2],allplanetparams[16,3],allplanetparams[16,4])
tgridW103,BmodelW103=indplanetbbod(allplanetparams[17,5],W103[:,0],allplanetparams[17,0],allplanetparams[17,2],allplanetparams[17,3],allplanetparams[17,4])
tgridW121,BmodelW121=indplanetbbod(allplanetparams[18,5],W121[:,0],allplanetparams[18,0],allplanetparams[18,2],allplanetparams[18,3],allplanetparams[18,4])

################################ ADD BEST-FIT MODELS INTERPOLATED FROM SC-CHIMERA FIDUCIAL GRID ##################
FmodelC2fid=np.loadtxt('./FiducialModels/modelfidC2short.txt')
FmodelH7fid=np.loadtxt('./FiducialModels/modelfidH7.txt')
FmodelH32fid=np.loadtxt('./FiducialModels/modelfidH32short.txt')
FmodelH41fid=np.loadtxt('./FiducialModels/modelfidH41.txt')
FmodelHD189fid=np.loadtxt('./FiducialModels/modelfidHD189.txt')
FmodelHD209fid=np.loadtxt('./FiducialModels/modelfidHD209.txt')
FmodelK7fid=np.loadtxt('./FiducialModels/modelfidK7.txt')
FmodelKep13fid=np.loadtxt('./FiducialModels/modelfidKep13.txt')
FmodelT3fid=np.loadtxt('./FiducialModels/modelfidT3short.txt')
FmodelW4fid=np.loadtxt('./FiducialModels/modelfidW4.txt')
FmodelW12fid=np.loadtxt('./FiducialModels/modelfidW12.txt')
FmodelW18fid=np.loadtxt('./FiducialModels/modelfidW18.txt')
FmodelW33fid=np.loadtxt('./FiducialModels/modelfidW33.txt')
FmodelW43fid=np.loadtxt('./FiducialModels/modelfidW43.txt')
FmodelW74fid=np.loadtxt('./FiducialModels/modelfidW74.txt')
FmodelW76fid=np.loadtxt('./FiducialModels/modelfidW76.txt')
FmodelW79fid=np.loadtxt('./FiducialModels/modelfidW79.txt')
FmodelW103fid=np.loadtxt('./FiducialModels/modelfidW103.txt')
FmodelW121fid=np.loadtxt('./FiducialModels/modelfidW121.txt')

FmodelC2=np.loadtxt('./BestFitModels/modelbestfitC2short.txt')
FmodelH7=np.loadtxt('./BestFitModels/modelbestfitH7.txt')
FmodelH32=np.loadtxt('./BestFitModels/modelbestfitH32short.txt')
FmodelH41=np.loadtxt('./BestFitModels/modelbestfitH41.txt')
FmodelHD189=np.loadtxt('./BestFitModels/modelbestfitHD189.txt')
FmodelHD209=np.loadtxt('./BestFitModels/modelbestfitHD209.txt')
FmodelK7=np.loadtxt('./BestFitModels/modelbestfitK7.txt')
FmodelKep13=np.loadtxt('./BestFitModels/modelbestfitKep13.txt')
FmodelT3=np.loadtxt('./BestFitModels/modelbestfitT3short.txt')
FmodelW4=np.loadtxt('./BestFitModels/modelbestfitW4.txt')
FmodelW12=np.loadtxt('./BestFitModels/modelbestfitW12.txt')
FmodelW18=np.loadtxt('./BestFitModels/modelbestfitW18.txt')
FmodelW33=np.loadtxt('./BestFitModels/modelbestfitW33.txt')
FmodelW43=np.loadtxt('./BestFitModels/modelbestfitW43.txt')
FmodelW74=np.loadtxt('./BestFitModels/modelbestfitW74.txt')
FmodelW76=np.loadtxt('./BestFitModels/modelbestfitW76.txt')
FmodelW79=np.loadtxt('./BestFitModels/modelbestfitW79.txt')
FmodelW103=np.loadtxt('./BestFitModels/modelbestfitW103.txt')
FmodelW121=np.loadtxt('./BestFitModels/modelbestfitW121.txt')

#daytemp order: HD189, HD209, W43, C2, T3, H32, W4, W79, W74, K7, H41, W76, W121, H7, W12, W18, W103, W33, Kep13
#daytemps=np.array([1446., 1711., 1775., 1796., 1842., 1939., 2079., 2083., 2298., 2447., 2461., 2523., 2651., 2772., 2890., 2979., 3018., 3126., 3484.])
daytemps=allplanetparams[:,6]
vmin=np.min(daytemps)
vmax=np.max(daytemps)+300.
normequtemps=(daytemps-vmin)/np.max(daytemps-vmin)
inferno = cm = plt.get_cmap('inferno') 
cNorm  = mplcolors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=inferno)

#space=650.
rc('axes',linewidth=2)


##################### Plot showing interpolated fiducial spectra ####################
space=650.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,12))
ax1.errorbar(Kep13[:,0],(Kep13[:,1]-np.median(Kep13[:,1]))+9*space,Kep13[:,2],color=scalarMap.to_rgba(daytemps[7]),linestyle='none',marker='.',markersize=15,linewidth=3,label='Kepler-13Ab',zorder=2)
ax1.errorbar(W33[:,0],(W33[:,1]-np.median(W33[:,1]))+8*space,W33[:,2],color=scalarMap.to_rgba(daytemps[12]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-33b',zorder=2)
ax1.errorbar(W103[:,0],(W103[:,1]-np.median(W103[:,1]))+7*space,W103[:,2],color=scalarMap.to_rgba(daytemps[17]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-103b',zorder=2)
ax1.errorbar(W18[:,0],(W18[:,1]-np.median(W18[:,1]))+6*space,W18[:,2],color=scalarMap.to_rgba(daytemps[11]),linestyle='none',marker='.',markersize=15,linewidth=3,label="WASP-18b",zorder=2)
ax1.errorbar(W12[:,0],(W12[:,1]-np.median(W12[:,1]))+5*space,W12[:,2],color=scalarMap.to_rgba(daytemps[10]),linestyle='none',marker='.',markersize=15,linewidth=3,label="WASP-12b",zorder=2)
ax1.errorbar(H7[:,0],(H7[:,1]-np.median(H7[:,1]))+4*space,H7[:,2],color=scalarMap.to_rgba(daytemps[1]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HAT-P-7b',zorder=2)
ax1.errorbar(W121[:,0],(W121[:,1]-np.median(W121[:,1]))+3*space,W121[:,2],color=scalarMap.to_rgba(daytemps[18]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-121b',zorder=2)
ax1.errorbar(W76[:,0],(W76[:,1]-np.median(W76[:,1]))+2*space,W76[:,2],color=scalarMap.to_rgba(daytemps[15]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-76b',zorder=2)
ax1.errorbar(H41[:,0],(H41[:,1]-np.median(H41[:,1]))+1*space,H41[:,2],color=scalarMap.to_rgba(daytemps[3]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HAT-P-41b',zorder=2)
ax1.errorbar(K7[:,0],(K7[:,1]-np.median(K7[:,1])),K7[:,2],color=scalarMap.to_rgba(daytemps[6]),linestyle='none',marker='.',markersize=15,linewidth=3,label='KELT-7b',zorder=2)
ax2.errorbar(W74[:,0],(W74[:,1]-np.median(W74[:,1]))+8*space,W74[:,2],color=scalarMap.to_rgba(daytemps[14]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-74b',zorder=2)
ax2.errorbar(W79[:,0],(W79[:,1]-np.median(W79[:,1]))+7*space,W79[:,2],color=scalarMap.to_rgba(daytemps[16]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-79b',zorder=2)
ax2.errorbar(W4[:,0],(W4[:,1]-np.median(W4[:,1]))+6*space,W4[:,2],color=scalarMap.to_rgba(daytemps[9]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-4b',zorder=2)
ax2.errorbar(H32[:,0],(H32[:,1]-np.median(H32[:,1]))+5*space,H32[:,2],color=scalarMap.to_rgba(daytemps[2]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HAT-P-32b',zorder=2)
ax2.errorbar(T3[:,0],(T3[:,1]-np.median(T3[:,1]))+4*space,T3[:,2],color=scalarMap.to_rgba(daytemps[8]),linestyle='none',marker='.',markersize=15,linewidth=3,label='TrES-3b',zorder=2)
ax2.errorbar(C2[:,0],(C2[:,1]-np.median(C2[:,1]))+3*space,C2[:,2],color=scalarMap.to_rgba(daytemps[0]),linestyle='none',marker='.',markersize=15,linewidth=3,label='CoRoT-2b',zorder=2)
ax2.errorbar(W43[:,0],(W43[:,1]-np.median(W43[:,1]))+2*space,W43[:,2],color=scalarMap.to_rgba(daytemps[13]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-43b',zorder=2)
ax2.errorbar(HD209[:,0],(HD209[:,1]-np.median(HD209[:,1]))*2.+1*space,HD209[:,2]*2.,color=scalarMap.to_rgba(daytemps[5]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HD 209458b',zorder=2)
ax2.errorbar(HD189[:,0],(HD189[:,1]-np.median(HD189[:,1]))*2.,HD189[:,2]*2.,color=scalarMap.to_rgba(daytemps[4]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HD 189733b',zorder=2)

ax1.text(1.13,5950,'Kepler-\n13Ab',color=scalarMap.to_rgba(daytemps[7]),fontsize=15)
ax1.text(1.13,5300,'WASP-33b',color=scalarMap.to_rgba(daytemps[12]),fontsize=15)
ax1.text(1.13,4500,'WASP-103b',color=scalarMap.to_rgba(daytemps[17]),fontsize=15)
ax1.text(1.13,3850,'WASP-18b',color=scalarMap.to_rgba(daytemps[11]),fontsize=15)
ax1.text(1.13,3250,'WASP-12b',color=scalarMap.to_rgba(daytemps[10]),fontsize=15)
ax1.text(1.13,2590,'HAT-P-7b',color=scalarMap.to_rgba(daytemps[1]),fontsize=15)
ax1.text(1.13,1850,'WASP-121b',color=scalarMap.to_rgba(daytemps[18]),fontsize=15)
ax1.text(1.13,1250,'WASP-76b',color=scalarMap.to_rgba(daytemps[15]),fontsize=15)
ax1.text(1.13,810,'HAT-P-41b',color=scalarMap.to_rgba(daytemps[3]),fontsize=15)
ax1.text(1.13,50,'KELT-7b',color=scalarMap.to_rgba(daytemps[6]),fontsize=15)
ax2.text(1.13,5300,'WASP-74b',color=scalarMap.to_rgba(daytemps[14]),fontsize=15)
ax2.text(1.13,4775,'WASP-79b',color=scalarMap.to_rgba(daytemps[16]),fontsize=15)
ax2.text(1.13,4100,'WASP-4b',color=scalarMap.to_rgba(daytemps[9]),fontsize=15)
ax2.text(1.13,3325,'HAT-P-32Ab',color=scalarMap.to_rgba(daytemps[2]),fontsize=15)
ax2.text(1.13,2850,'TrES-3b',color=scalarMap.to_rgba(daytemps[8]),fontsize=15)
ax2.text(1.13,2100,'CoRoT-2b',color=scalarMap.to_rgba(daytemps[0]),fontsize=15)
ax2.text(1.13,1500,'WASP-43b',color=scalarMap.to_rgba(daytemps[13]),fontsize=15)
ax2.text(1.13,800,'HD 209458b',color=scalarMap.to_rgba(daytemps[5]),fontsize=15)
ax2.text(1.13,250,'HD 189733b',color=scalarMap.to_rgba(daytemps[4]),fontsize=15)

ax1.plot(tgridKep13,BmodelKep13-np.median(Kep13[:,1])+9*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW33,BmodelW33-np.median(W33[:,1])+8*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW103,BmodelW103-np.median(W103[:,1])+7*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW18,BmodelW18-np.median(W18[:,1])+6*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW12,BmodelW12-np.median(W12[:,1])+5*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridH7,BmodelH7-np.median(H7[:,1])+4*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW121,BmodelW121-np.median(W121[:,1])+3*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW76,BmodelW76-np.median(W76[:,1])+2*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridH41,BmodelH41-np.median(H41[:,1])+1*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridK7,BmodelK7-np.median(K7[:,1]),color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW74,BmodelW74-np.median(W74[:,1])+8*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW79,BmodelW79-np.median(W79[:,1])+7*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW4,BmodelW4-np.median(W4[:,1])+6*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridH32,BmodelH32-np.median(H32[:,1])+5*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridT3,BmodelT3-np.median(T3[:,1])+4*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridC2,BmodelC2-np.median(C2[:,1])+3*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW43,BmodelW43-np.median(W43[:,1])+2*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridHD209,(BmodelHD209-np.median(HD209[:,1]))*2.+1*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridHD189,(BmodelHD189-np.median(HD189[:,1]))*2.,color='k',linestyle='--',linewidth=2,zorder=0)

ax1.plot(FmodelKep13[:,0],FmodelKep13[:,1]*10**6.-np.median(Kep13[:,1])+9*space,color=scalarMap.to_rgba(daytemps[7]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW33[:,0],FmodelW33[:,1]*10**6.-np.median(W33[:,1])+8*space,color=scalarMap.to_rgba(daytemps[12]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW103[:,0],FmodelW103[:,1]*10**6.-np.median(W103[:,1])+7*space,color=scalarMap.to_rgba(daytemps[17]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW18[:,0],FmodelW18[:,1]*10**6.-np.median(W18[:,1])+6*space,color=scalarMap.to_rgba(daytemps[11]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW12[:,0],FmodelW12[:,1]*10**6.-np.median(W12[:,1])+5*space,color=scalarMap.to_rgba(daytemps[10]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelH7[:,0],FmodelH7[:,1]*10**6.-np.median(H7[:,1])+4*space,color=scalarMap.to_rgba(daytemps[1]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW121[:,0],FmodelW121[:,1]*10**6.-np.median(W121[:,1])+3*space,color=scalarMap.to_rgba(daytemps[18]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW76[:,0],FmodelW76[:,1]*10**6.-np.median(W76[:,1])+2*space,color=scalarMap.to_rgba(daytemps[15]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelH41[:,0],FmodelH41[:,1]*10**6.-np.median(H41[:,1])+1*space,color=scalarMap.to_rgba(daytemps[3]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelK7[:,0],FmodelK7[:,1]*10**6.-np.median(K7[:,1]),color=scalarMap.to_rgba(daytemps[6]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW74[:,0],FmodelW74[:,1]*10**6.-np.median(W74[:,1])+8*space,color=scalarMap.to_rgba(daytemps[14]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW79[:,0],FmodelW79[:,1]*10**6.-np.median(W79[:,1])+7*space,color=scalarMap.to_rgba(daytemps[16]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW4[:,0],FmodelW4[:,1]*10**6.-np.median(W4[:,1])+6*space,color=scalarMap.to_rgba(daytemps[9]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelH32[:,0],FmodelH32[:,1]*10**6.-np.median(H32[:,1])+5*space,color=scalarMap.to_rgba(daytemps[2]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelT3[:,0],FmodelT3[:,1]*10**6.-np.median(T3[:,1])+4*space,color=scalarMap.to_rgba(daytemps[8]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelC2[:,0],FmodelC2[:,1]*10**6.-np.median(C2[:,1])+3*space,color=scalarMap.to_rgba(daytemps[0]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW43[:,0],FmodelW43[:,1]*10**6.-np.median(W43[:,1])+2*space,color=scalarMap.to_rgba(daytemps[13]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelHD209[:,0],(FmodelHD209[:,1]*10**6.-np.median(HD209[:,1]))*2.+1*space,color=scalarMap.to_rgba(daytemps[5]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelHD189[:,0],(FmodelHD189[:,1]*10**6.-np.median(HD189[:,1]))*2.,color=scalarMap.to_rgba(daytemps[4]),linestyle='-',linewidth=2,zorder=1)

ax1.set_xlim(1.1,1.66)
ax1.set_ylim(-250,6500)
ax1.set_xlabel('Wavelength [micron]',fontsize=20)
ax1.set_ylabel('Fp/Fs [ppm]',fontsize=20)
ax1.axvspan(1.22,1.33,color='k',alpha=0.2,zorder=0)
ax1.axvspan(1.53,1.61,color='k',alpha=0.2,zorder=0)
ax1.axvspan(1.35,1.48,color='k',alpha=0.2,zorder=0) #FINDME: having a problem with the yticks in subplots
ax1.tick_params(labelsize=15,axis="both",top=True,width=2,length=8,direction='in') #-96,54,204,498,648,798
ax1.set_yticks(ticks=[-152,-2,148,-153+1*space,-3+1*space,147+1*space,\
	-221+2*space,-71+2*space,79+2*space,-186+3*space,-36+3*space,114+3*space,\
	-170+4*space,-20+4*space,130+4*space,-310+5*space,-60+5*space,190+5*space,\
	-178+6*space,-28+6*space,121+6*space,-242+7*space,-42+7*space,158+7*space,\
	-150+8*space,8*space,150+8*space,-187+9*space,13+9*space,213+9*space])
ax1.set_yticklabels(labels=[str("{:.0f}".format(np.median(K7[:,1])-152)),str("{:.0f}".format(np.median(K7[:,1])-2)),str("{:.0f}".format(np.median(K7[:,1])+148)),\
	str("{:.0f}".format(np.median(H41[:,1])-153)),str("{:.0f}".format(np.median(H41[:,1])-3)),str("{:.0f}".format(np.median(H41[:,1])+147)),\
	str("{:.0f}".format(np.median(W76[:,1])-221)),str("{:.0f}".format(np.median(W76[:,1])-71)),str("{:.0f}".format(np.median(W76[:,1])+79)),\
	str("{:.0f}".format(np.median(W121[:,1])-186)),str("{:.0f}".format(np.median(W121[:,1])-36)),str("{:.0f}".format(np.median(W121[:,1])+114)),\
	str("{:.0f}".format(np.median(H7[:,1])-170)),str("{:.0f}".format(np.median(H7[:,1])-20)),str("{:.0f}".format(np.median(H7[:,1])+130)),\
	str("{:.0f}".format(np.median(W12[:,1])-310)),str("{:.0f}".format(np.median(W12[:,1])-60)),str("{:.0f}".format(np.median(W12[:,1])+190)),\
	str("{:.0f}".format(np.median(W18[:,1])-178)),str("{:.0f}".format(np.median(W18[:,1])-28)),str("{:.0f}".format(np.median(W18[:,1])+121)),\
	str("{:.0f}".format(np.median(W103[:,1])-242)),str("{:.0f}".format(np.median(W103[:,1])-42)),str("{:.0f}".format(np.median(W103[:,1])+158)),\
	str("{:.0f}".format(np.median(W33[:,1])-150)),str("{:.0f}".format(np.median(W33[:,1]))),str("{:.0f}".format(np.median(W33[:,1])+150)),\
	str("{:.0f}".format(np.median(Kep13[:,1])-187)),str("{:.0f}".format(np.median(Kep13[:,1])+13)),str("{:.0f}".format(np.median(Kep13[:,1])+213))])

ax2.set_xlim(1.1,1.85)
ax2.set_ylim(-400,5500)
ax2.set_xlabel('Wavelength [micron]',fontsize=20)
ax2.axvspan(1.22,1.33,color='k',alpha=0.2,zorder=0)
ax2.axvspan(1.53,1.61,color='k',alpha=0.2,zorder=0)
ax2.axvspan(1.35,1.48,color='k',alpha=0.2,zorder=0)
ax2.tick_params(labelsize=15,axis="both",top=True,width=2,length=8,direction='in') #-96,54,204,498,648,798
ax2.set_yticks(ticks=[-188,12,212,-152+1*space,-2+1*space,148+1*space,\
	-139+2*space,11+2*space,161+2*space,-142+3*space,8+3*space,158+3*space,\
	-150+4*space,4*space,150+4*space,-169+5*space,-19+5*space,131+5*space,\
	-220+6*space,-20+6*space,180+6*space,-177+7*space,-27+7*space,123+7*space,\
	-117+8*space,33+8*space,183+8*space])
ax2.set_yticklabels(labels=[0,100,200,\
	25,100,175,\
	str("{:.0f}".format(np.median(W43[:,1])-139)),str("{:.0f}".format(np.median(W43[:,1])+11)),str("{:.0f}".format(np.median(W43[:,1])+161)),\
	str("{:.0f}".format(np.median(C2[:,1])-142)),str("{:.0f}".format(np.median(C2[:,1])+8)),str("{:.0f}".format(np.median(C2[:,1])+158)),\
	str("{:.0f}".format(np.median(T3[:,1])-150)),str("{:.0f}".format(np.median(T3[:,1])-0)),str("{:.0f}".format(np.median(T3[:,1])+150)),\
	str("{:.0f}".format(np.median(H32[:,1])-169)),str("{:.0f}".format(np.median(H32[:,1])-19)),str("{:.0f}".format(np.median(H32[:,1])+131)),\
	str("{:.0f}".format(np.median(W4[:,1])-220)),str("{:.0f}".format(np.median(W4[:,1])-20)),str("{:.0f}".format(np.median(W4[:,1])+180)),\
	str("{:.0f}".format(np.median(W79[:,1])-176.7)),str("{:.0f}".format(np.median(W79[:,1])-27)),str("{:.0f}".format(np.median(W79[:,1])+123)),\
	str("{:.0f}".format(np.median(W74[:,1])-117)),str("{:.0f}".format(np.median(W74[:,1])+33)),str("{:.0f}".format(np.median(W74[:,1])+183))])

a = plt.axes([0.91,0.40,0.01,0.36], frameon=False)
a.yaxis.set_visible(False)
a.xaxis.set_visible(False)
a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.inferno, aspect='auto', visible=False)
cbar=plt.colorbar(a, fraction=3.0)
cbar.ax.tick_params(labelsize=15,width=2,length=6)
cbar.set_label('Dayside Temperature [K]',fontsize=15)


plt.tight_layout()
plt.savefig('allspectra_fiducial.png')
plt.show()

##################### Plot showing best fit spectra #########################################
space=950.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,12))
ax1.errorbar(Kep13[:,0],(Kep13[:,1]-np.median(Kep13[:,1]))+9*space,Kep13[:,2],color=scalarMap.to_rgba(daytemps[7]),linestyle='none',marker='.',markersize=15,linewidth=3,label='Kepler-13Ab',zorder=2)
ax1.errorbar(W33[:,0],(W33[:,1]-np.median(W33[:,1]))+8*space,W33[:,2],color=scalarMap.to_rgba(daytemps[12]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-33b',zorder=2)
ax1.errorbar(W103[:,0],(W103[:,1]-np.median(W103[:,1]))+7*space,W103[:,2],color=scalarMap.to_rgba(daytemps[17]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-103b',zorder=2)
ax1.errorbar(W18[:,0],(W18[:,1]-np.median(W18[:,1]))+6*space,W18[:,2],color=scalarMap.to_rgba(daytemps[11]),linestyle='none',marker='.',markersize=15,linewidth=3,label="WASP-18b",zorder=2)
ax1.errorbar(W12[:,0],(W12[:,1]-np.median(W12[:,1]))+5*space,W12[:,2],color=scalarMap.to_rgba(daytemps[10]),linestyle='none',marker='.',markersize=15,linewidth=3,label="WASP-12b",zorder=2)
ax1.errorbar(H7[:,0],(H7[:,1]-np.median(H7[:,1]))+4*space,H7[:,2],color=scalarMap.to_rgba(daytemps[1]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HAT-P-7b',zorder=2)
ax1.errorbar(W121[:,0],(W121[:,1]-np.median(W121[:,1]))+3*space,W121[:,2],color=scalarMap.to_rgba(daytemps[18]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-121b',zorder=2)
ax1.errorbar(W76[:,0],(W76[:,1]-np.median(W76[:,1]))+2*space,W76[:,2],color=scalarMap.to_rgba(daytemps[15]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-76b',zorder=2)
ax1.errorbar(H41[:,0],(H41[:,1]-np.median(H41[:,1]))+1*space,H41[:,2],color=scalarMap.to_rgba(daytemps[3]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HAT-P-41b',zorder=2)
ax1.errorbar(K7[:,0],(K7[:,1]-np.median(K7[:,1])),K7[:,2],color=scalarMap.to_rgba(daytemps[6]),linestyle='none',marker='.',markersize=15,linewidth=3,label='KELT-7b',zorder=2)
ax2.errorbar(W74[:,0],(W74[:,1]-np.median(W74[:,1]))+8*space,W74[:,2],color=scalarMap.to_rgba(daytemps[14]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-74b',zorder=2)
ax2.errorbar(W79[:,0],(W79[:,1]-np.median(W79[:,1]))+7*space,W79[:,2],color=scalarMap.to_rgba(daytemps[16]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-79b',zorder=2)
ax2.errorbar(W4[:,0],(W4[:,1]-np.median(W4[:,1]))+6*space,W4[:,2],color=scalarMap.to_rgba(daytemps[9]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-4b',zorder=2)
ax2.errorbar(H32[:,0],(H32[:,1]-np.median(H32[:,1]))+5*space,H32[:,2],color=scalarMap.to_rgba(daytemps[2]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HAT-P-32b',zorder=2)
ax2.errorbar(T3[:,0],(T3[:,1]-np.median(T3[:,1]))+4*space,T3[:,2],color=scalarMap.to_rgba(daytemps[8]),linestyle='none',marker='.',markersize=15,linewidth=3,label='TrES-3b',zorder=2)
ax2.errorbar(C2[:,0],(C2[:,1]-np.median(C2[:,1]))+3*space,C2[:,2],color=scalarMap.to_rgba(daytemps[0]),linestyle='none',marker='.',markersize=15,linewidth=3,label='CoRoT-2b',zorder=2)
ax2.errorbar(W43[:,0],(W43[:,1]-np.median(W43[:,1]))+2*space,W43[:,2],color=scalarMap.to_rgba(daytemps[13]),linestyle='none',marker='.',markersize=15,linewidth=3,label='WASP-43b',zorder=2)
ax2.errorbar(HD209[:,0],(HD209[:,1]-np.median(HD209[:,1]))*2.+1*space,HD209[:,2]*2.,color=scalarMap.to_rgba(daytemps[5]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HD 209458b',zorder=2)
ax2.errorbar(HD189[:,0],(HD189[:,1]-np.median(HD189[:,1]))*2.,HD189[:,2]*2.,color=scalarMap.to_rgba(daytemps[4]),linestyle='none',marker='.',markersize=15,linewidth=3,label='HD 189733b',zorder=2)

ax1.text(1.13,8750,r'Kepler-13Ab' '\n' r'C/O=0.01, $\chi^{2}_{\nu}=1.0$',color=scalarMap.to_rgba(daytemps[7]),fontsize=15)
ax1.text(1.13,7700,r'WASP-33b' '\n' r'C/O=0.85, $\chi^{2}_{\nu}=15.9$',color=scalarMap.to_rgba(daytemps[12]),fontsize=15)
ax1.text(1.13,6600,r'WASP-103b' '\n' r'[M/H]=-1.5, $\chi^{2}_{\nu}=1.5$',color=scalarMap.to_rgba(daytemps[17]),fontsize=15)
ax1.text(1.13,5650,r'WASP-18b' '\n' r'C/O=0.01, $\chi^{2}_{\nu}=1.8$',color=scalarMap.to_rgba(daytemps[11]),fontsize=15)
ax1.text(1.13,4750,r'WASP-12b' '\n' r'C/O=0.01, $\chi^{2}_{\nu}=1.2$',color=scalarMap.to_rgba(daytemps[10]),fontsize=15)
ax1.text(1.125,3850,r'HAT-P-7b'  '\n' r'C/O=0.85, $\chi^{2}_{\nu}=0.6$',color=scalarMap.to_rgba(daytemps[1]),fontsize=15)
ax1.text(1.13,2760,r'WASP-121b'  '\n' r'C/O=0.85, $\chi^{2}_{\nu}=1.8$',color=scalarMap.to_rgba(daytemps[18]),fontsize=15)
ax1.text(1.13,1900,r'WASP-76b'  '\n' r'$T_{*}=6300$ K, $\chi^{2}_{\nu}=5.9$',color=scalarMap.to_rgba(daytemps[15]),fontsize=15)
ax1.text(1.13,1100,r'HAT-P-41b'  '\n' r'[M/H]=1.5, $\chi^{2}_{\nu}=0.9$',color=scalarMap.to_rgba(daytemps[3]),fontsize=15)
ax1.text(1.13,40,r'KELT-7b'  '\n' r'C/O=0.85, $\chi^{2}_{\nu}=0.7$',color=scalarMap.to_rgba(daytemps[6]),fontsize=15)
ax2.text(1.13,7700,r'WASP-74b' '\n' r'[M/H]=1.5, $\chi^{2}_{\nu}=0.7$',color=scalarMap.to_rgba(daytemps[14]),fontsize=15)
ax2.text(1.13,6875,r'WASP-79b' '\n' r'C/O=0.01, $\chi^{2}_{\nu}=2.6$',color=scalarMap.to_rgba(daytemps[16]),fontsize=15)
ax2.text(1.13,5900,r'WASP-4b' '\n' r'[M/H]=-1.5, $\chi^{2}_{\nu}=0.6$',color=scalarMap.to_rgba(daytemps[9]),fontsize=15)
ax2.text(1.13,4825,r'HAT-P-32Ab, [M/H]=-1.5' '\n' r'$\chi^{2}_{\nu}=1.2$',color=scalarMap.to_rgba(daytemps[2]),fontsize=15)
ax2.text(1.13,4070,r'TrES-3b' '\n' r'TF19 Int. Heat, $\chi^{2}_{\nu}=0.1$',color=scalarMap.to_rgba(daytemps[8]),fontsize=15)
ax2.text(1.13,3000,r'CoRoT-2b' '\n' r'C/O=0.85, $\chi^{2}_{\nu}=1.5$',color=scalarMap.to_rgba(daytemps[0]),fontsize=15)
ax2.text(1.13,2100,r'WASP-43b' '\n' r'$T_{*}=4300$ K, $\chi^{2}_{\nu}=1.1$',color=scalarMap.to_rgba(daytemps[13]),fontsize=15)
ax2.text(1.13,1100,r'HD 209458b' '\n' r'$T_{*}=6300$ K, $\chi^{2}_{\nu}=1.2$',color=scalarMap.to_rgba(daytemps[5]),fontsize=15)
ax2.text(1.13,250,r'HD 189733b' '\n' r'C/O=0.85, $\chi^{2}_{\nu}=0.5$',color=scalarMap.to_rgba(daytemps[4]),fontsize=15)

ax1.plot(tgridKep13,BmodelKep13-np.median(Kep13[:,1])+9*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW33,BmodelW33-np.median(W33[:,1])+8*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW103,BmodelW103-np.median(W103[:,1])+7*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW18,BmodelW18-np.median(W18[:,1])+6*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW12,BmodelW12-np.median(W12[:,1])+5*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridH7,BmodelH7-np.median(H7[:,1])+4*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW121,BmodelW121-np.median(W121[:,1])+3*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridW76,BmodelW76-np.median(W76[:,1])+2*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridH41,BmodelH41-np.median(H41[:,1])+1*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax1.plot(tgridK7,BmodelK7-np.median(K7[:,1]),color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW74,BmodelW74-np.median(W74[:,1])+8*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW79,BmodelW79-np.median(W79[:,1])+7*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW4,BmodelW4-np.median(W4[:,1])+6*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridH32,BmodelH32-np.median(H32[:,1])+5*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridT3,BmodelT3-np.median(T3[:,1])+4*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridC2,BmodelC2-np.median(C2[:,1])+3*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridW43,BmodelW43-np.median(W43[:,1])+2*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridHD209,(BmodelHD209-np.median(HD209[:,1]))*2.+1*space,color='k',linestyle='--',linewidth=2,zorder=0)
ax2.plot(tgridHD189,(BmodelHD189-np.median(HD189[:,1]))*2.,color='k',linestyle='--',linewidth=2,zorder=0)

ax1.plot(FmodelKep13fid[:,0],FmodelKep13fid[:,1]*10**6.-np.median(Kep13[:,1])+9*space,color=scalarMap.to_rgba(daytemps[7]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW33fid[:,0],FmodelW33fid[:,1]*10**6.-np.median(W33[:,1])+8*space,color=scalarMap.to_rgba(daytemps[12]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW103fid[:,0],FmodelW103fid[:,1]*10**6.-np.median(W103[:,1])+7*space,color=scalarMap.to_rgba(daytemps[17]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW18fid[:,0],FmodelW18fid[:,1]*10**6.-np.median(W18[:,1])+6*space,color=scalarMap.to_rgba(daytemps[11]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW12fid[:,0],FmodelW12fid[:,1]*10**6.-np.median(W12[:,1])+5*space,color=scalarMap.to_rgba(daytemps[10]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelH7fid[:,0],FmodelH7fid[:,1]*10**6.-np.median(H7[:,1])+4*space,color=scalarMap.to_rgba(daytemps[1]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW121fid[:,0],FmodelW121fid[:,1]*10**6.-np.median(W121[:,1])+3*space,color=scalarMap.to_rgba(daytemps[18]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelW76fid[:,0],FmodelW76fid[:,1]*10**6.-np.median(W76[:,1])+2*space,color=scalarMap.to_rgba(daytemps[15]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelH41fid[:,0],FmodelH41fid[:,1]*10**6.-np.median(H41[:,1])+1*space,color=scalarMap.to_rgba(daytemps[3]),linestyle='-',linewidth=2,zorder=1)
ax1.plot(FmodelK7fid[:,0],FmodelK7fid[:,1]*10**6.-np.median(K7[:,1]),color=scalarMap.to_rgba(daytemps[6]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW74fid[:,0],FmodelW74fid[:,1]*10**6.-np.median(W74[:,1])+8*space,color=scalarMap.to_rgba(daytemps[14]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW79fid[:,0],FmodelW79fid[:,1]*10**6.-np.median(W79[:,1])+7*space,color=scalarMap.to_rgba(daytemps[16]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW4fid[:,0],FmodelW4fid[:,1]*10**6.-np.median(W4[:,1])+6*space,color=scalarMap.to_rgba(daytemps[9]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelH32fid[:,0],FmodelH32fid[:,1]*10**6.-np.median(H32[:,1])+5*space,color=scalarMap.to_rgba(daytemps[2]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelT3fid[:,0],FmodelT3fid[:,1]*10**6.-np.median(T3[:,1])+4*space,color=scalarMap.to_rgba(daytemps[8]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelC2fid[:,0],FmodelC2fid[:,1]*10**6.-np.median(C2[:,1])+3*space,color=scalarMap.to_rgba(daytemps[0]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelW43fid[:,0],FmodelW43fid[:,1]*10**6.-np.median(W43[:,1])+2*space,color=scalarMap.to_rgba(daytemps[13]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelHD209fid[:,0],(FmodelHD209fid[:,1]*10**6.-np.median(HD209[:,1]))*2.+1*space,color=scalarMap.to_rgba(daytemps[5]),linestyle='-',linewidth=2,zorder=1)
ax2.plot(FmodelHD189fid[:,0],(FmodelHD189fid[:,1]*10**6.-np.median(HD189[:,1]))*2.,color=scalarMap.to_rgba(daytemps[4]),linestyle='-',linewidth=2,zorder=1)

ax1.set_xlim(1.1,1.66)
ax1.set_ylim(-250,9300)
ax1.set_xlabel('Wavelength [micron]',fontsize=20)
ax1.set_ylabel('Fp/Fs [ppm]',fontsize=20)
ax1.tick_params(labelsize=15,axis="both",top=True,width=2,length=8,direction='in')
ax1.set_yticks(ticks=[-152,-2,148,-153+1*space,-3+1*space,147+1*space,\
	-221+2*space,-71+2*space,79+2*space,-186+3*space,-36+3*space,114+3*space,\
	-170+4*space,-20+4*space,130+4*space,-310+5*space,-60+5*space,190+5*space,\
	-178+6*space,-28+6*space,121+6*space,-242+7*space,-42+7*space,158+7*space,\
	-150+8*space,8*space,150+8*space,-187+9*space,13+9*space,213+9*space])
ax1.set_yticklabels(labels=[str("{:.0f}".format(np.median(K7[:,1])-152)),str("{:.0f}".format(np.median(K7[:,1])-2)),str("{:.0f}".format(np.median(K7[:,1])+148)),\
	str("{:.0f}".format(np.median(H41[:,1])-153)),str("{:.0f}".format(np.median(H41[:,1])-3)),str("{:.0f}".format(np.median(H41[:,1])+147)),\
	str("{:.0f}".format(np.median(W76[:,1])-221)),str("{:.0f}".format(np.median(W76[:,1])-71)),str("{:.0f}".format(np.median(W76[:,1])+79)),\
	str("{:.0f}".format(np.median(W121[:,1])-186)),str("{:.0f}".format(np.median(W121[:,1])-36)),str("{:.0f}".format(np.median(W121[:,1])+114)),\
	str("{:.0f}".format(np.median(H7[:,1])-170)),str("{:.0f}".format(np.median(H7[:,1])-20)),str("{:.0f}".format(np.median(H7[:,1])+130)),\
	str("{:.0f}".format(np.median(W12[:,1])-310)),str("{:.0f}".format(np.median(W12[:,1])-60)),str("{:.0f}".format(np.median(W12[:,1])+190)),\
	str("{:.0f}".format(np.median(W18[:,1])-178)),str("{:.0f}".format(np.median(W18[:,1])-28)),str("{:.0f}".format(np.median(W18[:,1])+121)),\
	str("{:.0f}".format(np.median(W103[:,1])-242)),str("{:.0f}".format(np.median(W103[:,1])-42)),str("{:.0f}".format(np.median(W103[:,1])+158)),\
	str("{:.0f}".format(np.median(W33[:,1])-150)),str("{:.0f}".format(np.median(W33[:,1]))),str("{:.0f}".format(np.median(W33[:,1])+150)),\
	str("{:.0f}".format(np.median(Kep13[:,1])-187)),str("{:.0f}".format(np.median(Kep13[:,1])+13)),str("{:.0f}".format(np.median(Kep13[:,1])+213))])

ax2.set_xlim(1.1,1.85)
ax2.set_ylim(-400,8150)
ax2.set_xlabel('Wavelength [micron]',fontsize=20)
ax2.tick_params(labelsize=15,axis="both",top=True,width=2,length=8,direction='in')
ax2.set_yticks(ticks=[-188,12,212,-152+1*space,-2+1*space,148+1*space,\
	-139+2*space,11+2*space,161+2*space,-142+3*space,8+3*space,158+3*space,\
	-150+4*space,4*space,150+4*space,-169+5*space,-19+5*space,131+5*space,\
	-220+6*space,-20+6*space,180+6*space,-177+7*space,-27+7*space,123+7*space,\
	-117+8*space,33+8*space,183+8*space])
ax2.set_yticklabels(labels=[0,100,200,\
	25,100,175,\
	str("{:.0f}".format(np.median(W43[:,1])-139)),str("{:.0f}".format(np.median(W43[:,1])+11)),str("{:.0f}".format(np.median(W43[:,1])+161)),\
	str("{:.0f}".format(np.median(C2[:,1])-142)),str("{:.0f}".format(np.median(C2[:,1])+8)),str("{:.0f}".format(np.median(C2[:,1])+158)),\
	str("{:.0f}".format(np.median(T3[:,1])-150)),str("{:.0f}".format(np.median(T3[:,1])-0)),str("{:.0f}".format(np.median(T3[:,1])+150)),\
	str("{:.0f}".format(np.median(H32[:,1])-169)),str("{:.0f}".format(np.median(H32[:,1])-19)),str("{:.0f}".format(np.median(H32[:,1])+131)),\
	str("{:.0f}".format(np.median(W4[:,1])-220)),str("{:.0f}".format(np.median(W4[:,1])-20)),str("{:.0f}".format(np.median(W4[:,1])+180)),\
	str("{:.0f}".format(np.median(W79[:,1])-176.7)),str("{:.0f}".format(np.median(W79[:,1])-27)),str("{:.0f}".format(np.median(W79[:,1])+123)),\
	str("{:.0f}".format(np.median(W74[:,1])-117)),str("{:.0f}".format(np.median(W74[:,1])+33)),str("{:.0f}".format(np.median(W74[:,1])+183))])

a = plt.axes([0.91,0.37,0.01,0.37], frameon=False)#left, bottom, width, height
a.yaxis.set_visible(False)
a.xaxis.set_visible(False)
a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.inferno, aspect='auto', visible=False)
cbar=plt.colorbar(a, fraction=3.0)
cbar.ax.tick_params(labelsize=15,width=2,length=6)
cbar.set_label('Dayside Temperature [K]',fontsize=15)

plt.tight_layout()
plt.savefig('allspectra_bestfit.png')
plt.show()





