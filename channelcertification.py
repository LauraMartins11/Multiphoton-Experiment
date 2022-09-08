# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:42:34 2022

@author: Simon
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import os as os
import glob as glob
import random as rnd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as tick
import matplotlib.colors as col

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({
    "font.family": "serif",
    "font.sans-serif": ["Palatino"]})

sourcefile = "C:/Users/Simon/Documents/Travail/ChannelCertification/"
os.chdir(sourcefile)


#%%

def truncate_colormap(cmapIn='RdBu', vals=[0,0.5,1], n=[50,50]):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''   
    arrayCol = np.array([])
    for k in range(len(n)):
        arrayCol = np.concatenate((arrayCol,np.arange(vals[k],vals[k+1],(vals[k+1]-vals[k])/n[k])))
    cmapIn = plt.get_cmap(cmapIn)
    new_cmap = col.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=vals[0], b=vals[-1]),
        cmapIn(arrayCol))
    return new_cmap

def delta(x,R,K):
    return 1/(K+1) + np.sqrt(2*x/(R*(K+1)))

def confStat1sDI(x):
    return (1-np.exp(-x))*(1-2*np.exp(-x))**2

def yFromConf(c):
    return np.log(2/(1-np.sqrt(c)))

def errState(x,R,K):
    return np.arccos((1-3*delta(x,R,K))/(1-delta(x,R,K)))

def xFromConf(c):
    return -np.log(1-c)

def minFidAnuSteering(x,eps,K):
    return 1-1.26*(8*np.sqrt(x/K)+eps/2 + (eps+8/K)/(2+1/K))

def minFidAnuCHSH(x,eps,K):
    return 1-0.9*(16*np.sqrt(2*x/K)+3*eps/4 + (eps+(4+2*np.sqrt(2))/K)/(4+4/K))

def FinIIDDI(x,delta,M):
    return 1-1.19*(8*np.sqrt(2*x/M)+delta)

def cOutCHSHDI(x,eps,K):
    return np.sqrt(1.19 * (16*np.sqrt(2*x/K)+3*eps/4 + (eps+(4+2*np.sqrt(2))/K)/(4+4/K)))

def minFidTotCHSH(x,Fin,eps,K,R):
    Cin = np.sqrt(1-Fin)
    return 1-4*(np.sin(np.arcsin(Cin/(R*(1-delta(x,R,K))))+errState(x,R,K)+np.arcsin(np.sqrt(1-minFidAnuCHSH(x,eps,K)))))**2

def minFidTotSteering(x,Fin,eps,K,R):
    Cin = np.sqrt(1-Fin)
    return 1-4*(np.sin(np.arcsin(Cin/(R*(1-delta(x,R,K))))+errState(x,R,K)+np.arcsin(np.sqrt(1-minFidAnuSteering(x,eps,K)))))**2 

def FidTotIIDDI(x,R,K,M,eps,eta):
    Cin = np.sqrt(1-FinIIDDI(x,eta,M))
    t = R*(1-delta(x,R,K))
    return 1-4*(np.sin(np.arcsin(Cin/t)+np.arcsin(cOutCHSHDI(x,eps,K))+errState(x,R,K)))**2

def minFidTotCHSHbis(x,Fin,eps,K,R,eff):
    Cin = np.sqrt(1-Fin)
    return 1-4*(np.sin(np.arcsin(Cin*eff/(R*(1-delta(x,R,K))))+errState(x,R,K)+np.arcsin(np.sqrt(1-minFidAnuCHSH(x,eps,K)))))**2

def minFidTotSteeringbis(x,Fin,eps,K,R,eff):
    Cin = np.sqrt(1-Fin)
    t = R*(1-delta(x,R,K))/eff    
    return 1-4*(np.sin(np.arcsin(Cin/t)+errState(x,R,K)+np.arcsin(np.sqrt(1-minFidAnuSteering(x,eps,K)))))**2 

def dFdFin(x,Fin,eps,K,R,eff):
    Cin = np.sqrt(1-Fin)
    t = R*(1-delta(x,R,K))/eff
    rad = np.arcsin(Cin/t)+errState(x,R,K)+np.arcsin(np.sqrt(1-minFidAnuSteering(x,eps,K)))
    return 4*np.cos(rad)*np.sin(rad)/np.sqrt((t*Cin)**2 - Cin**4)
    
def FidTotIIDDIbis(x,R,K,M,eps,eta,eff):
    Cin = np.sqrt(1-FinIIDDI(x,eta,M))
    t = R*(1-delta(x,R,K))/eff
    return 1-4*(np.sin(np.arcsin(Cin/t)+np.arcsin(cOutCHSHDI(x,eps,K))+errState(x,R,K)))**2

def dFdEps(x,Fin,eps,K,R,eff):
    Cin = np.sqrt(1-Fin)
    t = R*(1-delta(x,R,K))/eff
    rad = np.arcsin(Cin/t)+errState(x,R,K)+np.arcsin(np.sqrt(1-minFidAnuSteering(x,eps,K)))
    errfx = 0.5 + 1/(2+1/K)
    return 4*np.cos(rad)*np.sin(rad)*errfx*1.26/np.sqrt((1-minFidAnuSteering(x,eps,K))*minFidAnuSteering(x,eps,K))
#%%

def visibility(data):
    return np.abs((data[4]-data[5]-data[6]+data[7])/np.sum(data[4:8]))
                  
def getData(filename):
    text_file = open(filename,"r")
    temp = text_file.read().splitlines()
    data = np.array([list(map(int,line.split(" ")[:-1])) for line in temp],dtype='int64')
    return data

def genRndMap(p,N):
    N1 = int(np.round(p*N))
    li = [1]*N1+[0]*(N-N1)
    rnd.shuffle(li)
    return np.transpose(np.array([li]))

class protocolHonest():
    
    def __init__(self,filename,input_fid,correction="None"):
        self.filename = "HonestData/"+filename
        self.input_fid = input_fid
        self.correction = correction
        self.x_data, self.z_data = self.genData()

    def genData(self):
        x_data = getData(self.filename+"AllTimeBinsx.txt")
        z_data = getData(self.filename+"AllTimeBinsz.txt")
        self.time = np.shape(x_data)[0]+np.shape(z_data)[0]
        if self.correction == "None": 
            x_data_sum = np.sum(x_data,0)
            z_data_sum = np.sum(z_data,0)
        if self.correction == "Alice":
            x_data_sum = (np.sum(x_data,0)//self.get_rel_eff_Alice()).astype(np.int64)
            z_data_sum = (np.sum(z_data,0)//self.get_rel_eff_Alice()).astype(np.int64)
        if self.correction == "All":
            x_data_sum = (np.sum(x_data,0)//self.get_rel_eff()).astype(np.int64)
            z_data_sum = (np.sum(z_data,0)//self.get_rel_eff()).astype(np.int64)
        return x_data_sum, z_data_sum
    
    def avg_rate(self):
        return (self.x_data[0]+self.x_data[1]+self.z_data[0]+self.z_data[1])/self.time
    
    def getTomoData(self,io,basis):
        """return an array with the tomography data from one measurement basis.
        io = "In" or "Out" 
        basis = measurement basis"""
        filename = glob.glob(self.filename+"StateTomography*"+io+"*/StateTomo/*_"+basis+"_*.txt")[0]
        return getData(filename)

    def get_rel_eff_Alice(self):
        rel_eff = 0
        for basis in ["zz","az"]:
            rel_eff += self.getTomoData("Out",basis).astype(float)
        rel_eff[0,:2] = rel_eff[0,:2]/rel_eff[0,:2].min()
        rel_eff[0,2:4] = np.array([1.0,1.0])
        rel_eff[0,4:6] = np.ones(2)*rel_eff[0,0]
        rel_eff[0,6:8] = np.ones(2)*rel_eff[0,1]
        return rel_eff[0]
    
    def get_rel_eff(self):
        rel_eff = 0
        for basis in ["zz","za","az","aa"]:
            rel_eff += self.getTomoData("Out",basis).astype(float)
        rel_eff[0,:2] = rel_eff[0,:2]/rel_eff[0,:2].min()
        rel_eff[0,2:4] = rel_eff[0,2:4]/rel_eff[0,2:4].min()
        rel_eff[0,4:8] = rel_eff[0,4:8]/rel_eff[0,4:8].min()
        return rel_eff[0]
            
    def xVis(self):
        return visibility(self.x_data)
    
    def zVis(self):
        return visibility(self.z_data)
    
    def avgE(self):
        return self.xVis() + self.zVis()
    
    def eps(self):
        return 2-self.avgE()
    
    def NtransmittedStates(self):
        tot_data = self.x_data+self.z_data
        return np.sum(tot_data[4:8])
        
    def NemittedStates(self):
        tot_data = self.x_data + self.z_data
        return tot_data[0] + tot_data[1]
    
    def transmissionRatio(self):
        return self.NtransmittedStates()/self.NemittedStates()
    
    def minFidTotSteeringbis(self,eff):
        return minFidTotSteeringbis(7,self.input_fid,self.eps(),self.NtransmittedStates(),self.transmissionRatio(),eff)
    
    def dFdEps(self,eff):
        return dFdEps(7,self.input_fid,self.eps(),self.NtransmittedStates(),self.transmissionRatio(),eff)
    
    def dFdFin(self,eff):
        return dFdFin(7,self.input_fid,self.eps(),self.NtransmittedStates(),self.transmissionRatio(),eff)
        
class protocolMalicious(protocolHonest):
    
    def __init__(self,filename,flip_proba,input_fid,correction = "None"):
        """flip_proba: tuple (pX,pZ), pX= proba phase flip, pZ = proba bit flip"""
        self.filename = "MaliciousChannelData/"+filename
        self.flip_proba = flip_proba
        self.input_fid = input_fid
        self.rel_eff = self.get_rel_eff()
        self.correction = correction
        self.x_data, self.z_data = self.genData()
        
    def genData(self):
        x_data = getData(self.filename+"AllTimeBinsx.txt")
        z_data = getData(self.filename+"AllTimeBinsz.txt")
        xf_data = getData(self.filename+"AllTimeBinsm.txt")
        zf_data = getData(self.filename+"AllTimeBinsn.txt")
        rnd.shuffle(x_data)
        rnd.shuffle(z_data)
        rnd.shuffle(xf_data)
        rnd.shuffle(zf_data)
        mapX = genRndMap(self.flip_proba[0],10000)
        mapZ = genRndMap(self.flip_proba[1],10000)
        x_malData = (1-mapX)*x_data[:10000,:]+mapX*xf_data[:10000,:]
        z_malData = (1-mapZ)*z_data[:10000,:]+mapZ*zf_data[:10000,:]
        if self.correction == "Alice":
            x_malData = (np.sum(x_malData,0)//self.get_rel_eff_Alice()).astype(np.int64)
            z_malData = (np.sum(z_malData,0)//self.get_rel_eff_Alice()).astype(np.int64)
        if self.correction == "All":
            x_malData = (np.sum(x_malData,0)//self.get_rel_eff()).astype(np.int64)
            z_malData = (np.sum(z_malData,0)//self.get_rel_eff()).astype(np.int64)
        return x_malData, z_malData
    
    def avg_rate(self):
        return  (self.x_data[0]+self.x_data[1]+self.z_data[0]+self.z_data[1])/(0.2*20000)
        

#%% honest data
errdrift = 0.0002
errTomo = 0.001
inputFidelities = np.array([0.99283,0.99365,0.99126,0.99061,0.98308,0.99224,0.99183,0.99137,0.99276,0.99292,0.99239,0.99257,0.99032])
errFin = np.array([0.00052,0.00045,0.00043,0.0004,0.00068,0.0005,0.00046,0.00047,0.00051,0.00048,0.00045,0.00049,0.00045])
errFin = np.sqrt(errFin**2 + errdrift**2 + errTomo**2)
outputFidelities = np.array([1,0.9909,0.99557,0.99396,0.9852,0.99562,0.99596,0.99421,0.99574,0.99621,0.99779,0.99531,0.99188])
errFout = np.sqrt(np.array([0,0.0004,0.00052,0.00075,0.00055,0.00049,0.00169,0.00199,0.00178,0.00148,0.00196,0.00119,0.0014])**2+errTomo**2)
errFinOut = np.sqrt(errFout**2+errFin**2)
avgFin = np.sum(inputFidelities/errFin**2)/np.sum(errFin**(-2))
errAvgFin = np.sum(errFin**(-2))**(-1/2)
#inputFidelities -= errFin
Rlin = np.linspace(0.2,0.484,200)
foldernames = [name+"/" for name in os.listdir("HonestData/")]
eff = np.array([0.474,0.6,0.8,1])
labels = np.array([r"$0.526$",r"$0.4$",r"$0.2$",r"$0$"])
colors = ['b','g','y','r','cyan','pink']
markers = ["v",".","*","^"]
n_point = len(foldernames)
fig3 = plt.figure(figsize=(7.5,4.3))
eps = []
eps_corr = []
rate = []
plt.plot([0.2,0.484],[0.5,0.5],linestyle="--",color='purple',linewidth=1)
for ieff in range(4):
    minFid = []
    errMinFid = []
    if ieff == 0:
        R = []
    for i in range(n_point):
        protocol = protocolHonest(foldernames[i],inputFidelities[i],"Alice")
        protocol_corr = protocolHonest(foldernames[i],inputFidelities[i],"All")
        if ieff == 0:
            R.append(protocol.transmissionRatio())
            eps.append(protocol.eps())
            eps_corr.append(protocol_corr.eps())
            rate.append(protocol.avg_rate())
            print(protocol.transmissionRatio())
        minFid.append(max(protocol.minFidTotSteeringbis(eff[ieff]),0))
        errMinFid.append(np.sqrt((protocol.dFdFin(eff[ieff])*errFin[i])**2+(protocol.dFdEps(eff[ieff])*(eps[i]-eps_corr[i]))**2))
    if ieff == 3:
        errMinFid[2]=0
        errMinFid[4]=0
        errMinFid[12]=0
    if ieff == 2:
        errMinFid[4]=0
    if ieff == 0:
        print(minFid[0],errMinFid[0])
    plt.errorbar(R,minFid,yerr= np.asarray(errMinFid,dtype=float),fmt = markers[ieff],color=colors[ieff],label=labels[ieff],markersize=6)
plt.errorbar(R,outputFidelities,yerr= np.asarray(errFinOut,dtype=float),fmt = "x",color='purple',markersize=6)
plt.plot([0.312,0.28],[0.99,0.87],color='purple',linewidth=0.5)
plt.text(0.26,0.82,r"$F(\Phi_i,\Phi_o)$"+" "+r"$\it{via}$"+" "+r"$\it{tomography}$",fontsize=14,color='purple')
plt.text(0.188,0.485,r"$0.5$",fontsize=12,color='purple')
for l in range(4):
    minFidLin = minFidTotSteeringbis(7,np.mean(inputFidelities),np.mean(np.array(eps)),10**9,Rlin,eff[l])
    minFidLin[minFidLin<0]=0
    plt.plot(Rlin,minFidLin,color=colors[l])#,label=r"${0}\%$".format(np.int(eff[l]*100)))
plt.plot([0.2,0.484],[np.mean(outputFidelities),np.mean(outputFidelities)],color='purple')
plt.xlim(0.2,0.484)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r"Minimum Certified Fidelity",fontsize=16)
plt.xlabel(r"Measured Heralding Efficiency $\eta_s$",fontsize=16)
plt.grid()
leg = plt.legend(loc="lower right",bbox_to_anchor=(1,0.0),
                 ncol=2,columnspacing=1,handletextpad=0.1, shadow=True, title=r"Trusted Loss $\lambda_c$",fancybox=True,fontsize=16)
leg.get_title().set_fontsize(14)
plt.tight_layout(0.2)

#%%

fig, ax1 = plt.subplots(figsize=(7.5,2.5))
ax1.errorbar(R,inputFidelities,yerr=np.asarray(errFin,dtype=float),fmt='x',color='blue',markersize=6)
ax1.set_ylim(0.97,1)
ax1.tick_params(axis='y',labelsize=14,labelcolor='blue')
ax1.tick_params(axis='x',labelsize=14)
ax1.set_xlabel(r"Measured Heralding Efficiency $\eta_s$",fontsize=16)
ax1.set_yticks([0.97,0.98,0.99,1])
ax1.set_ylabel(r"$F^i$",fontsize=18,color = 'blue')

ax2 = ax1.twinx()
ax2.errorbar(R,2-np.array(eps),yerr=np.asarray(np.abs(np.array(eps)-np.array(eps_corr)),dtype=float),fmt='*',color='red',markersize=8)
ax2.set_ylim(1.97,2)
ax2.tick_params(axis='y',labelsize=14,labelcolor='red')
ax2.set_yticks([1.97,1.98,1.99,2])
ax2.set_ylabel(r"$2-\epsilon$",fontsize=18,color = 'red')

plt.tight_layout(0.3)



#%% Malicious data PHASE FLIP
    
foldernameBF = "BPF202205262200/"
foldernamePF = "BPF202205262200/"
foldernameBPF = "BPF202205262200/"
cheatProbas = np.linspace(0,0.06,100)
liEPhaseFlip = []
liEBitFlip = []
liEBitPhaseFlip = []
liEPhaseFlip_corr = []
liEBitFlip_corr = []
liEBitPhaseFlip_corr = []
errPF = []
errBF = []
errBPF = []
x = 7
Fin = 0.99115
errFinM = 0.0005
errFin
K = 10**9
R = 0.47

for p in cheatProbas:
    print(p)
    protocolBF = protocolMalicious(foldernameBF,(0,p),Fin,"Alice")
    protocolPF = protocolMalicious(foldernamePF,(p,0),Fin,"Alice")
    protocolBPF = protocolMalicious(foldernameBPF,(p,p),Fin,"Alice")
    protocolBF_corr = protocolMalicious(foldernameBF,(0,p),Fin,"All")
    protocolPF_corr = protocolMalicious(foldernamePF,(p,0),Fin,"All")
    protocolBPF_corr = protocolMalicious(foldernameBPF,(p,p),Fin,"All")
    liEBitFlip.append(protocolBF.avgE())
    liEPhaseFlip.append(protocolPF.avgE())
    liEBitPhaseFlip.append(protocolBPF.avgE())
    liEBitFlip_corr.append(protocolBF_corr.avgE())
    liEPhaseFlip_corr.append(protocolPF_corr.avgE())
    liEBitPhaseFlip_corr.append(protocolBPF_corr.avgE())
    errBF.append(np.sqrt((protocolBF.dFdEps(0.47)*errFinM)**2+(protocolBF.dFdEps(0.47)*(protocolBF.eps()-protocolBF_corr.eps()))**2))
    errPF.append(np.sqrt((protocolPF.dFdEps(0.47)*errFinM)**2+(protocolPF.dFdEps(0.47)*(protocolPF.eps()-protocolPF_corr.eps()))**2))
    errBPF.append(np.sqrt((protocolBPF.dFdEps(0.47)*errFinM)**2+(protocolBPF.dFdEps(0.47)*(protocolBPF.eps()-protocolBPF_corr.eps()))**2))
EPhaseFlip = np.array(liEPhaseFlip)
EBitFlip = np.array(liEBitFlip)
EBitPhaseFlip = np.array(liEBitPhaseFlip)
errBF = np.array(errBF)
errPF = np.array(errPF)
errBPF = np.array(errBPF)

#%% Malicious data 

errdrift = 0.0002
errTomo = 0.001

x = 7
Fin = 0.99115
K = 10**9
R = 0.47
foldername = "BPF202205262200/"
ProbaBF = np.linspace(0,0.05,71)
ProbaPF = np.linspace(0,0.05,71)
pB,pP = np.meshgrid(ProbaBF,ProbaPF)
Etab = np.zeros((71,71))
rates = np.zeros((71,71))
for iB in range(71):
    for iP in range(71):
        print(iB,iP)
        protocol = protocolMalicious(foldername,(ProbaPF[iP],ProbaBF[iB]),Fin,"Alice")
        rates[iP,iB] = protocol.avg_rate()
        Etab[iP,iB] = protocol.avgE()
print(np.mean(np.mean(rates,1)))

#%% save
        
np.save('MaliciousChannelData/MaliciousChannelData2.npy', Etab)
#%%

x = 7
Fin = 0.99115
K = 10**9
R = 0.47
foldername = "BPF202205262200/"
ProbaBF = np.linspace(0,0.05,71)
ProbaPF = np.linspace(0,0.05,71)
pB,pP = np.meshgrid(ProbaBF,ProbaPF)
Etabl = np.load('MaliciousChannelData/MaliciousChannelData.npy')
dp = 0.05/71

from mpl_toolkits.mplot3d import Axes3D
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
###patch end###

minFid = np.maximum(minFidTotSteeringbis(x,Fin,2-Etabl,K,0.47,0.47),np.zeros((71,71)))
dFdPB = (minFid[1:,:]-minFid[:70,:])/dp
dFdPP = (minFid[:,1:]-minFid[:,:70])/dp
fig, ax = plt.subplots(figsize=(8,5.5),subplot_kw={"projection": "3d"})
N = 71
cmap_mod = truncate_colormap(cmapIn='RdBu',vals=[0.1,0.35,0.5,0.7,0.95],n=[67,3,3,27])
surf= ax.plot_surface(pB[:N,:N], pP[:N,:N], minFid[:N,:N], cmap=cmap_mod,
                       linewidth=0.2, antialiased=True,rstride=1, cstride=1,edgecolor='black')
ax.view_init(elev=30., azim=240)
ax.set_zlim3d(0,0.73)    
ax.set_xlim(0,0.05)
ax.set_ylim(0,0.05)

#ax.set_ylim3d(fy[l-1],fy[0])
#ax.set_xlim3d(fx[0],fx[l-1])
plt.xticks(fontsize=12,rotation=-30, ha="left")
plt.yticks([0.01,0.02,0.03,0.04,0.05],fontsize=12,rotation=30, ha="right")
ax.tick_params(axis='x', which='major', pad=-6)
ax.tick_params(axis='y', which='major', pad=-8)
ax.zaxis.set_tick_params(labelsize=12,pad=-1)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.grid()
plt.xlabel(r"Bit Flip Probability $p$",fontsize=14,labelpad=10)
plt.ylabel(r"Phase Flip Probability $q$",fontsize=14,labelpad=8)
ax.set_zlabel(r"Minimum Certified Fidelity",fontsize=14,labelpad=0,rotation=90)
# A StrMethodFormatter is used automatically
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.show()
plt.tight_layout(0)

#%%
fig2 = plt.figure(figsize=(7,4))
plt.plot(cheatProbas,EPhaseFlip,color='b',label=r"Phase Flip")
plt.plot(cheatProbas,EBitFlip,color='r',label=r"Bit Flip")
plt.plot(cheatProbas,EBitPhaseFlip,color='g',label=r"Bit + Phase Flip")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(r"$\langle E \rangle$",fontsize=24)
plt.xlabel(r"Cheating Probability",fontsize=14)
leg = plt.legend(loc="lower left",
                 ncol=1, shadow=True, title=r"Type of Attack",fancybox=True,fontsize=14)
leg.get_title().set_fontsize(14)
plt.tight_layout(0.2)

#%%

fig3 = plt.figure(figsize=(7,4))
plt.plot(cheatProbas,np.maximum(minFidTotSteeringbis(x,Fin,2-EPhaseFlip,K,0.47,0.47),np.zeros(100)),color='b',label=r"Phase Flip")
plt.fill_between(cheatProbas,np.maximum(minFidTotSteeringbis(x,Fin,2-EPhaseFlip,K,0.47,0.47)-errPF,np.zeros(100)),np.maximum(minFidTotSteeringbis(x,Fin,2-EPhaseFlip,K,0.47,0.47)+errPF,np.zeros(100)), color='b', alpha=.1)
plt.plot(cheatProbas,np.maximum(minFidTotSteeringbis(x,Fin,2-EBitFlip,K,0.47,0.47),np.zeros(100)),color='r',label=r"Bit Flip",linestyle="-.")
plt.plot(cheatProbas,np.maximum(minFidTotSteeringbis(x,Fin,2-EBitPhaseFlip,K,0.47,0.47),np.zeros(100)),color='g',label=r"Bit + Phase Flip")
plt.fill_between(cheatProbas,np.maximum(minFidTotSteeringbis(x,Fin,2-EBitPhaseFlip,K,0.47,0.47)-errBPF,np.zeros(100)),np.maximum(minFidTotSteeringbis(x,Fin,2-EBitPhaseFlip,K,0.47,0.47)+errBPF,np.zeros(100)), color='g', alpha=.1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(r"$F_{min}$",fontsize=24)
plt.xlabel(r"Cheating Probability",fontsize=16)
leg = plt.legend(loc="lower right",bbox_to_anchor=(1.005,0.5),
                 ncol=1, shadow=True, title=r"Type of Attack",fancybox=True,fontsize=16)
leg.get_title().set_fontsize(16)
plt.grid()
plt.tight_layout(0.2)
    
#%% Malicious data BIT FLIP


for p in cheatProbas:
    protocol = protocolMalicious(foldername,p,(0,1))
    liEBitFlip.append(protocol.avgE())

#%% Malicious data PHASE + BIT FLIP
    
foldername = "BPF202205262200/"
cheatProbas = np.linspace(0,0.5,50)
for p in cheatProbas:
    protocol = protocolMalicious(foldername,p,(1,1))
    print(protocol.avgE())
    
   
#%%  stability
errTomo = 0.001
time = np.array([0.,16.08333333,32.18333333,48.26666667,64.36666667,80.45
        ,96.53333333,112.63333333,128.71666667,144.81666667,160.9
        ,177.,193.08333333,209.16666667,225.26666667,241.35,257.45
        ,273.53333333,289.63333333,305.71666667,321.8,337.9
        ,353.98333333,370.06666667,386.16666667,466.68333333,482.76666667])
n_data = np.shape(time)[0]    
time_err = np.ones(n_data)*15/2
time_err[24] = 79.5/2

time = time + time_err



fid_avg = np.array([0.99073,0.99044,0.9907,0.9909,0.99102,0.99108,0.99109,0.9912
                    ,0.99115,0.99136,0.99116,0.9909,0.99118,0.99109,0.9911
                    ,0.99124,0.99093,0.99131,0.99129,0.99116,0.99121,0.9912
                    ,0.99128,0.99147,0.99167,0.99156,0.99164])

fid_err = np.array([0.00041,0.00043,0.00046,0.00043,0.0004,0.00045,0.00044
                    ,0.00041,0.00042,0.00042,0.00044,0.00043,0.00044,0.0004
                    ,0.00043,0.00044,0.00043,0.00043,0.00042,0.00044,0.00041
                    ,0.00044,0.00046,0.00044,0.00041,0.00047,0.00042])
    
fid_err= np.sqrt(fid_err**2 + errTomo**2 )

err_tot= np.sum(fid_err**(-2))**(-1/2)
    
tot_err = np.sqrt(fid_avg.std()**2+np.sum(fid_err**2)/n_data**2)
orange =   "#ff8400ff"
fig, ax = plt.subplots(figsize=(9,4))
coeffs = np.polyfit(time,fid_avg,1)
plt.errorbar(time/60, fid_avg, xerr = time_err/60, yerr=fid_err, fmt='x', label='data',markersize=8,color="blue")
plt.plot(time/60,coeffs[1]+time*coeffs[0],color=orange)
ax.set_ylim(.98,1)
plt.xticks(fontsize=18)
plt.yticks([.98,.985,.99,.995,1],[r"$98.0\%$",r"$98.5\%$",r"$99.0\%$",r"$99.5\%$",r"$100\%$"],fontsize=18)
plt.ylabel('Fidelity',fontsize=20)
plt.xlabel('Time (hours)',fontsize=20)
plt.tight_layout(0.2)
plt.show()
print(coeffs[0]*60)
print(tot_err)
plt.text(4,0.9835,r"$F^i(t) = 0.9908 + t\cdot 10^{-4} h^{-1}$",fontsize=20,color=orange)
plt.plot([6.7,5.5],[0.9913,0.985],color=orange,linewidth=0.5)

#%% Quantum State 

state = np.array([[0.50898239-5.55111512e-17j,0.00207513-1.91041573e-03j
                   ,0.00419936+2.80613490e-03j,0.49577659+1.52700794e-07j]
    ,[0.00207513+1.91041573e-03j,0.00098563-2.54448697e-18j
      ,-0.00109879-7.25332571e-04j,0.0041944 +1.27281249e-03j]
    ,[0.00419936-2.80613490e-03j,-0.00109879+7.25332571e-04j
      ,0.00189361+3.25260652e-18j,0.00207019-3.77293219e-04j]
    ,[0.49577659-1.52700794e-07j,0.0041944 -1.27281249e-03j
      ,0.00207019+3.77293219e-04j,0.48813837+2.77555756e-17j]])


colors = np.array([0,0.029,0.045,0.07])
x = np.array([0.6,1.6,2.6,3.6])
y = np.array([0.6,1.6,2.6,3.6])
xx,yy = np.meshgrid(x,y)
bottom = np.zeros_like(state)
depth = width = 0.8   

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(4):
    ax.bar3d(xx[i,:],yy[i,:],bottom[i,:],width,depth,state[i,:],shade='auto',color=plt.cm.prism(colors),alpha = 0.5)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.tick_params(axis='x', which='major', pad=-2)
ax.tick_params(axis='y', which='major', pad=-2)
ax.zaxis.set_tick_params(labelsize=14,pad=1)
plt.xticks([1,2,3,4],[r"$\left|\, HH\, \right\rangle$",r"$\left|\, HV\, \right\rangle$",r"$\left|\, VH\, \right\rangle$",r"$\left|\, VV\, \right\rangle$"],fontsize=16)
plt.yticks([1,2,3,4],[r"$\left\langle\, HH\, \right|$",r"$\left\langle\, HV\, \right|$",r"$\left\langle\, VH\, \right|$",r"$\left\langle\, VV\, \right|$"],fontsize=16)

#%% Fusion of data files in protocols
    
def concDataProtocolHonest(basis):
    """fusing all data files together in a big txt file, for the honest protocols"""
    foldernames = ["HonestData/"+name+"/" for name in os.listdir("HonestData/")]
    for folder in foldernames:
        filenames = sorted(glob.glob(folder+"ChannelCertifST/*_"+basis+"_*.txt"),key=os.path.getmtime)
        with open(folder+'AllTimeBins'+basis+'.txt', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read())
                
def concDataProtocolMalicious(basis):
    """fusing all data files together in a big txt file, for the malicious protocol"""
    filenames = sorted(glob.glob("MaliciousChannelData/BPF202205262200/ChannelCertifST/*_"+basis+"_*.txt"),key=os.path.getmtime)
    with open("MaliciousChannelData/BPF202205262200/AllTimeBins"+basis+".txt", 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())     
                
def concDataCHSH(basis):
    """fusing all data files together in a big txt file, for CHSH"""
    filenames = sorted(glob.glob("CHSH/CHSHviolation/*_"+basis+"_*.txt"),key=os.path.getmtime)
    with open("CHSH/AllTimeBins"+basis+".txt", 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())     
                
                
