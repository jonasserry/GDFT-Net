
from Core import GDFT_Data
from Core import GDFT_Net
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

print("Tester Version: 1.00")

class GDFT_Net_Tester():

    def __init__(self,Tester_Path,Net_Path):
        
        self.Path=Tester_Path
        self.Net_Path = Net_Path
        self.Net=None

        self.errors = defaultdict(list)
        self.standard_dev_delays = None


    def load_Net(self):
        self.Net = GDFT_Net.load_GDFT_Net(self.Net_Path)
        self.Net.load_models()


    def run_RMSE_Testing(self,numImages=None,SNRs=None,DS=None):
        corr = []
        i=0
        if DS != None:
            SNRs = DS.SNRs
            
        for SNR in SNRs:
            if DS == None:
                raw_images,_,labels_1D =  GDFT_Data.Create_Images(numImages, self.Net.numSteps, self.Net.dimensions, self.Net.t0, self.Net.wavenumberRange, self.Net.numChan, self.Net.numCoherent, self.Net.numIncoherent, SNR,numSteps_simulated=1024*1024,print_flag=False)
            else:
                raw_images,_,labels_1D =  DS.get_Data(with_SNR=SNR)
            prediction = self.Net.process_Images(raw_images,verbose=0)[1]*self.Net.numChan*2-self.Net.numChan
            errors = prediction-labels_1D
            rmse = np.sqrt(np.mean(((errors)**2),axis=1))
            self.errors[SNR].extend(errors)
            print("SNR: {0:3.2f} RMSE: {1:3.2f} STD: {2:3.2f}".format(SNR,np.mean(rmse),np.std(rmse)))

            corr.append(np.sqrt(np.mean(((labels_1D)**2))))
            i+=1

        self.standard_dev_delays = np.mean(corr) #alter this?

        #return(np.mean(RMSE,axis=1),np.std(RMSE,axis=1),np.mean(corr))
    

    def get_RMSE_Data(self):
        means = []
        SNRs = []
        stds = []
        for SNR in sorted(self.RMSEs.keys()):
            SNRs.append(SNR)
            means.append(np.mean(self.RMSEs[SNR]))
            stds.append(np.std(self.RMSEs[SNR]))
        
        return(np.array(means),np.array(SNRs),np.array(stds),self.standard_dev_delays)
    
    def plot_RMSE_Data(self,fs=(8,8),corrected=True):
        means,SNRs,stds,corr = self.get_RMSE_Data()
        if not corrected or corr == None: corr = 1.0
        plt.figure(figsize=fs)
        plt.errorbar(SNRs,means/corr,yerr=stds/corr,capsize=3,elinewidth=0.5,c ="black", ecolor="Black") 
        plt.xlabel("SNR")
        plt.ylabel("Deviation")

    def save_Data_to_file(self,path):
        np.save(path, np.array(dict(self.RMSEs)),allow_pickle=True)
    
    def load_Data_from_file(self,path):
        P = np.load(path,allow_pickle=True)
        self.RMSEs.update(P.item())