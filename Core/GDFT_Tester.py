
from Core import GDFT_Data
from Core import GDFT_Net
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

print("Tester Version: 1.00")

class GDFT_Net_Tester():

    def __init__(self,Path,Net_Path,dimensions,t0,numChan,numSteps,wavenumberRange,numCoherent,numIncoherent,numSkip):
        
        self.dimensions = dimensions
        self.numSteps = numSteps
        self.t0 = t0
        self.numChan = numChan
        self.dimensions = dimensions
        self.wavenumberRange = wavenumberRange
        self.numCoherent = numCoherent
        self.numIncoherent = numIncoherent
        self.numSkip = numSkip

        self.Path=Path
        self.Net_Path = Net_Path
        self.Net=None

        self.RMSEs = defaultdict(list)
        self.standard_dev_delays = None


    def load_Nets(self):
        self.Net = GDFT_Net.load_GDFT_Net(self.Net_Path)
        self.Net.load_models()


    def run_RMSE_Testing(self,numImages,SNRs):
        corr = []
        i=0
        for SNR in SNRs:
            raw_images,_,labels_1D =  GDFT_Data.Create_Images(numImages, self.numSteps, self.dimensions, self.t0 , self.wavenumberRange, self.numChan, self.numCoherent, self.numIncoherent, SNR,numSteps_simulated=1024*1024,print_flag=False)
            prediction = self.Net.process_Images(raw_images,verbose=0)[1]*self.numChan*2-self.numChan
            rmse = np.sqrt(np.mean(((prediction-labels_1D)**2),axis=1))
            self.RMSEs[SNR].extend(rmse)

            print("SNR: {0:3.2f} RMSE: {1:3.2f}".format(SNR,np.mean(rmse)))

            corr.append(np.sqrt(np.mean(((labels_1D)**2))))
            i+=1

        self.standard_dev_delays = np.mean(corr) #alter this?

        #return(np.mean(RMSE,axis=1),np.std(RMSE,axis=1),np.mean(corr))


    def get_RMSE_Data(self):
        means = []
        SNRs = []
        stds = []
        for SNR in sorted(self.RMSEs.keys()):
            SNRs.append(SNRs)
            means.append(np.mean(self.RMSEs[SNR]))
            stds.append(np.std(self.RMSEs[SNR]))
        
        return(means,SNRs,stds)