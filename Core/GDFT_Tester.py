
from Core import GDFT_Data
from Core import GDFT_Net
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

print("Tester Version: 1.02")

def load_tester(path):
    with open(path, 'rb') as input:
        tester = pickle.load(input)
    return(tester)

class GDFT_Net_Tester():

    def __init__(self,Tester_Path,Net_Path,dimensions):
        
        self.Path=Tester_Path
        self.Net_Path = Net_Path
        self.Net=None
        self.version = 1.1
        self.dimensions=dimensions

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
            self.errors[round(SNR,2)].extend(errors)
            print("SNR: {0:3.2f} RMSE: {1:3.2f} STD: {2:3.2f}".format(SNR,np.mean(rmse),np.std(rmse)))

            corr.append(np.sqrt(np.mean(((labels_1D)**2))))
            i+=1

        self.standard_dev_delays = np.mean(corr) #alter this?

    def get_RMSE_Data(self):
        means = []
        SNRs = []
        stds = []
        for SNR in sorted(self.errors.keys()):
            SNRs.append(SNR)
            rmses = np.sqrt(np.mean((np.array(self.errors[SNR])**2),axis=1))
            means.append(np.mean(rmses))
            stds.append(np.std(rmses))
        
        return(np.array(SNRs),np.array(means),np.array(stds))
    
    def get_error_at_index(self,i):
        means = []
        SNRs = []
        stds = []
        for SNR in sorted(self.errors.keys()):
            SNRs.append(SNR)
            err = np.abs(np.array(self.errors[SNR])[:,i])
            means.append(np.mean(err))
            stds.append(np.std(err))
        return(np.array(SNRs),np.array(means),np.array(stds))
    
    def get_error_at_index(self,i):
        means = []
        SNRs = []
        stds = []
        for SNR in sorted(self.errors.keys()):
            SNRs.append(SNR)
            err = np.abs(np.array(self.errors[SNR])[:,i])
            means.append(np.mean(err))
            stds.append(np.std(err))
        return(np.array(SNRs),np.array(means),np.array(stds))
    
    def get_error_variation_at_SNR(self,SNR):
        means = []
        inds =[]
        stds = []
        for i in range(self.dimensions[0]):
            inds.append(i)
            err = np.abs(np.array(self.errors[SNR])[:,i])
            means.append(np.mean(err))
            stds.append(np.std(err))
        return(np.array(inds),np.array(means),np.array(stds))
    
    def get_max_error(self):
        means = []
        SNRs = []
        stds = []
        for SNR in sorted(self.errors.keys()):
            SNRs.append(SNR)
            rmses = np.max((np.abs(self.errors[SNR])),axis=1)
            means.append(np.mean(rmses))
            stds.append(np.std(rmses))
        
        return(np.array(SNRs),np.array(means),np.array(stds))

    
    def plot_this_data(self,SNRs,means,stds,fig_size=(8,8),corr=1,xlabel="SNR",ylabel="RMSE",label=None,title=None,fontsize=12):
        plt.figure(figsize=fig_size)
        plt.errorbar(SNRs,means/corr,yerr=stds/corr,capsize=3,elinewidth=0.5,c ="black", ecolor="Black",label=label) 
        plt.title(title,fontsize=fontsize*1.5)
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel(ylabel,fontsize=fontsize)

    def save_data_to_file(self,path):
        np.save(path, np.array(dict(self.errors)),allow_pickle=True)
    
    def load_data_from_file(self,path):
        P = np.load(path,allow_pickle=True)
        self.errors.update(P.item())
    
    def save(self,path=None):
        if not path:
            path = self.Path
        self.Net = None
        with open(path, 'wb') as output:  
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print("Reload Net")

