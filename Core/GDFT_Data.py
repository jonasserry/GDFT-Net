import time
import IPython
import gc
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from Core import GDFT_Sim as Sim

print("Data Version: 1.61")

###---------------- Image Creation --------------------
def DownSample(image,dimensions):
    """
    Takes image and downsamples and resizes to given dimensions using openCV
    Returns image with dimensions (dimensions[0],dimensions[1],1)
    """
    x = cv2.resize(image,dimensions,interpolation = cv2.INTER_AREA) #Interpolation type?
    x = cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return np.reshape(x,list(x.shape) + [1])

def Create_Image_From_Delays(delays,wavenumberRange,numChan,numCoherent,numIncoherent,SNR,dimensions,numSkip,numSteps=None,t0=None):
    """ Returns raw GDFT image, 2D Mask, and 1D Label from given set of delays using specified params"""
    raw_image,raw_label = Sim.modifiedGDT(delays,wavenumberRange,numChan,numCoherent,numIncoherent,SNR)
    raw_image = DownSample(raw_image[:,numSkip:],dimensions)
    raw_label = DownSample(raw_label[:,numSkip:],dimensions)
    decimated = Decimate_Delays(delays,dimensions[0])
    return raw_image,raw_label,decimated

def Create_Image(numSteps = 1024*128, dimensions =(256,256), t0 = 10, wavenumberRange=(0.8,1.2), numChan = 100, numCoherent=16, numIncoherent=25, SNR=1,numBatches=1,numSkip=20):
    """Returns raw GDFT image, 2D Mask, and 1D Label created using provided parameters"""
    delays = Sim.von_karman_temporal_samples(1024*1024,t0,T0=1e4, two_telescopes=True)[0:numSteps]
    return(Create_Image_From_Delays(delays,wavenumberRange,numChan,numCoherent,numIncoherent,SNR,dimensions,numSkip))

def Create_Images(NumImages, numSteps = 1024*128, dimensions =(256,256), t0 = 10, wavenumberRange=(0.8,1.2), numChan = 100, numCoherent=16, numIncoherent=25, SNR=1,numBatches=1,numSkip=20,numSteps_simulated=1024*1024,print_flag=True):
    """Returns specified number of raw GDFT image, 2D Mask, and 1D Label created using provided parameters"""
    Images = np.empty((NumImages,dimensions[1],dimensions[0],1))
    Labels_2D = np.empty((NumImages,dimensions[1],dimensions[0],1))
    Labels_1D = np.empty((NumImages,dimensions[0]))
    
    start_time = time.time()

    Images_per_simulated_delays = int(numSteps_simulated/numSteps)

    delays = Sim.von_karman_temporal_samples(numSteps_simulated,t0,T0=1e4, two_telescopes=True)

    image_index = 0
    
    for i in range(NumImages):

        if image_index == Images_per_simulated_delays:
            delays = Sim.von_karman_temporal_samples(numSteps_simulated,t0,T0=1e4, two_telescopes=True)
            image_index = 0

        image,label_2D,label_1D = Create_Image_From_Delays(delays[numSteps*image_index:numSteps*image_index+numSteps],wavenumberRange,numChan,numCoherent,numIncoherent,SNR,dimensions,numSkip)
        
        Labels_2D[i] = label_2D
        Labels_1D[i] = label_1D
        Images[i] = image
        image_index+=1

        
        if i%10 ==0 and print_flag:
            t = (time.time()-start_time) / (i+1) * (NumImages-i)
            print("\rBatches remaining: %i | Images Remaining in Batch: %s | Time left in Batch: %s" %(numBatches, NumImages-i, time.strftime("%H:%M:%S", time.gmtime(t))),end='\r')

    if print_flag:
        total_t = time.time()-start_time
        print("\rFinished Batch | Time taken: %s | Total Time Left: %s" % (time.strftime("%H:%M:%S", time.gmtime(total_t)),time.strftime("%H:%M:%S", time.gmtime(total_t*(numBatches-1)))))
    return (Images,Labels_2D,Labels_1D)

def ConvertForNextNetwork(train_labels):
    """Convert 2D Labels into 1D Labels simply using argmax. NOTE: this is now deprecated"""
    CorrectFeature = np.empty((train_labels.shape[0],train_labels.shape[2]))

    for i in range(train_labels.shape[0]):
        CorrectFeature[i] = Convert_to_1D_Label(train_labels[i])

    return (CorrectFeature)

def Convert_to_1D_Label(label):
    """Convert 2D Label into 1D Label simply using argmax. NOTE: this is now deprecated"""
    return(np.reshape(np.argmax(label,0),-1)/label.shape[0])

def Decimate_Delays(delays,x_dim):
    """Returns decimated (and filtered) delays with dimension given by x_dim"""
    decimated = sig.decimate(delays,int(len(delays)/x_dim),axis=0,ftype = "fir")

    assert(len(decimated)==x_dim), "Decimated length: {0} | Desired dimension {1}".format(len(decimated),x_dim)
    return(decimated/2/np.pi)

###---------------- Data Set Creation --------------------

def create_Data_Set(id,NumImages,SNRs,t0=16, numSteps = 1024*128, dimensions =(256,256), wavenumberRange=(1.5,2.0), numChan = 100, numCoherent=16, numIncoherent=25,numSkip=0,**kwargs):
    """Returns variable SNR GDFT Data Set with provided SNR distribution and GDFT parameters"""
    assert(len(NumImages)==len(SNRs))
    Images = np.empty((np.sum(NumImages),dimensions[1],dimensions[0],1))
    Labels_2D = np.empty((np.sum(NumImages),dimensions[1],dimensions[0],1))
    Labels_1D = np.empty((np.sum(NumImages),dimensions[0]))
    n=0
    i=0
    while i<len(NumImages):
        images,labels_2D,labels_1D = Create_Images(NumImages[i],SNR = SNRs[i],numSteps=numSteps, dimensions = dimensions, t0=t0, wavenumberRange = wavenumberRange, numChan = numChan, numCoherent=numCoherent, numIncoherent=numIncoherent,numBatches=(len(NumImages)-i),numSkip=numSkip)
        Images[n:n+NumImages[i]] = images
        Labels_2D[n:n+NumImages[i]] = labels_2D
        Labels_1D[n:n+NumImages[i]] = labels_1D
        n+=NumImages[i]
        i+=1
    return GDFT_Data_Set(id,Images,Labels_2D,Labels_1D,NumImages,SNRs,t0,numChan,dimensions,numSteps,wavenumberRange,numCoherent,numIncoherent,numSkip)

def create_Data_Sets(id,NumImages,SNRs,t0=10, numSteps = 128000, y_dim=64,x_dims=[16,32,64,128,256,512], wavenumberRange=(1.5,2.0), numChan = 32, numCoherent=10, numIncoherent=25,numSkip=0,**kwargs):
    """Returns variable SNR GDFT Data Sets. A single set of GDFT samples is created using the final provided dimension (x_dims[-1]).
    This set is chopped up to create data sets at other provided dimensions. """
    assert(len(NumImages)==len(SNRs))
    Images = np.empty((np.sum(NumImages),y_dim,x_dims[-1],1))
    Labels_2D = np.empty((np.sum(NumImages),y_dim,x_dims[-1],1))
    Labels_1D = np.empty((np.sum(NumImages),x_dims[-1]))
    
    
    n=0
    i=0
    while i<len(NumImages):         # Create Images at maximum dimension
        images,labels_2D,labels_1D = Create_Images(NumImages[i],SNR = SNRs[i],numSteps=numSteps, dimensions = (x_dims[-1],y_dim), t0=t0, wavenumberRange = wavenumberRange, numChan = numChan, numCoherent=numCoherent, numIncoherent=numIncoherent,numBatches=(len(NumImages)-i),numSkip=numSkip)
        Images[n:n+NumImages[i]] = images
        Labels_2D[n:n+NumImages[i]] = labels_2D
        Labels_1D[n:n+NumImages[i]] = labels_1D
        n+=NumImages[i]
        i+=1

    Sets = []
    for x in x_dims:    # Chop images into smaller dimensions
        images = []
        labels_2d =[]
        labels_1d = []
        j=0
        for n in NumImages:
            i=0
            while i < x_dims[-1]/x:
                images.extend(Images[j:j+n,:,x*i:x*i+x,:])
                labels_2d.extend(Labels_2D[j:j+n:,:,x*i:x*i+x,:])
                labels_1d.extend(Labels_1D[j:j+n:,x*i:x*i+x])
                i+=1
            j+=n
        Sets.append(GDFT_Data_Set(id+str(x),images,labels_2d,labels_1d,(np.array(NumImages)*x_dims[-1]/x).astype(int),SNRs,t0,numChan,(x,y_dim),int(numSteps*x/x_dims[-1]),wavenumberRange,numCoherent,numIncoherent,numSkip))
    return Sets


###---------------- GDFT Data Set --------------------


class GDFT_Data_Set():

    def __init__(self,id,Images,Labels_2D,Labels_1D,NumImages,SNRs,t0,numChan,dimensions,numSteps,wavenumberRange,numCoherent,numIncoherent,numSkip):
        self.path = None
        self.id = id
        self.SNRs = SNRs
        self.numSteps = numSteps
        self.t0 = t0
        self.numChan = numChan
        self.dimensions = dimensions
        self.wavenumberRange = wavenumberRange
        self.numCoherent = numCoherent
        self.numIncoherent = numIncoherent
        self.numSkip = numSkip
        self.Images = Images
        self.Labels_1D = Labels_1D
        self.Labels_2D = Labels_2D
        self.Image_Nums = NumImages

        self.dmax=numChan/(2*(wavenumberRange[1]-wavenumberRange[0]))

    def get_Params(self):
        return(self.numSteps,self.t0,self.numChan,self.wavenumberRange,self.numCoherent,self.numIncoherent,self.numSkip)

    def save_As(self,path):
        with open(path+self.id+".pkl", 'wb') as output:  
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.path=path
        print("Saved as: " + path+self.id+".pkl")
    
    def save(self):
        if self.path == None:
            raise Exception("No path set. Use save_As")
        with open(self.path, 'wb') as output:  
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print("Saved as: " + self.path)

    def describe(self):
        print("------------------------ID: %s ----------------------------"%(self.id))
        print("numChan {0}".format(self.numChan))
        print("FINISH THIS")
    
    def get_Data(self,with_SNR=None):
        "Returns Unshuffled Images and Labels,"
        if with_SNR == None:
            return(np.array(self.Images),np.array(self.Labels_2D),np.array(self.Labels_1D))
        else:
            i = self.SNRs.index(with_SNR) #SNR Index
            end = np.cumsum(self.Image_Nums)[i]
            if i == 0:
                start = 0
                
            else:
                start = np.cumsum(self.Image_Nums)[i-1]
            return(np.array(self.Images[start:end]),np.array(self.Labels_2D[start:end]),np.array(self.Labels_1D[start:end]))
    
    def get_Shuffled_Data(self):
        "returns shuffled COPY. Watch out for space"
        rng_state = np.random.get_state()
        a = np.random.permutation(self.Images)
        np.random.set_state(rng_state)
        b =  np.random.permutation(self.Labels_2D)
        np.random.set_state(rng_state)
        c =  np.random.permutation(self.Labels_1D)
        return(a,b,c)
    
    def findSNR(self,i):
        """Find SNR of sample i in variable SNR Data with distribution given by Bats and SNRs"""
        cum_Bats = np.cumsum(self.Image_Nums)
        n=0
        while i > cum_Bats[n] and n<len(self.Image_Nums):
            n+=1
        return(self.SNRs[n])

    def plot_Image_at_Index(self,i,title="",fs=10,aspect="auto", figsize=(10, 6)):
        """Plots Image and Label at given Index"""
        _, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize,sharex=True)
        axs[0].imshow(self.Images[i][:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect,extent=(0,self.numSteps/self.t0,-self.dmax,self.dmax))
        axs[1].imshow(self.Labels_2D[i][:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect,extent=(0,self.numSteps/self.t0,-self.dmax,self.dmax))
        axs[0].set_ylabel("OPD(Wavelengths)",fontsize=fs)
        axs[1].set_ylabel("OPD(Wavelengths)",fontsize=fs)
        axs[1].set_xlabel("time/$t_0$",fontsize=fs)
        axs[0].set_title("Image (SNR = %s)" % (self.findSNR(i)),fontsize=fs*1.5)
        axs[1].set_title("Label",fontsize=fs*1.5)
        
        axs[2].plot(np.linspace(0,self.numSteps/self.t0,len(self.Labels_1D[i])),self.Labels_1D[i])

        plt.suptitle(title)

    


def load_Data_Set(path):
    with open(path, 'rb') as input:
        Set = pickle.load(input)
    return(Set)