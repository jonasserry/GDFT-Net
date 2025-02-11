from Core import GDFT_Data
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# pylint: disable=E1130

print("Net Version: 1.72")


#FIX THESE IMPORTS
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend 
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,concatenate, Flatten, Dense, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def load_GDFT_Net(path):
    with open(path, 'rb') as input:
        Net = pickle.load(input)
    return(Net)

class GDFT_Net():

    def __init__(self,M1_path,M2_path,dimensions,Net_Path=None):
        """M1,M2 should be paths
        Dimensions written as (x,y)
        """
        self.M1_path = M1_path
        self.M2_path = M2_path
        self.M1 = None
        self.M2 = None
        self.dimensions = dimensions
        self.path = Net_Path

        self.P1_val_loss = []
        self.P1_loss = []
        self.P1_epochs_trained = 0
        self.P1_nN = None
        self.P2_val_loss = []
        self.P2_loss = []
        self.P2_epochs_trained = 0
        self.P2_nN = None

        self.numSteps = None # This would be better done with a simple params dict that can then be passed to all GDFT_Data functions
        self.t0 = None
        self.numChan = None
        self.wavenumberRange = None
        self.numCoherent = None
        self.numIncoherent = None
        self.numSkip = None
        self.dmax=None
        
        self.RMSEs = defaultdict(list)
        self.errors = defaultdict(list)
        self.standard_dev_delays = None

        print("Remember: Load Models")

    def describe(self):
        print("Dimensions: {0}x{1}".format(*self.dimensions))
        print("nN -> P1: {0} | P2: {1}".format(self.P1_nN,self.P2_nN))
        print("Epochs -->  P1: {0} | P2: {1}".format(self.P1_epochs_trained,self.P2_epochs_trained))
        print("Min Loss -->  P1: {0} | P2: {1}".format(min(self.P1_val_loss),min(self.P2_val_loss)))

    def set_training_params(self,numSteps,t0,numChan,wavenumberRange,numCoherent,numIncoherent,numSkip):

        self.numSteps = numSteps
        self.t0 = t0
        self.numChan = numChan
        self.wavenumberRange = wavenumberRange
        self.numCoherent = numCoherent
        self.numIncoherent = numIncoherent
        self.numSkip = numSkip
        self.dmax=numChan/(2*(wavenumberRange[1]-wavenumberRange[0]))
        
    def load_P1_Model(self):
        self.M1 = load_model(self.M1_path)

    def load_P2_Model(self):
        self.M2 = load_model(self.M2_path)

    def load_models(self):
        self.load_P1_Model()
        self.load_P2_Model()
    
    def check_if_loaded(self):
        if self.M1 == None:
            self.load_P1_Model()
        if self.M2 == None:
            self.load_P2_Model()

    def create_P1_Model(self,nN,model = None):
        self.P1_nN = nN
        self.P1_val_loss = []
        self.P1_loss = []
        self.P1_epochs_trained = 0
        if model == None:
            self.M1 = UNet_P1(input_size=(self.dimensions[1],self.dimensions[0],1),nN=nN) 
        else:
            self.M1 = model
        
    
    def train_P1(self,DS,epochs=10,batch_size=16,val_split=0.2):
        assert self.M1 != None,"No Model Loaded"

        train_images, train_labels, _ = DS.get_Shuffled_Data()

        checkpoint = ModelCheckpoint(self.M1_path,monitor="val_loss", save_best_only=True,save_weights_only=False,verbose=1)
        callbacks_list = [checkpoint]
        history = self.M1.fit(train_images, train_labels,batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_split=val_split, verbose = 1)
        self.P1_val_loss.extend(history.history['val_loss'])
        self.P1_loss.extend(history.history['loss'])
        self.P1_epochs_trained += epochs

        plt.figure()
        plt.plot(history.history['loss'],label="test_loss")
        plt.plot(history.history['val_loss'],label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
    
    def test_P1(self,SNR,fs=(10,10),aspect="auto"):
        #assert self.M1 != None,"No Model Loaded"

        raw_image, label_2d, _ = GDFT_Data.Create_Image(self.numSteps, self.dimensions, self.t0 , self.wavenumberRange, self.numChan, self.numCoherent, self.numIncoherent, SNR,self.numSkip)
        p1_pred = self.M1.predict(np.reshape(raw_image,[1] + list(raw_image.shape)))
        self.M1.evaluate(np.reshape(raw_image,[1] + list(raw_image.shape)),np.reshape(label_2d,[1] + list(label_2d.shape)),verbose=1)

        plt.figure(figsize=fs)
        plt.imshow(raw_image[:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect)

        plt.figure(figsize=fs)
        plt.imshow(p1_pred[0,:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect)

        plt.figure(figsize=fs)
        plt.imshow(label_2d[:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect)

    
    def create_P2_Model(self,nN,model = None):
        self.P2_nN = nN
        self.P2_val_loss = []
        self.P2_loss = []
        self.P2_epochs_trained = 0
        if model == None:
            self.M2 = UNet_P2(input_size=(self.dimensions[1],self.dimensions[0],1),nN=nN) 
        else:
            self.M2 = model
    
    def convert_Data_for_P2(self,DS,reload_P1=True):
        """returns shuffled P2 data from given data set"""
        if reload_P1 or not self.M1:
            self.load_P1_Model()
        images,_,Labels_1D = DS.get_Shuffled_Data()
        P2_images = self.M1.predict(images,verbose=1)

        return(P2_images,(Labels_1D+self.dimensions[1]/2)/self.dimensions[1])

    def train_P2(self,DS,epochs=10,batch_size=16,val_split=0.2,save_path = None):
        self.check_if_loaded()
        
        if not save_path:
            save_path = self.M2_path

        train_images, train_labels = self.convert_Data_for_P2(DS)

        checkpoint = ModelCheckpoint(save_path,monitor="val_loss", save_best_only=True,save_weights_only=False,verbose=1)
        callbacks_list = [checkpoint]
        history = self.M2.fit(train_images, train_labels,batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_split=val_split,verbose = 1)
        self.P2_val_loss.extend(history.history['val_loss'])
        self.P2_loss.extend(history.history['loss'])
        self.P2_epochs_trained += epochs

        plt.figure()
        plt.plot(history.history['loss'],label="test_loss")
        plt.plot(history.history['val_loss'],label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()

    def process_Images(self,images,verbose=0):
        First_Pass_Images = self.M1.predict(images,verbose)
        Second_Pass_Images = self.M2.predict(First_Pass_Images,verbose)
        return(First_Pass_Images,Second_Pass_Images)

    def process_Image(self,image,verbose=0):
        P1_Image = self.M1.predict(np.reshape(image,[1] + list(image.shape)),verbose)
        P2_Image = self.M2.predict(P1_Image,verbose)
        return(P1_Image[0],P2_Image[0])


    def plot_Example(self,raw_image,label_2d,label_1d,SNR=1.0,fs=(10,10),aspect="auto"):
        self.check_if_loaded()

        First_Pass_Image,Second_Pass_Image = self.process_Image(raw_image,verbose=0)


        RMSE = np.sqrt(np.mean((((Second_Pass_Image*self.dmax*2-self.dmax)-label_1d)**2)))
        print("Network RMSE: {0:3.1f} Wavelengths".format(RMSE))

        var = np.sqrt(np.mean(((label_1d**2))))

        print("Variation: {0:3.1f} Wavelengths".format(var))

        #Plotting
        
        _, axs = plt.subplots(nrows=2, ncols=2, figsize=fs,sharey=True)

        axs[0, 0].imshow(raw_image[:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect,extent=(0,self.dimensions[0],(-self.dmax),self.dmax))
        axs[0, 0].set_title(r"GDFT Image ($SNR_0$ = {0:3.2f})".format(SNR),fontsize=14)
        axs[0, 0].set_ylabel("OPD(Wavelengths)",fontsize=14)

        axs[1, 0].imshow(label_2d[:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect,extent=(0,self.dimensions[0],(-self.dmax),self.dmax))
        axs[1, 0].set_title("GDFT Image Correct Delays",fontsize=14)
        axs[1, 0].set_ylabel("OPD(Wavelengths)",fontsize=14)
        axs[1, 0].set_xlabel(r"Time/$t_0$",fontsize=14)

        axs[0, 1].imshow(First_Pass_Image[:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect=aspect,extent=(0,self.dimensions[0],(-self.dmax),self.dmax))
        axs[0, 1].set_title("First Pass Network Prediction",fontsize=14)

        x = np.linspace(0,self.numSteps/self.t0,len(Second_Pass_Image))
        axs[1, 1].set_title("Results",fontsize=14)
        axs[1, 1].plot(x,Second_Pass_Image*self.dmax*2-self.dmax,label="GDFT-Net",c="black",ls="--")
        axs[1, 1].plot(x,label_1d,label="True Delays",c="black",ls="-")
        axs[1, 1].set_xlabel(r"Time/$t_0$",fontsize=14)
        axs[1, 1].legend(fontsize=12)
        return()

    def plot_random_Example(self,SNR,fs=(10,10),aspect="auto"):
        raw_image, label_2d, label_1d = GDFT_Data.Create_Image(self.numSteps, self.dimensions, self.t0 , self.wavenumberRange, self.numChan, self.numCoherent, self.numIncoherent, SNR,self.numSkip)
        self.plot_Example(raw_image,label_2d,label_1d,SNR,fs,aspect="auto")

    def save_Net(self,filename=None):
        self.M1 = None
        self.M2 = None

        if filename is None:
            filename = self.path
        else:
            self.path=filename
        
        with open(filename, 'wb') as output:  
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print("Saved as: " + self.path)
        print("Remember to reload models")



    

def UNet_P1 (pretrained_weights = None,input_size = (256,256,1),nN=64,drop=0.4):

    inputs = Input(input_size)
    
    conv1 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(drop)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(nN*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(nN*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(drop)(conv5)

    up6 = Conv2D(nN*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    drop6 = Dropout(drop)(merge6)
    conv6 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop6)
    conv6 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(nN*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    drop7 = Dropout(drop)(merge7)
    conv7 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop7)
    conv7 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(nN*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    drop8 = Dropout(drop)(merge8)
    conv8 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop8)
    conv8 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(nN, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    drop9 = Dropout(drop)(merge9)
    conv9 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
    conv9 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = [inputs], outputs = [conv10])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    model.compile(optimizer = Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()



    return model


def UNet_P2 (pretrained_weights = None,input_size = (256,256,1),nN = 64,drop=0.4):
    
    inputs = Input(input_size)
    conv1 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(drop)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(nN*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(nN*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(drop)(conv5)

    up6 = Conv2D(nN*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    drop6 = Dropout(drop)(merge6)
    conv6 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop6)
    conv6 = Conv2D(nN*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(nN*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    drop7 = Dropout(drop)(merge7)
    conv7 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop7)
    conv7 = Conv2D(nN*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(nN*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    drop8 = Dropout(drop)(merge8)
    conv8 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop8)
    conv8 = Conv2D(nN*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(nN, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    drop9 = Dropout(drop)(merge9)
    conv9 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
    conv9 = Conv2D(nN, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 =  Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 =  Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    
    flatten = Flatten()(conv11)
    drop = Dropout(drop)(flatten)
    
    dense2 = Dense(input_size[1], activation = "sigmoid")(drop)

    model =  Model(inputs = [inputs], outputs = [dense2])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)


    model.compile(optimizer = "adam", loss = "mean_absolute_error", metrics = ["accuracy"])
    

    return (model)
