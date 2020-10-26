import matplotlib.pyplot as plt
import numpy as np


def findSNR(SNRs,Bats,i):
  """Find SNR of sample i in variable SNR Data with distribution given by Bats and SNRs"""
  cum_Bats = np.cumsum(Bats)
  n=0
  while i > cum_Bats[n] and n<len(Bats):
    n+=1
  return(SNRs[n])

def plotImage(Image,title = "", extent = (0,1000,-128,128)):
    """ Plot Data Frame x """
    plt.figure()
    plt.imshow(Image[:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect="auto",extent=extent)
    plt.suptitle(title)
    plt.xlabel("time/$t_0$")
    plt.ylabel("delay (wavelengths)")


def plotFromImages(Images,i,paramsDict):
    """ Plot Data Frame x """
    plt.figure()

    extent = paramsDict["extent"]
    title = "SNR= %s" % (findSNR(paramsDict["SNRs"],paramsDict["Bats"],i))

    plt.imshow(Images[i][:,:,0], cmap=plt.get_cmap('gray_r'),origin="lower",aspect="auto",extent=extent)
    plt.suptitle(title)
    plt.xlabel("time/$t_0$")
    plt.ylabel("delay (wavelengths)")

