import scipy.signal
import numpy as np
from numpy.random import normal


def temprl(nsamp,t0,index=-4.0/3.0):
    """Generate a time sequence of samples of atmospheric temporal
    perturbations with a Kolmogorov-Tatarski structure function."""
    temp=nsamp/float(t0)
    const=np.sqrt(0.011193/temp/2./2.)/temp**index*nsamp
    amplitude=np.arange(nsamp/2+1,dtype=np.float64)
    amplitude[1:]=const*(amplitude[1:]**index)
    noise=normal(size=(2,int(nsamp/2+1)))
    return np.fft.irfft(amplitude*(noise[0]+1j*noise[1]))
    
def RcFilter(samples,tau):
    e=np.exp(-1.0/tau)
    return scipy.signal.lfilter([1-e],[1,-e],samples,axis=0)

def chop(samples,blockSize):
    """Chop first dimension of array into a 2-d array of blocks of length blockSize.  The
    original dimension does not have to be a multiple of blockSize - the remainder
    is discarded. Will return an error for arrays which cannot be reshaped in this
    way without copying"""
    maxSamp=samples.shape[0]
    numBlock=maxSamp//blockSize
    numSamp=numBlock*blockSize
    self=samples[:numSamp].view()
    self.shape=(numBlock,blockSize)+samples.shape[1:]
    return self

def BlockAverage(samples,blockSize):
    return np.sum(chop(samples,blockSize),axis=1)/float(blockSize)

def DispersedFringes(delay,wavenumberRange,numChan):
    wavenumber=np.linspace(wavenumberRange[0],wavenumberRange[1],numChan)
    fringePhase=np.multiply.outer(delay,wavenumber)
    v=np.exp(1j*fringePhase)
    return v

def PowerSpectrum1d(v,oversample=2):
    window = np.hamming(v.shape[-1])
    return np.fft.fftshift(abs(np.fft.fft(v*window,axis=-1,n=v.shape[-1]*oversample))**2,axes=(-1,))

def ComplexNoise(shape,sigma=1.0):
    r=np.random.normal(size=shape+(2,),scale=sigma/np.sqrt(2))
    return r[...,0]+1j*r[...,1]


def GroupDelaySimulation(phase,wavenumberRange,numChan,numCoherent,numIncoherent,SNR):
    coherentVisibilities=BlockAverage(DispersedFringes(phase,wavenumberRange,numChan),numCoherent)
    coherentVisibilities+=ComplexNoise(coherentVisibilities.shape,sigma=np.sqrt(numChan)/SNR)
    delaySpectrum=RcFilter(PowerSpectrum1d(coherentVisibilities),numIncoherent)
    return delaySpectrum


def modifiedGDT(phase,wavenumberRange,numChan,numCoherent,numIncoherent,SNR):
    """Returns GDT output with given phase behaviour with and without applied random Noise"""
    coherentVisibilities=BlockAverage(DispersedFringes(phase,wavenumberRange,numChan),numCoherent)
    coherentVisibilities_withNoise=ComplexNoise(coherentVisibilities.shape,sigma=np.sqrt(numChan)/SNR) + coherentVisibilities
    withNoise=RcFilter(PowerSpectrum1d(coherentVisibilities_withNoise),numIncoherent)
    withoutNoise=RcFilter(PowerSpectrum1d(coherentVisibilities),numIncoherent)
    return np.transpose(withNoise), np.transpose(withoutNoise)



def von_karman_temporal_samples(nsamp, t0, T0=1e6, two_telescopes=False):
    """
    Return temporal samples of phase perturbations corresponding to Von Karman turbulence

    Parameters
    ----------
    nsamp : int
         Number of time samples to generate - should be much larger than T0
    t0 : float
         Coherence time measured in samples t_0=0.314 r_0/V where V is effective windspeed.
    T0 : float
         Temporal outer scale T_0=L_0/V.
    two_telescopes : boolean
         Simulate phase sequences corresponding to the phase difference between two
         uncorrelated telescopes i.e. twice the variance. If false, simulate the
         perturbations above a single telescope.

    Returns:
    --------
    samples : numpy.ndarray[float]
              Samples of the phase perturbations at intervals of 1/t0

    Notes:
    ------
    A suitable setting for t_0 might be of order 10 samples.

    For r_0=31.4cm (a moderate H-band value) and V=10m/s, then t_0=10ms.
    If L_0=100m then T0=10s, i.e. 1000t_0, or T0=10^4 samples in this example.
    """
    f = np.fft.rfftfreq(nsamp)
    # Spectrum scale factor: divide by a factor of 2 to account for noise having a variance of 2
    # Divide by a second factor of two to account for a single-sided spectrum
    # Multiply by a factor of 2 if we want to represent the differential between
    # two telescopes.
    # Multiply by delta-f(=f[1]) to account for power in the range f->f+delta-f
    scale = 0.011193 / (2.0 if two_telescopes else 4.0) * f[1]
    spectrum = scale * t0 ** (-5.0 / 3.0) * (f ** 2 + 1 / T0 ** 2) ** (-4.0 / 3.0)
    noise = normal(size=(2, len(f)))
    # Multiply by nsamp to counteract the normalisation of the inverse fft
    return nsamp * np.fft.irfft(np.sqrt(spectrum) * (noise[0] + 1j * noise[1]))
