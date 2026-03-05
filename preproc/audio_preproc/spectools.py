import numpy as np
from scipy.signal import hann, spectrogram
from fbtools import fft2melmx
from matplotlib import pyplot as plt

def powspec(x, sr=8000, wintime=0.025, steptime=0.010, dither=1):
    '''
    # compute the powerspectrum and frame energy of the input signal.
    # basically outputs a power spectrogram
    #
    # each column represents a power spectrum for a given frame
    # each row represents a frequency
    #
    # default values:
    # sr = 8000Hz
    # wintime = 25ms (200 samps)
    # steptime = 10ms (80 samps)
    # which means use 256 point fft
    # hamming window
    #
    # $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

    # for sr = 8000
    #NFFT = 256;
    #NOVERLAP = 120;
    #SAMPRATE = 8000;
    #WINDOW = hamming(200);
    '''

    winpts = np.round(wintime*sr);
    steppts = np.round(steptime*sr);

    NFFT = 2**(np.ceil(np.log(winpts)/np.log(2)));
    WINDOW = hann(np.int(winpts)).T

    # hanning gives much less noisy sidelobes
    NOVERLAP = winpts - steppts
    SAMPRATE = sr

    # Values coming out of rasta treat samples as integers, 
    # not range -1..1, hence scale up here to match (approx)
    f,t,Sxx = spectrogram(x*32768, nfft=NFFT, fs=SAMPRATE, nperseg=len(WINDOW), window= WINDOW, noverlap=NOVERLAP)
    y = np.abs(Sxx)**2

    # imagine we had random dither that had a variance of 1 sample 
    # step and a white spectrum.  That's like (in expectation, anyway)
    # adding a constant value to every bin (to avoid digital zero)
    if dither:
        y = y + winpts

    # ignoring the hamming window, total power would be = #pts
    # I think this doesn't quite make sense, but it's what rasta/powspec.c does

    # 2012-09-03 Calculate log energy - after windowing, by parseval
    e = np.log(np.sum(y))

    return y, e

def audspec(pspectrum, sr=16000, nfilts=80, fbtype='mel', minfreq=0, maxfreq=8000, sumpower=True, bwidth=1.0):
    ''' 
    perform critical band analysis (see PLP)
    takes power spectrogram as input
    '''

    [nfreqs,nframes] = pspectrum.shape

    nfft = (nfreqs-1)*2
    freqs = []

    if fbtype == 'mel':
        wts, freqs = fft2melmx(nfft=nfft, sr=sr, nfilts=nfilts, bwidth=bwidth, minfreq=minfreq, maxfreq=maxfreq);
    elif fbtype == 'htkmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 1);
    elif fbtype == 'fcmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 0);
    elif fbtype == 'bark':
        wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
    else:
        error(['fbtype ' + fbtype + ' not recognized']);

    wts = wts[:, 0:nfreqs]
    #figure(1)
    #plt.imshow(wts)

    # Integrate FFT bins into Mel bins, in abs or abs^2 domains:
    if sumpower:
        aspectrum = np.dot(wts, pspectrum)
    else:
        aspectrum = np.dot(wts, np.sqrt(pspectrum))**2.

    return aspectrum, wts, freqs
