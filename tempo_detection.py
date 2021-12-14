import librosa as lb
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
from scipy import signal
from scipy import stats
import statistics
import os
import glob
import csv
import pandas as pd
from scipy.io import wavfile
import math
import soundfile as sf
import copy

import pywt


# from HW
def FeatureTimeRms(xb):
    # number of results
    numBlocks = xb.shape[0]
    # allocate memory
    vrms = np.zeros(numBlocks)
    for n in range(0, numBlocks):
        # calculate the rms
        vrms[n] = np.sqrt(np.dot(xb[n,:], xb[n,:]) / xb.shape[1])
    # convert to dB
    epsilon = 1e-5  # -100dB
    vrms[vrms < epsilon] = epsilon
    vrms = 20 * np.log10(vrms)
    return (vrms)

# from HW
def block_audio(x,blockSize,hopSize,fs):    
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    #compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]   
    return (xb,t)

# from HW
def comp_acf(inputVector, bIsNormalized = True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1
    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size-1, afCorr.size)]
    return (afCorr)

def onset_detection(xb, t, thresh=0.5):
    # RMS
    rms = FeatureTimeRms(xb)
    # envelope
    B, A = signal.butter(2, 0.1)
    env = signal.filtfilt(B, A, rms)
    diff = np.diff(env)
    # threshold
    mask = diff < thresh
    diff[mask] = 0
    peaks = lb.util.peak_pick(diff, 50,50,10,10,0,50)
    peaks_t = t[peaks]

    return peaks_t

def IOI_histogram(peaks, bins):
    ioi = np.diff(peaks)
    return np.histogram(ioi, bins)

def get_tempo_v1(data, fs):
    xb, t = block_audio(data, 1024, 512, fs)
    peaks = onset_detection(xb, t)
    hist = IOI_histogram(peaks, 10)
    mode_interval = hist[1][np.argmax(hist[0])]
    bpm = 60 / mode_interval
    return bpm


def get_tempo_dwt(x, fs, minTempo=50, maxTempo=200):
    block_len = int(fs / 6)
    xb, t = block_audio(x, block_len, int(block_len/2), fs)
    maxes = np.empty(xb.shape[0] * 5)
    for i in range(xb.shape[0]):
        curr_max = dwt_tempo(xb[i], fs, minTempo, maxTempo)
        maxes[i*5:i*5+5] = curr_max
    hist = np.histogram(maxes, 20)

    print(stats.mode(maxes))
    print(statistics.median(maxes))
    print(hist)
    return hist[1][np.argmax(hist[0])]

def five_peak_detect(data):
    sorted_inds = np.argsort(data)
    maxes = sorted_inds[-5:]
    return maxes

# Based on methods described in Audio analysis using the discrete wavelet transform
# Tzanetakis, G., Essl, G., & Cook, P. (2001, September). Audio analysis using the discrete wavelet transform. 
# In Proc. conf. in acoustics and music theory applications (Vol. 66).
def dwt_tempo(data, fs, levels=4, minTempo=50, maxTempo=200):

    downsample_rate = 2 ** (levels - 1)
    min_tempo = int(60 / maxTempo * (fs / downsample_rate))
    max_tempo = int(60 / minTempo * (fs / downsample_rate))

    [cA, cD] = pywt.dwt(data, "db4")
    cD_len = int(len(cD) / downsample_rate + 1)
    cD_sum = np.zeros(int(cD_len))

    # LPF
    B, A = signal.butter(2, 0.1)
    cD = signal.filtfilt(B, A, cD)
    # FWR
    cD = np.abs(cD)
    # down sample
    cD = signal.decimate(cD, (2**(levels-1)))
    # norm
    cD = cD - np.mean(cD)

    cD_sum = cD[0:cD_len] + cD_sum

    for i in range(1, levels-1):
        
        [cA, cD] = pywt.dwt(cA, "db4")
        # LPF
        B, A = signal.butter(2, 0.1)
        cD = signal.filtfilt(B, A, cD)
        # FWR
        cD = np.abs(cD)
        # down sample
        cD = signal.decimate(cD, (2**(levels-i-1)))
        # norm
        cD = cD - np.mean(cD)

        cD_sum = cD[0:cD_len] + cD_sum

    # add approx coeffs
    # LPF
    B, A = signal.butter(2, 0.1)
    cA = signal.filtfilt(B, A, cA)
    # FWR
    cA = np.abs(cA)
    # down sample
    # cA = signal.decimate(cA, (2**(levels-1)))
    # norm
    cA = cA - np.mean(cA)

    cD_sum = cA[0:cD_len] + cD_sum

    # ACF
    acf = comp_acf(cD_sum)

    arg_max_acf = np.argmax(np.abs(acf[min_tempo:max_tempo]))
    tempo = 60 / (arg_max_acf + min_tempo) * (fs / downsample_rate)
    # maxes = five_peak_detect(np.abs(acf[min_tempo:max_tempo]))
    # tempo = 60 / (maxes + min_tempo) * (fs / downsample_rate)
    return tempo