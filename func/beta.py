import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from func import beta


def lowpass_filter_lfp(lfp, fs, Fc = 500., if_plot = 0):
    """
    lowpass filter
    :param lfp:
    :param fs: sampling rate, Hz
    :param Fc: cut off frequency, Hz
    :return:
    """
    Wn = Fc/(fs/2)
    [b,a] = signal.butter(6, Wn, btype = 'low')
    lfp_low = signal.filtfilt(b, a, lfp)
    if if_plot:
        t = np.arange(lfp.shape[1])/fs
        plt.figure()
        plt.plot(t, lfp[0,:])
        plt.plot(t, lfp_low[0,:])
        plt.show()
    return lfp_low

def customFilt(data, Fs, filtbound = [10.,30.], if_plot = 0):
    # Nyquist frequency
    nyquist = Fs/2
    nbchan = data.shape[0]
    pnts = data.shape[1]
    times = np.arange(pnts)/Fs

    # transition width
    trans_width = 0.2

    # filter order
    filt_order = int(np.round(3*(Fs/filtbound[0])))
    if filt_order%2 == 0:
        filt_order = filt_order + 1

    # frequency vector
    ffrequencies = np.asarray([0, (1-trans_width)*filtbound[0], filtbound[0], filtbound[1], (1+trans_width)*filtbound[1], nyquist])/nyquist

    # shape of filter (must be the same number of element as frequency vector)
    idealresponse = np.asarray([0,0,1,1,0,0])

    # get filter weights
    filterweights = signal.firls(filt_order, ffrequencies, idealresponse)

    if if_plot:
        plt.figure()
        ax = plt.subplot(211)
        ax.plot(ffrequencies*nyquist, idealresponse, 'ko--')
        ax.set_xlim(0, 2*filtbound[1])
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')

        ax = plt.subplot(212)
        ax.plot(np.arange(filt_order)*(1000/Fs), filterweights)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        plt.show()

    filtered_data = np.zeros([nbchan, pnts])
    for chani in range(nbchan):
        filtered_data[chani,:] = signal.filtfilt(filterweights, 1, data[chani,:])

    if if_plot:
        plt.figure()
        plt.plot(times, np.squeeze(data[0,:]))
        plt.plot(times, np.squeeze(filtered_data[0,:]), 'r', linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('voltage [\muV]')
        plt.show()

    return filtered_data

def bandFilter(lfp, filtbound = [10., 30.]):
    filtered_lfp = np.zeros(lfp.shape)
    power = np.zeros(lfp.shape[0])
    filtered_lfp= beta.customFilt(lfp, Fs, filtbound=filtbound)
    for i in range(lfp.shape[0]):
        power[i] = 10*np.log10(np.sum(np.abs(signal.hilbert(filtered_lfp[i,:]))**2))
    return filtered_lfp, power
#%%
def betaBurstDetection(Fs, beta_signal,channel = None, window = [0,1000]):
    trialFlag = 0

    lowThresholdFactor = 1
    highThresholdFactor = 2.5
    minInterRippleInterval = 25  # ms
    minBetaDuration = 30
    maxBetaDuration = 250
    noise = np.asarray([])

    windowLength = np.round(11)
    beta_events = []
    if channel == None:
        channel_range = range(beta_signal.shape[0])
    else:
        channel_range = list(channel)
    if trialFlag:
        timestamps = np.arange(window[idx, 0], 1 / Fs, window[idx, 1])
    else:
        timestamps = np.arange(beta_signal.shape[1]) * 1 / Fs
    for idx in channel_range:
        signal_c = beta_signal[idx,:]
        squaredSignal = signal_c**2
        normalizedSquaredSignal = (squaredSignal - np.mean(squaredSignal))/np.std(squaredSignal)
        # detect beta period by thresholding normalized squared signal
        thresholded = normalizedSquaredSignal > lowThresholdFactor
        thresholded = thresholded.astype('float')

        start = np.where(np.diff(thresholded)>0)[0]
        stop = np.where(np.diff(thresholded)<0)[0]
        # exclude last beta if it is incomplete
        if len(stop) == len(start)-1:
            start = start[:-1]

        # Exclude first beta if it is incomplete
        if len(stop)-1 == len(start):
            stop = stop[1:]

        # correct special case when both first and last ripples are incomplete
        if start[0] > stop[0]:
            stop = stop[1:]
            start = start[:-1]

        firstPass = np.asarray([start, stop])
        if firstPass.shape[1]==0:
            print('Detection by thresholding failed')
        else:
            print('After detection by thresholding: ' + str(firstPass.shape[1]) + ' events')

        # merge beta events if inter-beta period is too short
        minInterRippleSamples = minInterRippleInterval/1000*Fs
        secondPass = np.zeros([2,1])
        betaEvent = firstPass[:,0].reshape([2,1])
        for i in range(1, firstPass.shape[1]):
            if firstPass[0,i] - betaEvent[1,0] < minInterRippleSamples:
                # merge
                betaEvent = np.asarray([betaEvent[0,0], firstPass[1,i]]).reshape([2,1])
            else:
                secondPass = np.append(secondPass, betaEvent, axis = 1)
                betaEvent = firstPass[:,i].reshape([2,1])

        secondPass = np.append(secondPass, betaEvent, axis = 1)
        secondPass = secondPass[:,1:]

        if secondPass.shape[1]==0:
            print('Ripple merge failed')
        else:
            print('After ripple merge: ' + str(secondPass.shape[1]) + ' events')

        # discard beta events with a peak power < highThreshold Factor
        thirdPass = np.zeros([2,1])
        for i in range(secondPass.shape[1]):
            maxValue = np.max(normalizedSquaredSignal[int(secondPass[0,i]): int(secondPass[1,i])])
            if maxValue > highThresholdFactor:
                thirdPass = np.append(thirdPass, secondPass[:,i].reshape([2,1]), axis = 1)
        thirdPass = thirdPass[:, 1:]

        if thirdPass.shape[1] == 0:
            print('Peak threshold thresholding failed')
        else:
            print('After peak thresholding: ' + str(thirdPass.shape[1]) + ' events')

        # discard beta Event that's too long or too short
        fourthPass = np.zeros([2,1])
        for i in range(thirdPass.shape[1]):
            duration = timestamps[int(thirdPass[1,i])]-timestamps[int(thirdPass[0,i])]
            if (duration > maxBetaDuration/1000) or (duration < minBetaDuration/1000):
                continue
            else:
                fourthPass = np.append(fourthPass, thirdPass[:,i].reshape([2,1]), axis = 1)
        fourthPass = fourthPass[:,1:]

        if fourthPass.shape[1] == 0:
            print("Duration thresholding failed")
        else:
            print("After duration thresholding: " + str(fourthPass.shape[1]) + " events")

        peakPosition = []
        peakNormalizedPower = []
        for i in range(fourthPass.shape[1]):
            minIndex = np.argmin(signal_c[int(fourthPass[0,i]): int(fourthPass[1,i])])
            maxValue = np.max(normalizedSquaredSignal[int(fourthPass[0, i]): int(fourthPass[1, i])])
            peakPosition.append(minIndex + thirdPass[0,i]-1)
            peakNormalizedPower.append(maxValue)

        beta_1 = np.asarray([fourthPass[0,:], np.asarray(peakPosition), fourthPass[1,:], np.asarray(peakNormalizedPower)])
        beta_events.append(beta_1)

    return beta_events


