#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from func import beta
from func import visualization
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib


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

def bandFilter(lfp, Fs, filtbound = [10., 30.]):
    filtered_lfp = np.zeros(lfp.shape)
    power = np.zeros(lfp.shape[0])
    filtered_lfp= customFilt(lfp, Fs, filtbound=filtbound)
    for i in range(lfp.shape[0]):
        power[i] = 10*np.log10(np.sum(np.abs(signal.hilbert(filtered_lfp[i,:]))**2))
    return filtered_lfp, power

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
        if isinstance(channel, int):
            channel_range = [channel]
        else:
            channel_range = channel
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

def betaEvent(lfp_beta, betaBurst, Fs, channel = None, win = [-20,20], if_plot = 0):
    if len(betaBurst) > 1:
        raise ValueError('specify betaBurst channel')
    window = np.arange(np.round(win[0]/1000*Fs), np.round(win[1]/1000*Fs), 1).astype(int)
    t = window.astype(float)*1/Fs # in s
    if channel == None:
        LFP = np.zeros((lfp_beta.shape[0], len(window)))
        for i in range(lfp_beta.shape[0]):
            idx_peak = betaBurst[0][1,:].astype(int)
            a = np.zeros((len(idx_peak), len(window)))
            for j in range(len(idx_peak)):
                if (idx_peak[j]+window[0]>0) & (idx_peak[j]+window[-1]<lfp_beta.shape[1]):
                    a[j,:]= lfp_beta[i,idx_peak[j]+window]
            LFP[i,:] = np.mean(a, axis = 0)
    else:
        if isinstance(channel, int):
            idx_peak = betaBurst[0][1,:]
            a = np.zeros((len(idx_peak), len(window)))
            for j in range(len(idx_peak)):
                if (idx_peak[j] + window[0] > 0) & (idx_peak[j] + window[-1] < lfp_beta.shape[1]):
                    a[j,:]= lfp_beta[channel,idx_peak[j]+window]
            LFP = np.mean(a, axis = 0)

        else:
            LFP = np.zeros((len(channel), len(window)))
            for i_beta, i_channel in enumerate(channel):
                idx_peak = betaBurst[0][1, :]
                a = np.zeros((len(idx_peak), len(window)))
                for j in range(len(idx_peak)):
                    if (idx_peak[j] + window[0] > 0) & (idx_peak[j] + window[-1] < lfp_beta.shape[1]):
                        a[j, :] = lfp_beta[i_channel, idx_peak[j] + window]
                LFP[i_beta,:] = np.mean(a, axis = 0)

    if if_plot:
        plt.figure()
        norm_color = matplotlib.colors.Normalize(vmin=0.0, vmax=63.0, clip=True)
        mapper = cm.ScalarMappable(norm=norm_color, cmap=cm.viridis)
        for i in range(LFP.shape[0]):
            plt.plot(t, LFP[i,:]-np.max(np.max(LFP))/5*i, color = mapper.to_rgba(i))
        plt.show()

    return LFP, t

def customCSD(LFP, t, spacing, if_plot = 0, smooth = 1):
    # convert unit
    spacing = spacing/1e6 #um to m
    data = LFP/1e6 #uV to V

    # correct for the edges
    vakData = np.zeros([data.shape[0] + 4, data.shape[1]])
    vakData[0,:] = data[0,:]
    vakData[1,:] = data[0,:]
    vakData[2:-2,:] = data
    vakData[-2,:] = data[-1,:]
    vakData[-1,:] = data[-1,:]

    n = 2
    CSD1 = np.zeros(vakData.shape)

    for i in range(2, vakData.shape[0]-2):
        V_b = vakData[i+n,:]
        V_o =  vakData[i,:]
        V_a = vakData[i-n,:]

        CSD1[i,:] = -(V_b + V_a-2*V_o)/((2*spacing)**2)

    CSD = CSD1[2:-2,:]
    x = np.arange(CSD.shape[0])

    if smooth:
        x_interp = np.arange(0,CSD.shape[0], 1/5)
        CSD_smooth = np.zeros([len(x_interp), CSD.shape[1]])
        for i in range(CSD.shape[1]):
            CSD_smooth[:,i] = np.interp(x_interp, np.arange(CSD.shape[0]), CSD[:,i].reshape([CSD.shape[0],]))
        CSD = CSD_smooth
        x = x_interp

    x = x*spacing*1e6
    if if_plot:
        plt.figure()
        plt.imshow(CSD, origin = 'upper', cmap = 'jet', extent = [t[0]*1e3, t[-1]*1e3, x[-1], x[0]], aspect = 1/5)
        plt.colorbar()
        plt.show()

    return CSD, x

def analyze_beta(cell, electrode, lfp, result_file, if_plot = 1, if_save = 1):
    # M = electrode.calc_mapping(cell)
    # lfp = M @ cell.imem
    lfp = lfp * 1e3  # unit in uV

    Fs = 1 / (cell.dt / 1000.)
    lfp_low = lowpass_filter_lfp(lfp, Fs, Fc=500.)
    [lfp_beta, power] = bandFilter(lfp_low, Fs)
    betaBurst_all = betaBurstDetection(Fs, lfp_beta, channel=None)
    if if_plot:
        visualization.plot_beta_event(lfp, lfp_beta, np.arange(0, 64, 10), cell, betaBurst_all)

    [LFP_beta_narrow, t_CSD] = betaEvent(lfp_beta, [betaBurst_all[13]], Fs, channel=None, win=[-100, 100], if_plot=if_plot)
    [LFP_beta_broad, t_CSD] = betaEvent(lfp, [betaBurst_all[13]], Fs, channel=None, win=[-100, 100], if_plot=if_plot)
    spacing = electrode.y[0]-electrode.y[1]
    [CSD, x_CSD] = customCSD(LFP_beta_narrow, t_CSD, spacing, if_plot=1, smooth=1)
    data = {'t': cell.tvec,
            'Fs': Fs,
            'vm': cell.vmem,
            'im': cell.imem,
            'lfp': lfp,
            'lfp_beta': lfp_beta[13],
            'betaBurst': betaBurst_all[13],
            'LFP_beta_narrow': LFP_beta_narrow,
            'LFP_beta_broad': LFP_beta_broad,
            't_CSD': t_CSD,
            'x_CSD': x_CSD,
            'CSD': CSD}
    if if_save:
        sio.savemat(result_file, data)

