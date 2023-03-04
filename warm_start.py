#%%
import sys
import os
wd = 'E:\\Code\\l5_beta' # working directory
sys.path.insert(1, wd)
import LFPy

import numpy as np
import pickle

import matplotlib.pyplot as plt
from neuron import h
import neuron

from func import l5_neuron_model
from func import parametersL5_Hay
from func import visualization
from func import sequences
from func import beta
import scipy.io as sio

h('forall pop_section()')
h('forall delete_section()')

P = parametersL5_Hay.init_params(wd)
cell = l5_neuron_model.L5Model(wd, P, verbool = True)
cell.dt = 0.1


x_off = -20
y_off = 0
z_off = 0
nChannel = 64
sigma = 0.3

N = np.empty((nChannel, 3))
x = np.zeros(nChannel) + round(cell.xmid_entire) + x_off
y = np.linspace(cell.yrange_entire[0] - 5, cell.yrange_entire[1] + 5, nChannel) + y_off
y = np.flip(y, axis=0)
z = np.zeros(nChannel) + round(cell.zmid_entire) + z_off
for i in range(N.shape[0]):
    N[i, :] = [1, 0, 0]

electrodeParameters = {
    'sigma': sigma,  # Extracellular potential
    'x': x,  # x,y,z-coordinates of electrode contacts
    'y': y,
    'z': z,
    'n': 10,  # number of point on the electrode surface to average the potential
    'r': 5,  # radius of electrode surface
    'N': N,
    'method': 'pointsource',
}
electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
M = electrode.calc_mapping(cell)  # use M = elestrode.get_transformation_matrix(cell) for newer version of LFPy

mod_freqs = [0, 1, 5, 10, 15, 20, 25, 30, 40]
# mod_freqs = [15]
for mod_freq in mod_freqs:
    result_file = 'i_mod_%d.mat' % mod_freq
    result_file = wd + '\\outputs\\' + result_file
    data = sio.loadmat(result_file)
    
    t = data['t']
    Fs = data['Fs']
    Vm = data['vm']
    # im = data['im']
    lfp = data['lfp']
    lfp_beta = data['lfp_beta']
    betaBurst = data['betaBurst']
    LFP_beta_narrow = data['LFP_beta_narrow']
    # LFP_beta_broad = data['LFP_beta_broad']
    t_CSD = data['t_CSD']
    x_CSD = data['x_CSD']
    CSD = data['CSD']

    cell.vmem = Vm
    # cell.imem = im
    cell.tvec = t.flatten()


    if_plot = 0
    Fs = 1 / (cell.dt / 1000.)
    lfp_low = beta.lowpass_filter_lfp(lfp, Fs, Fc=500.)
    [lfp_beta, power, phase_beta, amp_beta] = beta.bandFilter(lfp_low, Fs, filtbound = [10., 30.])
    [lfp_theta, power, phase_theta, amp_theta] = beta.bandFilter(lfp_low, Fs, filtbound = [4.0, 10.0])
    betaBurst_all = beta.betaBurstDetection(Fs, lfp_beta, channel=None)
    idx_beta = 13# np.argmax(power)
    [LFP_beta_narrow, t_CSD] = beta.betaEvent(lfp_beta, [betaBurst_all[idx_beta]], Fs, channel=None, win=[-100, 100], if_plot=if_plot)
    # [LFP_beta_broad, t_CSD] = betaEvent(lfp, [betaBurst_all[13]], Fs, channel=None, win=[-100, 100], if_plot=if_plot)
    spacing = electrode.y[0]-electrode.y[1]
    [CSD, x_CSD] = beta.customCSD(LFP_beta_narrow, t_CSD, spacing, if_plot=if_plot, smooth=1)
    # Vm = visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]', 'apic[61]'], [0, cell.tvec[-1]], if_plot=0)
    vs1 = beta.spike_phase_coherance(phase_beta, Vm)
    vs2 = beta.band_phase_coherance(phase_theta, amp_beta)
    vs3 = beta.spike_apic_coherance(Vm, Fs, filtbound = [10., 30.])
    # gGABA_apic = beta.record_apical_inhibition(cell)
    # data = {'t': cell.tvec,
    #         'Fs': Fs,
    #         'vm': Vm,
    #         'gGABA': gGABA_apic,
    #         # 'im': cell.imem,
    #         'lfp': lfp,
    #         'idx_beta': idx_beta,
    #         'lfp_beta': lfp_beta[idx_beta],
    #         'lfp_theta': lfp_theta[idx_beta],
    #         'betaBurst': betaBurst_all[idx_beta],
    #         'LFP_beta_narrow': LFP_beta_narrow,
    #         # 'LFP_beta_broad': LFP_beta_broad,
    #         't_CSD': t_CSD,
    #         'x_CSD': x_CSD,
    #         'CSD': CSD}

    betaBurst = betaBurst_all[idx_beta]
    beta_dur = []
    beta_amp = []
    beta_num = betaBurst.shape[1]
    spike_time = []
    for i in range(1, len(Vm[0])):
        if (Vm[0][i]>0) & (Vm[0][i-1]<0):
            spike_time.append(cell.tvec[i])
    spike_time = np.asarray(spike_time)
    isi = np.diff(spike_time)

    for i in range(betaBurst.shape[1]):
        beta_amp.append(betaBurst[3, i])
        beta_dur.append((betaBurst[2, i] - betaBurst[0, i]) / Fs * 1000)
    beta_amp = np.asarray(beta_amp).flatten()
    beta_dur = np.asarray(beta_dur).flatten()

    datastat = {
        'beta_amp': beta_amp,
        'beta_dur': beta_dur,
        'beta_num': beta_num,
        'spike_time': spike_time,
        'spike_beta_cohe': vs1,
        'beta_theta_cohe': vs2,
        'spike_apicbeta_cohe': vs3,
        'isi': isi
    }

    sio.savemat(result_file[:-4] + '_stat.mat', datastat)
