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

file_name = 'apicinhinput085_percentage03_no_stochastic_0'
result_file = wd + '\\outputs\\' + file_name + '.mat'
data = sio.loadmat(result_file)

t = data['t']
Fs = data['Fs']
vm = data['vm']
im = data['im']
lfp = data['lfp']
lfp_beta = data['lfp_beta']
betaBurst = data['betaBurst']
LFP_beta_narrow = data['LFP_beta_narrow']
LFP_beta_broad = data['LFP_beta_broad']
t_CSD = data['t_CSD']
x_CSD = data['x_CSD']
CSD = data['CSD']

h('forall pop_section()')
h('forall delete_section()')

P = parametersL5_Hay.init_params(wd)
cell = l5_neuron_model.L5Model(wd, P, verbool = True)
cell.dt = 0.1
cell.vmem = vm
cell.imem = im
cell.tvec = t.flatten()

# visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,1000], if_plot = 1)
visualization.plot_beta_event(lfp, lfp_beta, 13, cell, [betaBurst], T_range = [0,1000])
# visualization.plot_aligned_beta(LFP_beta_narrow, t_CSD.flatten())
# visualization.plot_CSD(CSD, t_CSD.flatten(), x_CSD.flatten(), c_range = [-2e5, 2e5])

LFP_beta_narrow[13,:] = np.mean(LFP_beta_narrow[[12,14],:], axis = 0)
visualization.plot_aligned_beta(LFP_beta_narrow, t_CSD.flatten())
[CSD, x_CSD] = beta.customCSD(LFP_beta_narrow, t_CSD.flatten(), 20.0, if_plot=1, smooth=1)

beta_dur = []
beta_amp = []
beta_num = betaBurst.shape[1]

for i in range(betaBurst.shape[1]):
    beta_amp.append(betaBurst[3,i])
    beta_dur.append((betaBurst[2,i]-betaBurst[0,i])/Fs.flatten()*1000)
beta_amp = np.asarray(beta_amp).flatten()
beta_dur = np.asarray(beta_dur).flatten()

# datastat = {
#     'beta_amp':beta_amp,
#     'beta_dur':beta_dur,
#     'beta_num':beta_num
# }
# sio.savemat(result_file[:-4] + '_stat.mat', datastat)


# lfp_low = beta.lowpass_filter_lfp(lfp, Fs.flatten(), Fc=500.)
# [lfp_beta, power] = beta.bandFilter(lfp_low, Fs.flatten())