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
import time

def create_tuft_inhibition_mod(cell):
    list_mod = []
    for i, s in enumerate(cell.GABA_meta):
        sec_name = s['sec_name']
        if ('apic' in sec_name) & (int(sec_name[5:-1])>36):
            list_mod.append(i)
    return list_mod

def create_higherbasal(rates, cell, amp = 2.5):
    list_mod = []
    for i, s in enumerate(cell.AMPA_meta):
        sec_name = s['sec_name']
        if ('dend' in sec_name):
            list_mod.append(i)
    for i in list_mod:
        rates[i] = rates[i] * amp
    return rates

# mod_freqs = [0,1,3,5,10,15,20,25,30,40,60]
mod_freqs = [0]
for ii in range(1):
    path_name = wd + '\\outputs\\uniform_I_30_E_8_apical1_5_basal1_2__%d' % ii
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    h('forall pop_section()')
    h('forall delete_section()')
    T = 3000  # simulation time (ms)
    dt = 0.1  # time step (ms)
    v_init = -75  # initial voltage (mV)
    seed = int(time.time())  # random seed

    P = parametersL5_Hay.init_params(wd)
    np.random.seed(seed)

    ### create cell ##
    cell = l5_neuron_model.L5Model(wd, P, verbool=False)

    ### electrode ###
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

    ### Run Simulation ###
    rates_e, a = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.8, 1)
    a, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 1, 1.5)
    # rates_e, a = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 2.3, 1)
    # a, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 3.4, 1.5)
    # rates_e = [np.repeat(0.008, P['N_e'])]
    # rates_i = [np.repeat(0.05, P['N_i'])]

    # [rates_e, rates_i] = pickle.load(open(wd + '\\outputs\\rate', 'rb'))
    rates = create_higherbasal(rates_e[0], cell, amp = 1.2)
    # S_i = sequences.build_rate_seq(rates_i[0], 0, T)
    mod_list = create_tuft_inhibition_mod(cell)
    pickle.dump([rates_e, rates_i], open(os.path.join(path_name, 'rate_%d' % ii), 'wb'))

    for mod_freq in mod_freqs:
        S_e = sequences.build_rate_seq(rates, 0, T)
        S_i = sequences.build_rate_seq_modulated(rates_i[0], 0, T, mod_freq = mod_freq, mod_list = mod_list, mod_amp = 1.5)
        result_file = 'i_mod_%d.mat' % mod_freq
        t, v = cell.simulate(T, dt, v_init, S_e, S_i, record_syn = True)
    
        lfp = M @ cell.imem
        lfp = lfp * 1e3  # unit in uV
        data = beta.analyze_beta(cell, electrode, lfp, os.path.join(path_name, result_file), if_plot = 0, if_save = 1)