#%%
import sys
import os
wd = 'C:\\work\\Code\\l5_beta' # working directory
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

result_file = 'rand_input_no_stochastic_10s.mat'
h('forall pop_section()')
h('forall delete_section()')
T = 10000        # simulation time (ms)
dt = 0.1        # time step (ms)
v_init = -75    # initial voltage (mV)
seed = 1        # random seed

P = parametersL5_Hay.init_params(wd)
np.random.seed(seed)

### create cell ##
rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.5, 1)
S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
cell = l5_neuron_model.L5Model(wd, P, verbool = True)

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
M = electrode.calc_mapping(cell)   # use M = elestrode.get_transformation_matrix(cell) for newer version of LFPy

#%% ### Run Simulation ###
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV

visualization.plot_nsg(cell, electrode = electrode)
visualization.plot_electrode_LFP_1D(lfp, cell, electrode, [0,T], if_plot_morphology = 0)

Fs = 1/(cell.dt/1000.)
lfp_low = beta.lowpass_filter_lfp(lfp, Fs, Fc = 500.)
[lfp_beta, power] = beta.bandFilter(lfp_low)
betaBurst_all = beta.betaBurstDetection(Fs, lfp_beta,channel = None)
visualization.plot_beta_event(lfp, lfp_beta, np.arange(0,64,10), cell, betaBurst_all)
data = {'t':t, 'Fs': Fs, 'vm': cell.vmem, 'im': cell.imem, 'lfp': lfp, 'lfp_beta': lfp_beta, 'betaBurst': betaBurst_all[13]}
import scipy.io as sio
sio.savemat(wd + '\\outputs\\' + result_file,data)

#%% ## blocked NMDA ###
result_file_noNMDA = 'noapicalNMDA' + result_file
cell.set_deficit_NMDA(sec_name = 'apic')
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV

Fs = 1/(cell.dt/1000.)
lfp_low = beta.lowpass_filter_lfp(lfp, Fs, Fc = 500.)
[lfp_beta, power] = beta.bandFilter(lfp_low)
betaBurst_all = beta.betaBurstDetection(Fs, lfp_beta,channel = None)
visualization.plot_beta_event(lfp, lfp_beta, np.arange(0,64,10), cell, betaBurst_all)
data = {'t':t, 'Fs': Fs, 'vm': cell.vmem, 'im': cell.imem, 'lfp': lfp, 'lfp_beta': lfp_beta, 'betaBurst': betaBurst_all[13]}
import scipy.io as sio
sio.savemat(wd + '\\outputs\\' + result_file_noNMDA,data)

#%% ## blocked Ca ###
result_file_noCa = 'noapicCa' + result_file
cell.set_deficite_channels('Ca_HVA', sec_name = 'apic',  percentage = 0)
cell.set_deficite_channels('Ca_LVAst', sec_name = 'apic',  percentage = 0)
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV

Fs = 1/(cell.dt/1000.)
lfp_low = beta.lowpass_filter_lfp(lfp, Fs, Fc = 500.)
[lfp_beta, power] = beta.bandFilter(lfp_low)
betaBurst_all = beta.betaBurstDetection(Fs, lfp_beta,channel = None)
visualization.plot_beta_event(lfp, lfp_beta, np.arange(0,64,10), cell, betaBurst_all)
data = {'t':t, 'Fs': Fs, 'vm': cell.vmem, 'im': cell.imem, 'lfp': lfp, 'lfp_beta': lfp_beta, 'betaBurst': betaBurst_all[13]}
import scipy.io as sio
sio.savemat(wd + '\\outputs\\' + result_file_noCa,data)