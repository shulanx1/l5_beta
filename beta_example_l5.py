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

# for loop in range(1,6):
loop = 0

result_file = 'input085_percentage03_no_stochastic_%d.mat' % loop
h('forall pop_section()')
h('forall delete_section()')
T = 1000        # simulation time (ms)
dt = 0.1        # time step (ms)
v_init = -75    # initial voltage (mV)
seed = int(time.time())        # random seed

P = parametersL5_Hay.init_params(wd)
np.random.seed(seed)

### create cell ##
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

#%%### Run Simulation ###

rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.85, 1)
S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)
visualization.plot_nsg(cell,  electrode = electrode, if_plot_synapses = 0, sparse_plot = 8)
#%%
lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV

visualization.plot_nsg(cell, electrode = electrode)
visualization.plot_electrode_LFP_1D(lfp, cell, electrode, [0,1000], if_plot_morphology = 0)
data = beta.analyze_beta(cell, electrode, lfp, wd + '\\outputs\\control' + result_file, if_plot = 0, if_save = 1)
    
#%% ### blocked NMDA ###
result_file_noNMDA = 'noapicalNMDA' + result_file

h('forall pop_section()')
h('forall delete_section()')
cell = l5_neuron_model.L5Model(wd, P, verbool = True)
rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.85, 1)
S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
cell.set_deficit_NMDA(sec_name = 'apic', percentage = 0.5)
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV

data = beta.analyze_beta(cell, electrode, lfp, wd + '\\outputs\\' + result_file_noNMDA, if_plot = 0, if_save = 1)

### blocked Ca ###
h('forall pop_section()')
h('forall delete_section()')
cell = l5_neuron_model.L5Model(wd, P, verbool = True)

result_file_noCa = 'noapicCa' + result_file

# rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.85, 1)
# S_e = sequences.build_rate_seq(rates_e[0], 0, T)
# S_i = sequences.build_rate_seq(rates_i[0], 0, T)
cell.set_deficite_channels('Ca_HVA', sec_name = 'apic',  percentage = 0.0)
cell.set_deficite_channels('Ca_LVAst', sec_name = 'apic',  percentage = 0.0)
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV
data = beta.analyze_beta(cell, electrode, lfp, wd + '\\outputs\\' + result_file_noCa, if_plot = 0, if_save = 1)
### strong apical inhibition ###
h('forall pop_section()')
h('forall delete_section()')
cell = l5_neuron_model.L5Model(wd, P, verbool = True)
result_file_apicinh = 'apicinh' + result_file

# rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.85, 1)
for i, s in enumerate(cell.GABA_meta):
    if 'apic' in s["sec_name"]:
        rates_i[0][i] = rates_i[0][i]*5
# S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
t, v = cell.simulate(T, dt, v_init, S_e, S_i)
visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

lfp = M @ cell.imem
lfp = lfp*1e3 #unit in uV
data = beta.analyze_beta(cell, electrode, lfp, wd + '\\outputs\\' + result_file_apicinh, if_plot = 0, if_save = 1)