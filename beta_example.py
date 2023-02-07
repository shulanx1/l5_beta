#%%
import sys
import os
wd = 'C:\\work\\Code\\L5_beta' # working directory
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

h('forall pop_section()')
h('forall delete_section()')
T = 2000        # simulation time (ms)
dt = 0.1        # time step (ms)
v_init = -75    # initial voltage (mV)
seed = 1        # random seed

P = parametersL5_Hay.init_params(wd)
np.random.seed(seed)

### Create stimulation ###
# stim_dur = 300							# stimulus duration
# stim_on = 100							# stimulus on
# stim_off = stim_on + stim_dur           # stimulus off
# t_on = 0								# background on
# t_off = stim_on							# background off
# r_0 = 1.25								# background rate
# dt = 0.1            					# time step
# r_mean = 2.5
# num_patterns = 4
# input = 'opt'
# param_sets = {'rate':[40., 0, 0.], 'temp':[2.5, 1, 1.], 'opt':[20., 1, 1.]}
# r_max, num_t, s = param_sets[input]
# def init_input(P, num_patterns, stim_on, stim_off, r_mean, r_max, num_t, s):
#     """
#     Initialise input rates and spike time sequences for feature-binding task.
#
#     Parameters
#     ----------
#     P : dict
#         model parameters
#     num_patterns : int
#         number of input patterns to be classified
#     stim_on, stim_off : int
#         time of stimulus onset and termination (ms)
#     r_mean : float
#         average presynaptic population rate (Hz)
#     r_max : float
#         time averaged input rate to active synapses
#     num_t : int
#         number of precisely timed events per active synapse
#     s : float
#         interpolates between rate (s=0) and temporal (s=1) input signals (mostly
#         unused parameter -- to be removed)
#
#     Returns
#     -------
#     rates_e, rates_i : list
#         excitatory and inhibitory input rates for all patterns
#     S_E, S_I : list
#         times of precisely timed events for all patterns
#     """
#     N_e, N_i = P['N_e'], P['N_i']
#     ind_e = np.arange(N_e)
#     ind_i = np.arange(N_i)
#     np.random.shuffle(ind_e)
#     np.random.shuffle(ind_i)
#     rates_e, rates_i = sequences.assoc_rates(num_patterns, N_e, N_i, r_mean,
#                                              r_max)
#     rates_e = [r[ind_e] for r in rates_e]
#     rates_i = [r[ind_i] for r in rates_i]
#     if s > 0:
#         S_E, S_I = sequences.assoc_seqs(num_patterns, N_e, N_i, stim_on, stim_off,
#                                         num_t)
#         S_E = [s[ind_e] for s in S_E]
#         S_I = [s[ind_i] for s in S_I]
#         for s_e, r_e in zip(S_E, rates_e):
#             s_e[r_e == 0] = np.inf
#         for s_i, r_i in zip(S_I, rates_i):
#             s_i[r_i == 0] = np.inf
#     else:
#         S_E, S_I = sequences.build_seqs(num_patterns, N_e, N_i, stim_on, stim_off,
#                                         0)
#     return rates_e, rates_i, S_E, S_I
#
# def pad_S(S0):
#     l = np.max([len(s) for s in S0])
#     S = np.full((len(S0), l), np.inf)
#     for k, s in enumerate(S0):
#         S[k, :len(s)] = s
#     return S
#
# jitter = 2.5
# if num_t > 0:
#     sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
# else:
#     sigma = jitter
# rates_e, rates_i, S_E, S_I, = init_input(P, num_patterns, stim_on, stim_off,
#                                         r_mean, r_max, num_t, s)
# pre_syn_e = sequences.PreSyn(r_0, sigma)
# pre_syn_i = sequences.PreSyn(r_0, sigma)
# S_e = [pre_syn_e.spike_train(t_on, t_off, stim_on, stim_off, s,
#                              rates_e[0][k], S_E[0][k]) for k in range(len(rates_e[0]))]
# S_i = [pre_syn_i.spike_train(t_on, t_off, stim_on, stim_off, s,
#                              rates_i[0][k], S_I[0][k]) for k in range(len(rates_i[0]))]
# S_e = pad_S(S_e)
# S_i = pad_S(S_i)

### Run Simulation ###
rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0.5, 1)
S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
cell = l5_neuron_model.L5Model(wd, P, verbool = True)
t, v = cell.simulate(T, dt, v_init, S_e, S_i)

visualization.plot_Vm_traces(cell, ['soma[0]', 'apic[36]'], [0,T], if_plot = 1)

### extracellular LFP ###
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
lfp = M @ cell.imem


visualization.plot_nsg(cell, electrode = electrode)
visualization.plot_electrode_LFP_1D(lfp, cell, electrode, [0,T], if_plot_morphology = 0)

data = {'t':t, 'vm': v, 'lfp': lfp}
import scipy.io as sio
sio.savemat(wd + '\\outputs\\rand_input_1cell_2000ms.mat',data)