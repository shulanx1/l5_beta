"""
Parameters for simulation and training with morphology 1.
"""
#%%
import numpy as np
import os
import neuron
import json
from neuron import h

def init_params(wd):
    """ Create dictionary of model parameters """
    # neuron.load_mechanisms(wd + "\\mod")
    # neuron.load_mechanisms(wd + "\\mod_Gao2020")
    neuron.load_mechanisms(wd + "\\mod_stochastic")
    param_file = wd + "\\input\\biophys4.json"

    f = open(param_file)
    data = json.load(f)
    h.celsius = data['conditions'][0]['celsius']
    passive = data['passive'][0]
    genome = data['genome']
    conditions = data['conditions'][0]

    tree = wd + '\\input\\cell1.asc'
    if_stochastic = True
    stochastic_channel = ['na', 'K_Tst', 'NaTs2_t', 'K_Pst', 'Nap_Et2', 'NaTa_t_2F']
    N_e = 1600  # number of excitatory synapses
    N_i = 400  # number of inhibitory synapses
    soma = [0]
    basal = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]
    oblique = list(np.asarray([4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,21,22,23,24,25,27,28,29,30,31,32,33,35])+87)
    apical = np.asarray([0,1,2,3,14,20,26,34,36])+87
    apical = list(np.concatenate((apical,np.arange(37,109)+87), axis=None))
    locs_e = ['basal', 'apical']
    locs_i = ['basal', 'apical']
    l_seg = 10  # maximum segment size (um)
    c_m = 1  # somatic specific capacitance (uF/cm^2)
    c_m_d = 2    # dendritic specific capacitance (uF/cm^2)
    R_a = passive['ra'] * 1e-3  # axial resistivity (k ohm cm)
    tau_m = 10. # membrane time constant (ms)
    E_r = passive["e_pas"]  # resting/leak reversal potential (mv)
    E_e = 0.  # excitatory reversal potential (mv)
    E_i = -80.  # inhibitory reversal potential (mv)
    tauA = np.array([0.1, 2.])  # AMPA synapse rise and decay time (ms)
    g_max_A = 0.25 * 1e-3 #0.4 * 1e-3  # AMPA conductance (uS)
    tauN = np.array([2., 75.])  # NMDA synapse rise and decay time (ms)
    g_max_N = 0.5 * 1e-3 #0.8 * 1e-3  # NMDA conductance (uS)
    tauG = np.array([1., 5.])  # GABA synapse rise and decay time (ms)
    g_max_G = 1.0 * 1e-3 # 1.6 * 1e-3  # GABA conductance (uS)
    r_na = 2.  # NMDA/AMPA ratio
    E_na = 50.  # Na reversal potential (mV)
    E_k = -85.  # K reversal potential (mV)
    E_hcn = -45. # HCN reversal potential (mV)
    v_th = -55.  # Traub and Miles threshold parameter (mV)
    t_max = 0.2e3  # slow K adaptation time scale (ms)
    active_d = True  # active or passive dendrites
    active_n = True  # active or passive NMDA receptors

    P = {'tree': tree,
         'N_e': N_e,
         'N_i': N_i,
         'soma': soma,
         'basal': basal,
         'oblique': oblique,
         'apical': apical,
         'locs_e': locs_e,
         'locs_i': locs_i,
         'l_seg': l_seg,
         'c_m': c_m,
         'c_m_d': c_m_d,
         'R_a': R_a,
         'tau_m': tau_m,
         'E_r': E_r,
         'E_e': E_e,
         'E_i': E_i,
         'tauA': tauA,
         'tauN': tauN,
         'tauG': tauG,
         'g_max_A': g_max_A,
         'g_max_N': g_max_N,
         'g_max_G': g_max_G,
         'r_na': r_na,
         'E_na': E_na,
         'E_k': E_k,
         'E_hcn': E_hcn,
         'data': data,
         'param_file': param_file,
         'if_stochastic': if_stochastic,
         'stochastic_channel': stochastic_channel,
         'v_th': v_th,
         't_max': t_max,
         'active_d': active_d,
         'active_n': active_n}
    return P
