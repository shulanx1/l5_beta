import LFPy
import os
from neuron import h
import numpy as np
import neuron
import statistics
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
from scipy.integrate import odeint
from scipy import signal
import matplotlib.transforms as mtransforms
# from neurom import viewer
# import neurom

def plot_Vm_traces(cell, sections, time_range, if_plot = 0):
    """
    plot the Vm trace in given sections
    :param cell:
    :param sections: str or list of str
    :param time_range: list, [start,  stop] in ms
    :return: vm: np.array, section * time
    """
    if if_plot:
        plt.figure()
    vm = []
    if isinstance(sections, list):
        for sec in sections:
            idx = cell.get_idx(sec)
            vm_sec = np.mean(cell.vmem[idx, :], axis = 0)
            vm.append(vm_sec)
            if if_plot:
                plt.plot(cell.tvec, vm_sec)
    else:
        sec = sections
        idx = cell.get_idx(sec)
        vm_sec = np.mean(cell.vmem[idx, :], axis=0)
        vm.append(vm_sec)
        if if_plot:
            plt.plot(cell.tvec, vm_sec)

    vm = np.asarray(vm)
    if if_plot:
        plt.xlim(time_range[0], time_range[1])
        plt.xlabel('Time (ms)')
        plt.ylabel('V_m (mV)')
        plt.legend(sections)
        plt.show()

    return vm

def plot_nsg(cell,  electrode = None, if_plot_synapses = 1, sparse_plot = 8):
    fig1 = plt.figure()
    max_size = 3
    min_size = 1
    color_AMPA = np.asarray([251,176,59])/256
    color_NMDA = np.asarray([240,90,36])/256
    color_GABA = np.asarray([27,117,187])/256
    color_electrode = np.asarray([166,124,82])/256
    W_AMPA = cell.w_1 / max(cell.w_1)
    W_NMDA = cell.w_2 / max(cell.w_2)
    W_GABA = cell.w_3 / max(cell.w_3)
    # show morphology, synapse and electrode location on x-z plane
    ax = fig1.add_axes([0.1, 0.05, 0.4, 0.9])
    for secname in cell.allsecnames:
        idx = cell.get_idx(secname)
        if idx.size == 0:
            continue
        else:
            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.ystart[idx], cell.yend[idx][-1]],
                    color='k')
    if if_plot_synapses:
        for k, s in enumerate(cell.AMPA_meta):
            if (k % sparse_plot) == 0:
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"] - 1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=color_AMPA, marker='^', markersize=min_size + (max_size - min_size) * W_AMPA[k])
        for k, s in enumerate(cell.NMDA_meta):
            if (k % sparse_plot) == 1:
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"] - 1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=color_NMDA, marker='^', markersize=min_size + (max_size - min_size) * W_NMDA[k])
        for k, s in enumerate(cell.GABA_meta):
            if (k % sparse_plot) == 0:
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"] - 1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=color_GABA, marker='.', markersize=min_size + (max_size - min_size) * W_GABA[k])

    if electrode is not None:
        ax.scatter(electrode.x, electrode.y,
                 c=electrode.y[0]-electrode.y,cmap = 'viridis', marker='o', s=20)

    ax.set_xticks([])
    ax.set_yticks([])

    # show morphology, synapse and electrode location on y-z plane
    ax = fig1.add_axes([0.5, 0.05, 0.2, 0.9])
    for secname in cell.allsecnames:
        idx = cell.get_idx(secname)
        if idx.size == 0:
            continue
        else:
            ax.plot(np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    np.r_[cell.ystart[idx], cell.yend[idx][-1]],
                    color='k')
        if if_plot_synapses:
            for k, s in enumerate(cell.AMPA_meta):
                if (k % sparse_plot) == 0:
                    idx = cell.get_idx(s["sec_name"])[s["seg_idx"] - 1]
                    ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                            color=color_AMPA, marker='^', markersize=min_size + (max_size - min_size) * W_AMPA[k])
            for k, s in enumerate(cell.NMDA_meta):
                if (k % sparse_plot) == 1:
                    idx = cell.get_idx(s["sec_name"])[s["seg_idx"] - 1]
                    ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                            color=color_NMDA, marker='^', markersize=min_size + (max_size - min_size) * W_NMDA[k])
            for k, s in enumerate(cell.GABA_meta):
                if (k % sparse_plot) == 0:
                    idx = cell.get_idx(s["sec_name"])[s["seg_idx"] - 1]
                    ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                            color=color_GABA, marker='^', markersize=min_size + (max_size - min_size) * W_GABA[k])

    if electrode is not None:
        ax.scatter(electrode.z, electrode.y,
                c=electrode.y[0]-electrode.y,cmap = 'viridis',  marker='o', s=20)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def plot_nsg_weight(cell,  W_e, W_i, rates_e, rates_i, sparse_plot = 4):
    fig1 = plt.figure()
    max_size = 10
    min_size = 1
    W_e = W_e/max(W_e)

    # show morphology, synapse and electrode location on x-z plane
    syn_color1 = np.asarray([0,146,69])/255
    syn_color2 = np.asarray([240,90,36])/255
    syn_colorg = np.asarray([180,180,180])/255
    ax = fig1.add_axes([0.1, 0.05, 0.4, 0.9])
    for secname in cell.allsecnames:
        idx = cell.get_idx(secname)
        if idx.size == 0:
            continue
        else:
            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.ystart[idx], cell.yend[idx][-1]],
                    color='k')
    for k, s in enumerate(cell.AMPA_meta):
        if (k%sparse_plot)==0:
            rate0 = rates_e[0][k]
            rate1 = rates_e[1][k]
            rate2 = rates_e[2][k]
            rate3 = rates_e[3][k]
            if (abs(rate0)>1e-4)and(abs(rate1)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=syn_colorg, marker='s', markersize = min_size + (max_size-min_size)*W_e[k])
            if (abs(rate0)>1e-4)and(abs(rate2)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=syn_color1, marker='o', markersize = min_size + (max_size-min_size)*W_e[k])
            if (abs(rate1)>1e-4)and(abs(rate3)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=syn_color2, marker='o', markersize = min_size + (max_size-min_size)*W_e[k])
            if (abs(rate2)>1e-4)and(abs(rate3)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.xmid[idx]], [cell.ymid[idx]],
                        color=syn_colorg, marker='^', markersize = min_size + (max_size-min_size)*W_e[k])



    ax.set_xticks([])
    ax.set_yticks([])

    # show morphology, synapse and electrode location on y-z plane
    ax = fig1.add_axes([0.5, 0.05, 0.4, 0.9])
    for secname in cell.allsecnames:
        idx = cell.get_idx(secname)
        if idx.size == 0:
            continue
        else:
            ax.plot(np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    np.r_[cell.ystart[idx], cell.yend[idx][-1]],
                    color='k')
    for k, s in enumerate(cell.AMPA_meta):
        if (k%sparse_plot)==0:
            rate0 = rates_e[0][k]
            rate1 = rates_e[1][k]
            rate2 = rates_e[2][k]
            rate3 = rates_e[3][k]
            if (abs(rate0)>1e-4)and(abs(rate1)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                        color=syn_colorg, marker='s', markersize = min_size + (max_size-min_size)*W_e[k], linestyle = 'None')
            if (abs(rate0)>1e-4)and(abs(rate2)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                        color=syn_color1, marker='o', markersize = min_size + (max_size-min_size)*W_e[k], linestyle = 'None')
            if (abs(rate1)>1e-4)and(abs(rate3)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                        color=syn_color2, marker='o', markersize = min_size + (max_size-min_size)*W_e[k], linestyle = 'None')
            if (abs(rate2)>1e-4)and(abs(rate3)>1e-4):
                idx = cell.get_idx(s["sec_name"])[s["seg_idx"]-1]
                ax.plot([cell.zmid[idx]], [cell.ymid[idx]],
                        color=syn_colorg, marker='^', markersize = min_size + (max_size-min_size)*W_e[k], linestyle = 'None')

    ax.set_xticks([])
    ax.set_yticks([])

def plot_input_output(V, rates_e, rates_i, S_e_all, S_i_all, t,W_e, W_i, rep_plot = 2, sparse_plot = [4, 40]):
    # sparse_plot: how every 4 synapses, 40 shown synapses per row
    max_size = 10
    min_size = 1
    W_e = W_e/max(W_e)
    W_i = W_i / max(W_i)
    syn_color1 = np.asarray([0,146,69])/255
    syn_color2 = np.asarray([240,90,36])/255
    syn_colorg = np.asarray([180,180,180])/255
    plt.figure()
    for i in range(min(rep_plot,len(V[0]))):
        for j in range(len(V)):
            y_start = -80
            plt.plot(t+t[-1]*(i*len(V)+j), V[j][i], 'k')
            for k in range(len(rates_e[0])):
                if len(S_e_all[j][i][k]>0) and ((k%sparse_plot[0])==0):
                    rate0 = rates_e[0][k]
                    rate1 = rates_e[1][k]
                    rate2 = rates_e[2][k]
                    rate3 = rates_e[3][k]
                    if (abs(rate0) > 1e-4) and (abs(rate1) > 1e-4):
                        plt.plot(S_e_all[j][i][k]+t[-1]*(i*len(V)+j), y_start*np.ones([len(S_e_all[j][i][k]),]), color=syn_colorg, marker='s', markersize = min_size + (max_size-min_size)*W_e[k], linestyle = 'None')
                    if (abs(rate0) > 1e-4) and (abs(rate2) > 1e-4):
                        plt.plot(S_e_all[j][i][k] + t[-1] * (i * len(V) + j), y_start*np.ones([len(S_e_all[j][i][k]),]), color=syn_color1, marker='o',linestyle = 'None',
                                 markersize=min_size + (max_size - min_size) * W_e[k])
                    if (abs(rate1) > 1e-4) and (abs(rate3) > 1e-4):
                        plt.plot(S_e_all[j][i][k] + t[-1] * (i * len(V) + j), y_start*np.ones([len(S_e_all[j][i][k]),]), color=syn_color2, marker='o',linestyle = 'None',
                                 markersize=min_size + (max_size - min_size) * W_e[k])
                    if (abs(rate2) > 1e-4) and (abs(rate3) > 1e-4):
                        plt.plot(S_e_all[j][i][k] + t[-1] * (i * len(V) + j), y_start*np.ones([len(S_e_all[j][i][k]),]), color=syn_colorg, marker='^',linestyle = 'None',
                                 markersize=min_size + (max_size - min_size) * W_e[k])
                if (k % (sparse_plot[1]*sparse_plot[0])) == 0:
                    y_start = y_start - 10
            y_start = y_start - 20
            for k in range(len(rates_i[0])):
                if len(S_i_all[j][i][k] > 0) and ((k%sparse_plot[0])==0):
                    rate0 = rates_i[0][k]
                    rate1 = rates_i[1][k]
                    rate2 = rates_i[2][k]
                    rate3 = rates_i[3][k]
                    if (abs(rate0) > 1e-4) and (abs(rate1) > 1e-4):
                        plt.plot(S_i_all[j][i][k]+t[-1]*(i*len(V)+j), y_start*np.ones([len(S_i_all[j][i][k]),]), color=syn_colorg, marker='s', markersize = min_size + (max_size-min_size)*W_i[k], linestyle = 'None')
                    if (abs(rate0) > 1e-4) and (abs(rate2) > 1e-4):
                        plt.plot(S_i_all[j][i][k] + t[-1] * (i * len(V) + j), y_start*np.ones([len(S_i_all[j][i][k]),]), color=syn_color1, marker='o',linestyle = 'None',
                                 markersize=min_size + (max_size - min_size) * W_i[k])
                    if (abs(rate1) > 1e-4) and (abs(rate3) > 1e-4):
                        plt.plot(S_i_all[j][i][k] + t[-1] * (i * len(V) + j), y_start*np.ones([len(S_i_all[j][i][k]),]), color=syn_color2, marker='o',linestyle = 'None',
                                 markersize=min_size + (max_size - min_size) * W_i[k])
                    if (abs(rate2) > 1e-4) and (abs(rate3) > 1e-4):
                        plt.plot(S_i_all[j][i][k] + t[-1] * (i * len(V) + j), y_start*np.ones([len(S_i_all[j][i][k]),]), color=syn_colorg, marker='^',linestyle = 'None',
                                 markersize=min_size + (max_size - min_size) * W_i[k])
                if (k % (sparse_plot[1]*sparse_plot[0])) == 0:
                    y_start = y_start - 5
    plt.show()


def plot_electrode_LFP_1D(LFP, cell, electrode, time_range, if_plot_morphology = 0):
    """
    plot the trace of electrode grid
    :param LFP_class:
    :param time_range:
    :param if_plot_morphology:
    :return:
    """

    v = LFP*1000 # in uV
    maxv = []
    for i in range(len(LFP)):
        maxv.append(max(abs(v[i, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)])))
    maxima = np.nanmax(maxv)
    minima = np.nanmin(maxv)
    norm_color = matplotlib.colors.Normalize(vmin=minima, vmax=maxima/5, clip=True)
    mapper = cm.ScalarMappable(norm=norm_color, cmap=cm.viridis)

    if if_plot_morphology:
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.plot(electrode.x, electrode.y, marker = 'o', color = 'g', markersize = 2, zorder = 0, linestyle='None')
        for i in range(len(LFP)):
            line = np.asarray(list(zip(cell.tvec[int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)] + electrode.x[i] +2 - time_range[0],
                        v[i,int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)]/max(abs(v[i, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)]))*8 + electrode.y[i])))
            line_segments = LineCollection([line],
                                           linewidths=(1.5),
                                           linestyles='solid',
                                           color = mapper.to_rgba(max(abs(v[i, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)]))),
                                           zorder=1,
                                           rasterized=False)
            ax.add_collection(line_segments)
        for secname in cell.allsecnames:
            idx = cell.get_idx(secname)
            if idx.size == 0:
                continue
            else:
                ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                        np.r_[cell.ystart[idx], cell.yend[idx][-1]],
                        color='k')
        ax.set_xlim([min(electrode.x)-200, max(electrode.x)+200])
        ax.set_ylim([min(electrode.y)-20, max(electrode.y)+20])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.colorbar(mapper, ax=ax)

    else:
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        for i in range(v.shape[0]):
            ax.plot(cell.tvec, v[i, :]/(max(abs(v[i, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)])))-0.5*i, color=mapper.to_rgba(max(abs(v[i, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)]))))
        ax.axis('off')
        ax.set_xlim(time_range[0], time_range[1])
        cbar = plt.colorbar(mapper, ax=ax)
        cbar.set_label('LFP (uV)', rotation=270)
    plt.show()

def plot_beta_event(lfp, beta_lfp, channel, cell, betaBurst, T_range = None):
    if isinstance(channel, int) or isinstance(channel, float):
        channel = [channel]
    t = np.arange(beta_lfp.shape[1])*cell.dt
    fig, axs = plt.subplots(1+len(channel), 1)
    axs[0].plot(t, cell.vmem[0])
    # axs[0].plot(t, cell.vmem[int(np.round(np.median(cell.get_idx('apic[36]'))))])  L5
    # axs[0].plot(t, cell.vmem[int(np.round(np.median(cell.get_idx('apic[60]'))))])  L5
    axs[0].plot(t, cell.vmem[int(np.round(np.median(cell.get_idx('apic[10]'))))])
    axs[0].plot(t, cell.vmem[int(np.round(np.median(cell.get_idx('apic[11]'))))])
    if T_range == None:
        axs[0].set_xlim(t[0], t[-1])
    else:
        axs[0].set_xlim(T_range[0], T_range[1])
    for i, ax in enumerate(axs[1:]):
        # ax.plot(t, lfp[i,:])
        ax.plot(t, beta_lfp[i,:])
        yplot = np.linspace(np.min(lfp[i,:]), np.max(lfp[i,:]), 100)
        if len(betaBurst)>len(channel):
            for j in range(betaBurst[channel[i]].shape[1]):
                ax.axvspan(t[int(betaBurst[channel[i]][0,j])],t[int(betaBurst[channel[i]][2,j])], color = 'g', alpha = 0.5 )
        else:
            for j in range(betaBurst[i].shape[1]):
                ax.axvspan(t[int(betaBurst[i][0,j])],t[int(betaBurst[i][2,j])], color = 'g', alpha = 0.5)
        if T_range == None:
            ax.set_xlim(t[0], t[-1])
        else:
            ax.set_xlim(T_range[0], T_range[1])
    plt.show()

def plot_aligned_beta(LFP, t):
    plt.figure()
    norm_color = matplotlib.colors.Normalize(vmin=0.0, vmax=63.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm_color, cmap=cm.viridis)
    for i in range(LFP.shape[0]):
        # plt.plot(t, LFP[i, :] - np.max(np.max(LFP)) / 5 * i, color=mapper.to_rgba(i))
        plt.plot(t, LFP[i, :] - 100 * i, color=mapper.to_rgba(i))
    plt.xlim([-0.05, 0.05])
    # plt.ylim([-5000, 500])
    plt.show()

def plot_CSD(CSD, t, x, c_range = [-2e5, 2e5]):
    plt.figure()
    plt.imshow(CSD, origin='upper', cmap='gray', extent=[t[0] * 1e3, t[-1] * 1e3, x[-1], x[0]], aspect=1 / 5, vmin=c_range[0], vmax=c_range[1])
    plt.colorbar()
    plt.xlim([-50, 50])
    plt.show()

# def plot_nsg(cell, synapses, electrode, time_range):
#     fig1 = plt.figure()
#     # plot somatic trace
#     ax = fig1.add_axes([0.1, 0.55, 0.5, 0.4])
#     ax.plot(cell.tvec, cell.somav)
#     ax.set_xlim(left=time_range[0], right=time_range[1])
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel('somatic Vm [mV]')

#     # plot synaptic current
#     ax = fig1.add_axes([0.1, 0.05, 0.5, 0.4])
#     for s in synapses:
#         if s.kwargs['name'] == 'NMDA':
#             syn_color = 'm'
#         elif s.kwargs['name'] == 'AMPA':
#             syn_color = 'r'
#         elif s.kwargs['name'] == 'GABA_A':
#             syn_color = 'k'
#         else:
#             syn_color = 'b'
#         ax.plot(cell.tvec, s.i, color=syn_color)
#     ax.set_xlim(left=time_range[0], right=time_range[1])
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel('Syn current [nA]')

#     # show morphology, synapse and electrode location on x-z plane
#     ax = fig1.add_axes([0.55, 0.05, 0.2, 0.9])
#     for secname in cell.allsecnames:
#         idx = cell.get_idx(secname)
#         if idx.size == 0:
#             continue
#         else:
#             ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
#                     np.r_[cell.ystart[idx], cell.yend[idx][-1]],
#                     color='k')
#     for s in synapses:
#         if s.kwargs['name'] == 'NMDA':
#             syn_color = 'm'
#         elif s.kwargs['name'] == 'AMPA':
#             syn_color = 'r'
#         elif s.kwargs['name'] == 'GABA_A':
#             syn_color = 'k'
#         else:
#             syn_color = 'b'
#         ax.plot([s.x], [s.y],
#                 color=syn_color, marker='.')
#     for i in range(electrode.x.size):
#         ax.plot(electrode.x[i], electrode.y[i], color='g', marker='o')
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # show morphology, synapse and electrode location on y-z plane
#     ax = fig1.add_axes([0.75, 0.05, 0.2, 0.9])
#     for secname in cell.allsecnames:
#         idx = cell.get_idx(secname)
#         if idx.size == 0:
#             continue
#         else:
#             ax.plot(np.r_[cell.zstart[idx], cell.zend[idx][-1]],
#                     np.r_[cell.ystart[idx], cell.yend[idx][-1]],
#                     color='k')
#     for s in synapses:
#         if s.kwargs['name'] == 'NMDA':
#             syn_color = 'm'
#         elif s.kwargs['name'] == 'AMPA':
#             syn_color = 'r'
#         elif s.kwargs['name'] == 'GABA_A':
#             syn_color = 'k'
#         else:
#             syn_color = 'b'
#         ax.plot([s.z], [s.y],
#                 color=syn_color, marker='.')
#     for i in range(electrode.x.size):
#         ax.plot(electrode.z[i], electrode.y[i], color='g', marker='o')
#     ax.set_xticks([])
#     ax.set_yticks([])

# def plot_Vm_traces(cell, sections, time_range, if_plot = 0):
#     """
#     plot the Vm trace in given sections
#     :param cell:
#     :param sections: str or list of str
#     :param time_range: list, [start,  stop] in ms
#     :return: vm: np.array, section * time
#     """
#     if if_plot:
#         plt.figure()
#     vm = []
#     if isinstance(sections, list):
#         for sec in sections:
#             idx = cell.get_idx(sec)
#             vm_sec = np.mean(cell.vmem[idx, :], axis = 0)
#             vm.append(vm_sec)
#             if if_plot:
#                 plt.plot(cell.tvec, vm_sec)
#     else:
#         sec = sections
#         idx = cell.get_idx(sec)
#         vm_sec = np.mean(cell.vmem[idx, :], axis=0)
#         vm.append(vm_sec)
#         if if_plot:
#             plt.plot(cell.tvec, vm_sec)

#     vm = np.asarray(vm)
#     if if_plot:
#         plt.xlim(time_range[0], time_range[1])
#         plt.xlabel('Time (ms)')
#         plt.ylabel('V_m (mV)')
#         plt.legend(sections)

#     return vm

# def plot_Cai_traces(cell, sections, time_range, if_plot = 0):
#     """
#     plot the Vm trace in given sections
#     :param cell:
#     :param sections: str or list of str
#     :param time_range: list, [start,  stop] in ms

#     """
#     plt.figure()
#     cai = []
#     if isinstance(sections, list):
#         for sec in sections:
#             idx = cell.get_idx(sec)
#             cai_sec = np.mean(cell.rec_variables["cai"][idx, :], axis = 0)
#             cai.append(cai_sec)
#             if if_plot:
#                 plt.plot(cell.tvec, cai_sec)
#     else:
#         sec = sections
#         idx = cell.get_idx(sec)
#         cai_sec = np.mean(cell.rec_variables["cai"][idx, :], axis=0)
#         cai.append(cai_sec)
#         if if_plot:
#             plt.plot(cell.tvec, cai_sec)

#     cai = np.asarray(cai)
#     if if_plot:
#         plt.xlim(time_range[0], time_range[1])
#         plt.xlabel('Time (ms)')
#         plt.ylabel('intracellular Ca (mM)')
#         plt.legend(sections)

#     return cai


# def Vm_dynamic(cell, wd, time_range, if_save=False):
#     """
#     visualize the Vm dynapmic as color coded

#     Parameters
#     ----------
#     cell : LFPy.Cell
#     time_range: list
#         time_range[0] is the start time and time_Range[1] is the stop time, in ms
#     if_save : bool, optional
#         Save the animation. The default is False.

#     Returns
#     -------
#     None.

#     """

#     # color mapping
#     Vm = cell.vmem
#     maxima = np.nanmax(Vm[:, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)])
#     minima = np.nanmin(Vm[:, int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)])
#     norm_color = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
#     mapper = cm.ScalarMappable(norm=norm_color, cmap=cm.viridis)

#     idx_start = int(time_range[0]/cell.dt)
#     idx_stop = int(time_range[1]/cell.dt)
#     # for i in range(idx_start, idx_stop, 40):
#     #     plt.figure()
#     #     for j in range(cell.totnsegs):
#     #         im = plt.plot(np.r_[cell.xstart[j], cell.xend[j]],
#     #                       np.r_[cell.ystart[j], cell.yend[j]],
#     #                       color=mapper.to_rgba(Vm[j, int(i)]))
#     #         plt.hold = True
#     #     plt.hold = False
#     #     plt.colorbar(mapper)
#     #     # plt.pause(0.5)
#     #     ims.append(im)
#     #     del im
#     fig = plt.figure(figsize=(5, 10))
#     ax1 = fig.add_axes([0.05, 0.05, 0.7, 0.9])
#     ax1.set_xlim([-200, 200])
#     ax1.set_ylim([-250, 1150])
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.axis('off')
#     plt.colorbar(mapper)
#     ims_list = []
#     for i in range(idx_start, idx_stop, int(1/cell.dt)):
#         ims_frame = []
#         for sec, name in zip(cell.get_pt3d_polygons(projection=('x', 'y')), cell.allsecnames):
#             Vm_sec = np.mean(Vm[cell.get_idx(name), int(i)], axis = 0)
#             color = mapper.to_rgba(Vm_sec)
#             polycol = PolyCollection([np.transpose(np.asarray(sec))], edgecolors='none',
#                                      facecolors=color, zorder=-1, rasterized=False)
#             im = ax1.add_collection(polycol)
#             ims_frame.append(im)
#         ims_list.append(ims_frame)

#     ani = animation.ArtistAnimation(fig = fig, artists = ims_list, interval=100, blit=True, repeat = True, repeat_delay = 1000)
#     # wd = 'C:\\work\\Code\\neuron-l5pn-model'
#     if if_save:
#         ani.save(os.path.join(wd, 'Vm_dynapmic.gif'), writer = 'imagemagick', fps = 10)


# def plot_electrode_LFP(LFP_class, time_range, if_plot_morphology = 0):
#     """
#     plot the trace of electrode grid
#     :param LFP_class:
#     :param time_range:
#     :param if_plot_morphology:
#     :return:
#     """
#     electrode = LFP_class.electrode
#     cell = LFP_class.cell
#     v = electrode.LFP*1000 # in uV
#     maxv = []
#     for i in range(100):
#         maxv.append(max(abs(v[i, int(time_range[0]/LFP_class.cell.dt):int(time_range[1]/LFP_class.cell.dt)])))
#     maxima = np.nanmax(maxv)
#     minima = np.nanmin(maxv)
#     norm_color = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
#     mapper = cm.ScalarMappable(norm=norm_color, cmap=cm.viridis)

#     if if_plot_morphology:
#         fig = plt.figure()
#         ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
#         ax.plot(electrode.x, electrode.y, marker = 'o', color = 'g', markersize = 2, zorder = 0, linestyle='None')
#         for i in range(100):
#             line = np.asarray(list(zip(cell.tvec[int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)]*1.5 + electrode.x[i] +2 - time_range[0]*1.5,
#                         v[i,int(time_range[0]/cell.dt):int(time_range[1]/cell.dt)]/max(abs(v[i, int(time_range[0]/LFP_class.cell.dt):int(time_range[1]/LFP_class.cell.dt)]))*8 + electrode.y[i])))
#             line_segments = LineCollection([line],
#                                            linewidths=(1.5),
#                                            linestyles='solid',
#                                            color = mapper.to_rgba(max(abs(v[i, int(time_range[0]/LFP_class.cell.dt):int(time_range[1]/LFP_class.cell.dt)]))),
#                                            zorder=1,
#                                            rasterized=False)
#             ax.add_collection(line_segments)
#         zips = []
#         for sec in cell.get_pt3d_polygons(projection=('x', 'y')):
#             zips.append(np.transpose(np.asarray(sec)))
#         polycol = PolyCollection(zips, edgecolors='none',
#                                  facecolors='gray', zorder=-1, rasterized=False)
#         ax.add_collection(polycol)
#         ax.set_xlim([min(electrode.x)-20, max(electrode.x)+20])
#         ax.set_ylim([min(electrode.y)-20, max(electrode.y)+20])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.axis('off')
#         plt.colorbar(mapper, ax=ax)

#     else:
#         fig, axs = plt.subplots(10, 10)
#         for i in range(100):
#             ax = axs[int(i / 10), i % 10]
#             ax.plot(cell.tvec, v[i, :], color=mapper.to_rgba(max(abs(v[i, int(time_range[0]/LFP_class.cell.dt):int(time_range[1]/LFP_class.cell.dt)]))))
#             ax.axis('off')
#             ax.set_xlim(time_range[0], time_range[1])
#         cbar = plt.colorbar(mapper, ax=axs)
#         cbar.set_label('LFP (uV)', rotation=270)


# def plot_morphology(LFP_class, if_show_synapses = 0, if_show_electrode = 0, if_show_spine = 0, sections = [], colors = plt.rcParams['axes.prop_cycle'].by_key()['color']):
#     """
#     plot the morphology and location of electrode, synapses
#     :param LFP_class:
#     :param if_show_synapses: bool
#     :param if_show_electrode: bool
#     :param sections: list of name of sections to label with specific color
#     :param colors: list of colors corresponding with sections
#     :return:
#     """
#     electrode = LFP_class.electrode
#     cell = LFP_class.cell
#     fig = plt.figure()
#     ax = fig.add_axes([0.1, 0.05, 0.4, 0.9])

#     for sec, name in zip(cell.get_pt3d_polygons(projection=('x', 'y')), cell.allsecnames):
#         try:
#             if name in sections:
#                 color = colors[sections.index(name)]
#             else:
#                 color = 'gray'
#         except:
#             color = 'gray'
#         polycol = PolyCollection([np.transpose(np.asarray(sec))], edgecolors='none',
#                              facecolors=color, zorder=-1, rasterized=False)
#         ax.add_collection(polycol)
#     ax.set_xlim([-200, 200])
#     ax.set_ylim([-250, 1150])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.axis('off')
#     if if_show_synapses:
#         for s in LFP_class.cell.synapses:
#             if s.kwargs['name'] == 'NMDA':
#                 syn_color = 'c'
#             elif s.kwargs['name'] == 'AMPA':
#                 syn_color = 'b'
#             elif s.kwargs['name'] == 'GABA_A':
#                 syn_color = 'm'
#             else:
#                 syn_color = 'y'
#             ax.plot([s.x], [s.y],
#                     color=syn_color, marker='.')
#     if if_show_spine: # color code: unclustered are black and clustered are red
#         for spine in LFP_class.spines:
#             if spine["if_clustered"]:
#                 for s in spine["name"]:
#                     idx = LFP_class.cell.get_idx(s)[0]
#                     ax.plot([LFP_class.cell.xmid[idx]], [LFP_class.cell.ymid[idx]],
#                             color='r', marker='.')
#             else:
#                 idx = LFP_class.cell.get_idx(spine["name"])[0]
#                 ax.plot([LFP_class.cell.xmid[idx]], [LFP_class.cell.ymid[idx]],
#                         color='k', marker='.')
#     if if_show_electrode:
#         for i in range(electrode.x.size):
#             ax.plot(electrode.x[i], electrode.y[i], color='g', marker='o', markersize = 2)

#     ax2 = fig.add_axes([0.5, 0.05, 0.4, 0.9])
#     for sec, name in zip(cell.get_pt3d_polygons(projection=('z', 'y')), cell.allsecnames):
#         if name in sections:
#             color = colors[sections.index(name)]
#         else:
#             color = 'gray'
#         polycol = PolyCollection([np.transpose(np.asarray(sec))], edgecolors='none',
#                              facecolors=color, zorder=-1, rasterized=False)
#         ax2.add_collection(polycol)
#     ax2.set_xlim([-100, 100])
#     ax2.set_ylim([-250, 1150])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.axis('off')
#     if if_show_synapses:
#         for s in LFP_class.cell.synapses:
#             if s.kwargs['name'] == 'NMDA':
#                 syn_color = 'c'
#             elif s.kwargs['name'] == 'AMPA':
#                 syn_color = 'm'
#             elif s.kwargs['name'] == 'GABA_A':
#                 syn_color = 'k'
#             else:
#                 syn_color = 'y'
#             ax2.plot([s.z], [s.y],
#                     color=syn_color, marker='.')
#     if if_show_spine:
#         for spine in LFP_class.spines:
#             if spine["if_clustered"]:
#                 for s in spine["name"]:
#                     idx = LFP_class.cell.get_idx(s)[0]
#                     ax2.plot([LFP_class.cell.zmid[idx]], [LFP_class.cell.ymid[idx]],
#                             color='r', marker='.')
#             else:
#                 idx = LFP_class.cell.get_idx(spine["name"])[0]
#                 ax2.plot([LFP_class.cell.zmid[idx]], [LFP_class.cell.ymid[idx]],
#                         color='k', marker='.')
#     if if_show_electrode:
#         for i in range(electrode.x.size):
#             ax2.plot(electrode.z[i], electrode.y[i], color='g', marker='o', markersize=2)


# def plot_csd(LFPclass, csd, interp_ratio = 100, col = 5, time_range = [50., 100.]):
#     """

#     :param electrode:
#     :param csd:
#     :param col:
#     :return:
#     """
#     t = LFPclass.cell.tvec
#     x = LFPclass.electrode.x
#     y = LFPclass.electrode.y
#     z = LFPclass.electrode.z

#     X = np.sort(np.unique(x))
#     Y = np.sort(np.unique(y))
#     Z = np.sort(np.unique(z))

#     deltaX = np.unique(0.01*np.floor(np.diff(X)/0.01))
#     deltaY = np.unique(0.01*np.floor(np.diff(Y)/0.01))
#     deltaZ = np.unique(0.01*np.floor(np.diff(Z)/0.01))


#     idxl = np.argwhere(t==time_range[0])[0][0]
#     idxr = np.argwhere(t == time_range[1])[0][0]
#     [mesh_X, mesh_Y] = np.meshgrid(t[idxl:idxr], np.arange(np.min(Y), np.max(Y), 1 / interp_ratio))
#     if deltaX.size == 0 and deltaZ.size == 0:
#         csd = csd[:,idxl:idxr]
#     elif deltaX.size == 0 or deltaZ.size == 0:
#         if deltaX.size == 0:
#             Nh = Z.size
#         else:
#             Nh = X.size
#         csd = csd[np.arange(col, Nh*(Nh)+col, Nh), idxl:idxr]
#     else:
#         raise ValueError("3D electrode not supported yet")

#     y_interp = np.arange(np.max(Y), np.min(Y), -1 / interp_ratio)
#     csd_interp = np.zeros([y_interp.size, csd.shape[1]])
#     for i in range(csd.shape[1]):
#         csd_interp[:,i] = np.interp(y_interp, Y, csd[:,i])

#     fig = plt.figure()
#     # ax = fig.gca(projection='3d')
#     # surf = ax.plot_surface(mesh_X, mesh_Y, csd_interp, cmap=matplotlib.cm.coolwarm,
#     #                        linewidth=0, antialiased=False)
#     plt.contourf(mesh_X, mesh_Y, csd_interp, 20, cmap=matplotlib.cm.coolwarm)
#     cbar = plt.colorbar()
#     cbar.set_label('csd (A/(cm)^2)', rotation=270)
#     plt.xlim(time_range)
#     plt.xlabel('t (ms)')
#     plt.ylabel('y (um)')
#     # ax.set_zlabel('csd (A/(cm)^2)')
