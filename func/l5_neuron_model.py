import numpy as np
import LFPy
import os
import sys
from scipy.integrate import odeint
from scipy import signal
import scipy
import warnings
import pylab as pl
import statistics

from neuron import h

from func import morphology
import math

import json
import neuron

h.load_file("stdrun.hoc")
def locate_center(section):
    """
    locate the center of the section, return the x, y, z position

    Parameters
    ----------
    section : nrn.Section
        the target section

    Returns
    -------
    np.array([x, y, z])

    """
    n3d = section.n3d()
    x3d = []
    z3d = []
    y3d = []
    for i in range(n3d):
        x3d.append(section.x3d(i))
        y3d.append(section.y3d(i))
        z3d.append(section.z3d(i))

    mid_x = statistics.median(x3d)
    mid_y = statistics.median(y3d)
    mid_z = statistics.median(z3d)
    return np.array([mid_x, mid_y, mid_z])

def get_longest_branch(cell):
    seg_d = []

    soma_seg_idx = cell.get_idx("soma[0]")

    for sec in h.allsec():
        for seg, seg_idx in zip(sec, cell.get_idx(sec.name())):
            try:
                d = cell.get_intersegment_distance(soma_seg_idx, seg_idx)
            except:
                pass
            seg_d.append(d)

    max_d = np.max(seg_d)
    return max_d

def distribute_channels_linear(cell, sec_name, mec, value_name, value, verbool = False):
    """
    set gradient in of a value in the mechanism in sections, based on the distance between the center of the section and soma
    The value is a np vector of the param_value, the sections will be assigned based distance to soma and linear interpolation

    Parameters
    ----------
    sec_name : str or list
        'soma', 'dend', 'apic' or 'axon'
    mec : str
        name of the mechaniem
    value_name : str
        name of the value to be set
    value : np array,
    g = value[0] + d * value[1]
    Returns
    -------
    None.

    """
    # load and sort the sections
    sections = []
    sec_d = []

    for sec in h.allsec():
        if sec.name()[0:4] in sec_name:
            sections.append(sec)

    # sections.remove(sections[0])

    # load the soma section and its position
    soma_seg_idx = cell.get_idx("soma[0]")

    # calculate the distance and set gradient
    for sec in sections:
        if ("shead" not in sec.name()) and ("sneck" not in sec.name()) and ("pre" not in sec.name()):
            for seg, seg_idx in zip(sec, cell.get_idx(sec.name())):
                d = cell.get_intersegment_distance(soma_seg_idx, seg_idx)
                if h.ismembrane(str(mec), sec=sec) != 1:
                    sec.insert(str(mec))
                if verbool:
                    print('Setting %s to %.6g in %s'
                          % (value_name, value[0] + d * value[1], sec.name()))
                seg_mec = getattr(seg, str(mec))
                setattr(sec_mec, value_name, value[0] + d * value[1])


def distribute_channels_step(cell, sec_name, mec, value_name, value, verbool = False):
    """
    set the value of mechanism parameter with step function, usually to set conducntance of Ca channels on apical dendrite
    value: np.array(start_dis, stop_dis, value_in, value_out, scale factor)

    Parameters
    ----------
    sec_name : str or list
        'soma', 'dend', 'apic' or 'axon'
    mec : str
        name of the mechaniem
    value_name : str
        name of the value to be set
    value : np array,
        a vector of the values. np.array(start_dis, stop_dis, value_in, value_out)

    Returns
    -------
    None.

    """
    # load and sort the sections
    # load and sort the sections
    sections = []

    for sec in h.allsec():
        if sec.name()[0:4] in sec_name:
            sections.append(sec)

    # sections.remove(sections[0])
    # load the soma section and its position
    soma_seg_idx = cell.get_idx("soma[0]")

    # calculate the distance and set parameter value
    for sec in h.allsec():
        if ("shead" not in sec.name()) and ("sneck" not in sec.name()) and ("pre" not in sec.name()):
            for seg, seg_idx in zip(sec, cell.get_idx(sec.name())):
                d = cell.get_intersegment_distance(soma_seg_idx, seg_idx)
                if d >= value[0] and d <= value[1]:
                    if h.ismembrane(str(mec), sec=sec) != 1:
                        sec.insert(str(mec))
                    if verbool:
                        print('Setting %s to %.6g in %s'
                              % (value_name, value[2]*value[4], sec.name()))
                    seg_mec = getattr(seg, str(mec))
                    setattr(seg_mec, value_name, value[2]*value[4])
                else:
                    if h.ismembrane(str(mec), sec=sec) != 1:
                        sec.insert(str(mec))
                    if verbool:
                        print('Setting %s to %.6g in %s'
                              % (value_name, value[3]*value[4], sec.name()))
                    seg_mec = getattr(seg, str(mec))
                    setattr(seg_mec, value_name, value[3] * value[4])


def distribute_channels_exponential(cell, sec_name, mec, value_name, value, verbool = False):
    """
    set the value of mechanism parameter with exponential function, usually to set conducntance of Ih on apical dendrite
    g_sec = value[4]* (value[0] + value[3]*exp(value[1]*(d - value[2])))

    Parameters
    ----------
    sec_name : str or list
        'soma', 'dend', 'apic' or 'axon'
    mec : str
        name of the mechaniem
    value_name : str
        name of the value to be set
    value : np array,
        a vector of the values. g_sec = value[4]* (value[0] + value[3]*exp(value[1]*(d - value[2])))

    Returns
    -------
    None.

    """
    # load and sort the sections
    sections = []

    for sec in h.allsec():
        if sec.name()[0:4] in sec_name:
            sections.append(sec)
    # sections.remove(sections[0])

    soma_seg_idx = cell.get_idx("soma[0]")

    max_d = get_longest_branch(cell)

    # calculate the distance and set the prameter value
    for sec in sections:
        if ("shead" not in sec.name()) and ("sneck" not in sec.name()) and ("pre" not in sec.name()):
            for seg, seg_idx in zip(sec, cell.get_idx(sec.name())):
                d = cell.get_intersegment_distance(soma_seg_idx, seg_idx)
                value_insert = value[4]*(value[0] + value[3] * math.exp(value[1] * (d/max_d - value[2])))
                if value_insert <= 0:
                    value_insert = 0
                if h.ismembrane(str(mec), sec=sec) != 1:
                    sec.insert(str(mec))
                if verbool:
                    print('Setting %s to %.6g in %s'
                          % (value_name, value_insert, sec.name()))
                seg_mec = getattr(seg, str(mec))
                setattr(seg_mec, value_name, value_insert)

def distribute_channels_sigmoid(cell, sec_name, mec, value_name, value, verbool = False):
    """
    set the value of mechanism parameter with exponential function, usually to set conducntance of Ih on apical dendrite
    g_sec = value[0] + value[1]*(1 + exp((d - value[2])/value[3]))

    Parameters
    ----------
    sec_name : str or list
        'soma', 'dend', 'apic' or 'axon'
    mec : str
        name of the mechaniem
    value_name : str
        name of the value to be set
    value : np array,
        a vector of the values. g_sec = value[0] + value[1]*(1 + exp((d - value[2])/value[3]))

    Returns
    -------
    None.

    """
    # load and sort the sections
    sections = []

    for sec in h.allsec():
        if sec.name()[0:4] in sec_name:
            sections.append(sec)

    # sections.remove(sections[0])

    soma_seg_idx = cell.get_idx("soma[0]")

    # calculate the distance and set the prameter value
    for sec in sections:
        for seg, seg_idx in zip(sec, cell.get_idx(sec.name())):
            d = cell.get_intersegment_distance(soma_seg_idx, seg_idx)
            value_insert = value[0] + value[1]*(1 + math.exp((d - value[2])/value[3]))
            if value_insert <= 0:
                value_insert = 0
            if h.ismembrane(str(mec), sec=sec) != 1:
                sec.insert(str(mec))
            if verbool:
                print('Setting %s to %.6g in %s'
                      % (value_name, value_insert, sec.name()))
            seg_mec = getattr(seg, str(mec))
            setattr(seg_mec, value_name, value_insert)

class L5Model:
    """
    create L5 PN model with the parameter in the dictionary P
    """
    def __init__(self, wd, P, verbool = False):
        self.wd = wd
        self.P = P
        self.verbool = verbool
        A, nseg, L, a, sec_points, self.sec_e, self.sec_i, self.b_type_e, self.b_type_i, self.basal, self.apical, self.trunk, self.axon = \
            self.define_morphology()
        _, a_s, _ = morphology.seg_geometry(L, a, nseg)
        self.tree, self.seg_list, self.seg_e, self.seg_i = \
            self.create_sections(A, L, a_s, nseg, sec_points)
        self.build_tree(A)
        self.nseg = nseg
        if 'data' not in P:
            f = open(P["param_file"])
            P['data'] = json.load(f)
        if 'if_stochastic' not in P:
            self.if_stochastic = False
        else:
            self.if_stochastic = P['if_stochastic']
        self.data = P['data']
        self.insert_active()
        self.insert_active_gradient()
        if P['active_d']:
            if self.if_stochastic:
                self.insert_active_basal_stochastic()
            else:
                self.insert_active_basal()
        self.AMPA = self.attach_ampa(self.sec_e)
        self.GABA = self.attach_gaba(self.sec_i)
        if P['active_n']:
            self.NMDA = self.attach_nmda(self.sec_e)
        else:
            self.NMDA = self.attach_pas_nmda(self.sec_e)
        self.w_1 = self.sec_e.shape[1] * [P['g_max_A']]
        self.w_2 = self.sec_e.shape[1] * [P['g_max_N']]
        self.w_3 = self.sec_i.shape[1] * [P['g_max_G']]
        self.r_na = P['r_na']
        self.set_weights(np.array(self.w_1) + np.array(self.w_2), np.array(self.w_3))
        self.P = P

        sec_coor = []
        for sec in self.allseclist:
            try:
                sec_coor.append(locate_center(sec))
            except:
                pass

        sec_coor = np.asarray(sec_coor)
        self.xrange_entire = [min(sec_coor[:, 0]), max(sec_coor[:, 0])]
        self.xmid_entire = np.median(sec_coor[:, :])
        self.yrange_entire = [min(sec_coor[:, 1]), max(sec_coor[:, 1])]
        self.ymid_entire = np.median(sec_coor[:, 1])
        self.zrange_entire = [min(sec_coor[:, 2]), max(sec_coor[:, 2])]
        self.zmid_entire = np.median(sec_coor[:, 2])

    def Section(self, name):
                # h("create " + name)
                # return h.__getattribute__(name)
        for sec in h.allsec():
            if sec.name()==name:
                return sec

    def define_morphology(self):
        """ Create adjacency matrix, lists of segment numbers and dimensions,
        and synapse locations.

        Returns
        -------
        A : ndarray
            adjacency matrix for dendritic sections
        nseg : ndarray
            number of segments in each section
        L : ndarray
            section lengths
        a : ndarray
            section radii
        sec_points : list
            section coordinates
        sec_e, sec_i :  ndarray
            locations of excitatory and inhibitory synapses
        b_type_e, b_type_i : list
            branch types (basal/apical/soma) for all synapses
        """
        P = self.P
        N_e, N_i, l_seg, P_locs_e, P_locs_i, branch_ids = \
            (P['N_e'], P['N_i'], P['l_seg'], P['locs_e'], P['locs_i'],
             [P['basal'], P['oblique'], P['apical']])
        A, L, a, sec_points, secs, basal, apical, trunk, axon = morphology.reconstruction(P['tree'])

        branch_ids = [basal, apical, trunk]
        nseg = np.array(len(L) * [1])
        dseg = L[secs] // (l_seg * 1e-4) + 1
        dseg[dseg == 1] = 2
        nseg[secs] = dseg
        locs_e = np.asarray([]).astype(int)
        locs_i = np.asarray([]).astype(int)
        if 'basal' in P_locs_e:
            locs_e = np.append(locs_e, basal)
        if 'apical' in P_locs_e:
            locs_e = np.append(locs_e, apical)
        elif 'tuft' in P_locs_e:
            locs_e = np.append(locs_e, apical[37:])
        elif 'trunk' in P_locs_e:
            locs_e = np.append(locs_e, apical[:37])

        if 'basal' in P_locs_i:
            locs_i = np.append(locs_i, basal)
        if 'apical' in P_locs_i:
            locs_i = np.append(locs_i, apical)
        elif 'tuft' in P_locs_e:
            locs_i = np.append(locs_i, apical[37:])
        elif 'trunk' in P_locs_i:
            locs_i = np.append(locs_i, apical[:37])


        sec_e = morphology.synapse_locations_rand(locs_e, N_e, nseg[locs_e], 0)
        sec_i = morphology.synapse_locations_rand(locs_i, N_i, nseg[locs_i], 0)
        b_type_e = morphology.branch_type(sec_e, branch_ids)
        b_type_i = morphology.branch_type(sec_i, branch_ids)
        return A, nseg, L, a, sec_points, sec_e, sec_i, b_type_e, b_type_i, basal, apical, trunk, axon

    def create_sections(self, A, L, a_s, nseg, sec_points):
        """Create sections from adjacency matrix A.

        Parameters
        ----------
        A : ndarray
            adjacency matrix for dendritic sections
        L : ndarray
            section lengths
        a_s : ndarray
            segment radii
        nseg : ndarray
            number of segments in each section
        sec_points : list
            section coordinates

        Returns
        -------
        seg_list : ndarray
            sections and positions for each segment
        seg_e, seg_i : ndarray
            synapse segment numbers
        tree : list
            section objects (tree[0]=soma)

        """
        self.totnsegs = np.sum(nseg)
        # soma = self.Section("soma[0]")
        # soma.L = L[0]*1e4			#(um)
        # tree = [soma]
        self.soma = [0]
        tree = []
        count_axon = 0
        count_dend = 0
        count_apic = 0
        count_trunk = 0
        count_soma = 0
        for k in range(0, A.shape[0]):
            if k in self.axon:
                if count_axon == 0:
                    exec('h(\'create axon[' + str(int(len(self.axon))) + ']\')')
                sec = self.Section('axon[%d]' % count_axon)
                count_axon = count_axon + 1
            elif k in self.basal:
                if count_dend == 0:
                    exec('h(\'create dend[' + str(int(len(self.basal))) + ']\')')
                sec = self.Section('dend[%d]' % count_dend)
                count_dend = count_dend + 1
            elif k in self.apical:
                if count_apic == 0:
                    exec('h(\'create apic[' + str(int(len(self.apical))) + ']\')')
                sec = self.Section('apic[%d]' % count_apic)
                count_apic = count_apic + 1
            elif k in self.trunk:
                if count_trunk == 0:
                    exec('h(\'create trunk[' + str(int(len(self.trunk))) + ']\')')
                sec = self.Section('trunk[%d]' % count_trunk)
                count_trunk = count_trunk + 1
            elif k in self.soma:
                if count_soma == 0:
                    exec('h(\'create soma[' + str(int(len(self.soma))) + ']\')')
                sec = self.Section('soma[%d]' % count_soma)
                count_soma = count_soma + 1
            print(sec.name())
            sec.L = L[k] * 1e4  # (um)
            sec.nseg = nseg[k]
            tree.append(sec)
        for k, branch in enumerate(tree):
            h.pt3dclear(sec=branch)
            for pt in sec_points[k]:
                h.pt3dadd(pt[0], pt[1], pt[2], 2 * pt[3], sec=branch)
        j = 0
        for branch in tree:
            for seg in branch:
                seg.diam = 2 * a_s[j] * 1e4
                j += 1
        # soma.L = L[0]*1e4

        j = 0
        seg_list = []
        for num, sec in enumerate(tree):
            for k in range(sec.nseg):
                seg_list.append([num, (k + 1 / 2) / sec.nseg, j])
                j += 1
        seg_list = np.array(seg_list)

        seg_e = np.array([np.where(np.sum(np.abs(seg_list[:, :2] - sec), 1) < 1e-15)[0][0]
                          for sec in self.sec_e.T])
        seg_i = np.array([np.where(np.sum(np.abs(seg_list[:, :2] - sec), 1) < 1e-15)[0][0]
                          for sec in self.sec_i.T])
        return tree, seg_list, seg_e, seg_i

    def build_tree(self, A):
        """Connect sections to form tree (tree[0]=soma)

        Parameters
        ----------
        A : ndarray
            adjacency matrix
        """
        if A.shape[0] > 1:
            for i in range(A.shape[0]):
                for j in range(i, A.shape[0]):
                    if A[i, j] == 1:
                        self.tree[j].connect(self.tree[i](1))
        self.allsecnames = []
        self.allseclist = neuron.h.SectionList()
        for sec in neuron.h.allsec():
            self.allsecnames.append(sec.name())
            self.allseclist.append(sec=sec)

        # list of soma sections, assuming it is named on the format "soma*"
        self.nsomasec = 0
        self.somalist = neuron.h.SectionList()
        for sec in neuron.h.allsec():
            if sec.name().find('soma') >= 0:
                self.somalist.append(sec=sec)
                self.nsomasec += 1

        # find the 3d coortinate
        areavec = np.zeros(self.totnsegs)
        diamvec = np.zeros(self.totnsegs)
        lengthvec = np.zeros(self.totnsegs)

        xstartvec = np.zeros(self.totnsegs)
        xendvec = np.zeros(self.totnsegs)
        ystartvec = np.zeros(self.totnsegs)
        yendvec = np.zeros(self.totnsegs)
        zstartvec = np.zeros(self.totnsegs)
        zendvec = np.zeros(self.totnsegs)

        counter = 0

        for sec in self.allseclist:
            n3d = int(neuron.h.n3d())
            nseg = sec.nseg
            gsen2 = 1. / 2 / nseg
            if n3d > 0:
                # create interpolation objects for the xyz pt3d info:
                L = np.zeros(n3d)
                x = np.zeros(n3d)
                y = np.zeros(n3d)
                z = np.zeros(n3d)
                for i in range(n3d):
                    L[i] = neuron.h.arc3d(i)
                    x[i] = neuron.h.x3d(i)
                    y[i] = neuron.h.y3d(i)
                    z[i] = neuron.h.z3d(i)

                # normalize as seg.x [0, 1]
                L /= sec.L

                # temporary store position of segment midpoints
                segx = np.zeros(nseg)
                for i, seg in enumerate(sec):
                    segx[i] = seg.x

                # can't be >0 which may happen due to NEURON->Python float transfer:
                segx0 = (segx - gsen2).round(decimals=6)
                segx1 = (segx + gsen2).round(decimals=6)

                # fill vectors with interpolated coordinates of start and end points
                xstartvec[counter:counter + nseg] = np.interp(segx0, L, x)
                xendvec[counter:counter + nseg] = np.interp(segx1, L, x)

                ystartvec[counter:counter + nseg] = np.interp(segx0, L, y)
                yendvec[counter:counter + nseg] = np.interp(segx1, L, y)

                zstartvec[counter:counter + nseg] = np.interp(segx0, L, z)
                zendvec[counter:counter + nseg] = np.interp(segx1, L, z)

                # fill in values area, diam, length
                for i, seg in enumerate(sec):
                    areavec[counter] = neuron.h.area(seg.x)
                    diamvec[counter] = seg.diam
                    lengthvec[counter] = sec.L / nseg

                    counter += 1

        # set cell attributes
        self.xstart = xstartvec
        self.ystart = ystartvec
        self.zstart = zstartvec

        self.xend = xendvec
        self.yend = yendvec
        self.zend = zendvec

        self.area = areavec
        self.diam = diamvec
        self.length = lengthvec

        self.xmid = .5 * (self.xstart + self.xend).flatten()
        self.ymid = .5 * (self.ystart + self.yend).flatten()
        self.zmid = .5 * (self.zstart + self.zend).flatten()

    def _get_idx(self, seclist):
        """Return boolean vector which indexes where segments in seclist
        matches segments in neuron.h.allsec(), rewritten from
        LFPy.hoc function get_idx()"""
        if neuron.h.allsec() == seclist:
            return np.ones(self.totnsegs, dtype=bool)
        else:
            idxvec = np.zeros(self.totnsegs, dtype=bool)
            # get sectionnames from seclist
            seclistnames = []
            for sec in seclist:
                seclistnames.append(sec.name())
            seclistnames = np.array(seclistnames, dtype='|S128')
            segnames = np.empty(self.totnsegs, dtype='|S128')
            i = 0
            for sec in self.allseclist:
                secname = sec.name()
                for seg in sec:
                    segnames[i] = secname
                    i += 1
            for name in seclistnames:
                idxvec[segnames == name] = True

            return idxvec

    def get_idx(self, section='allsec', z_min=-np.inf, z_max=np.inf):
        """Returns compartment idx of segments from sections with names that match
        the pattern defined in input section on interval [z_min, z_max].
        Parameters
        ----------
        section : str
            Any entry in cell.allsecnames or just 'allsec'.
        z_min : float
            Depth filter. Specify minimum z-position
        z_max : float
            Depth filter. Specify maximum z-position
        Examples
        --------
        >>> idx = cell.get_idx(section='allsec')
        >>> print(idx)
        >>> idx = cell.get_idx(section=['soma', 'dend', 'apic'])
        >>> print(idx)
        """

        if section == 'allsec':
            seclist = neuron.h.allsec()
        else:
            seclist = neuron.h.SectionList()
            if type(section) == str:
                for sec in self.allseclist:
                    if sec.name().find(section) >= 0:
                        seclist.append(sec=sec)
            elif type(section) == list:
                for secname in section:
                    for sec in self.allseclist:
                        if sec.name().find(secname) >= 0:
                            seclist.append(sec=sec)
            else:
                if self.verbool:
                    print('%s did not match any section name' % str(section))

        idx = self._get_idx(seclist)
        sel_z_idx = (self.zmid[idx] > z_min) & (self.zmid[idx] < z_max)
        return np.arange(self.totnsegs)[idx][sel_z_idx]

    def get_intersegment_vector(self, idx0=0, idx1=0):
        """Return the distance between midpoints of two segments with index
        idx0 and idx1. The argument returned is a vector [x, y, z], where
        x = self.xmid[idx1] - self.xmid[idx0] etc.
        Parameters
        ----------
        idx0 : int
        idx1 : int
        """
        vector = []
        try:
            if idx1 < 0 or idx0 < 0:
                raise Exception('idx0 < 0 or idx1 < 0')
            vector.append(self.xmid[idx1] - self.xmid[idx0])
            vector.append(self.ymid[idx1] - self.ymid[idx0])
            vector.append(self.zmid[idx1] - self.zmid[idx0])
            return vector
        except:
            ERRMSG = 'idx0 and idx1 must be ints on [0, %i]' % self.totnsegs
            raise ValueError(ERRMSG)

    def get_intersegment_distance(self, idx0=0, idx1=0):
        """
        Return the Euclidean distance between midpoints of two segments.
        Parameters
        ----------
        idx0 : int
        idx1 : int
        Returns
        -------
        float
            Will return a float in unit of micrometers.
        """
        try:
            vector = np.array(self.get_intersegment_vector(idx0, idx1))
            return np.sqrt((vector ** 2).sum())
        except:
            ERRMSG = 'idx0 and idx1 must be ints on [0, %i]' % self.totnsegs
            raise ValueError(ERRMSG)

    def define_biophysics(self):
        """Set biophysical parameters with unit conversions."""
        if 'c_m_d' not in self.P:
            self.P['c_m_d'] = self.P['c_m']
        for sec in self.tree:
            sec.Ra = self.P['R_a'] * 1e3  # (ohm cm)
            sec.cm = self.P['c_m_d']
            sec.insert('pas')
            sec.g_pas = self.P['c_m_d'] / self.P['tau_m'] * 1e-3  # (S/cm^2)
            sec.e_pas = self.P['E_r']
        self.tree[0].cm = self.P['c_m']
        self.tree[0].g_pas = self.P['c_m'] / self.P['tau_m'] * 1e-3

    def attach_ampa(self, E):
        """Attach double exponential AMPA synapses.

        Parameters
        ----------
        E : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        AMPA : list
            synapse objects
        """
        AMPA = []
        self.AMPA_meta = []
        for k in range(E.shape[1]):
            syn = h.Exp2Syn(self.tree[int(E[0, k])](E[1, k]))
            syn.tau1 = self.P['tauA'][0]
            syn.tau2 = self.P['tauA'][1]
            syn.e = self.P['E_e']
            AMPA.append(syn)
            syn_meta = {}
            syn_meta["sec"] = self.tree[int(E[0, k])]
            syn_meta["sec_name"] = self.tree[int(E[0, k])].name()
            syn_meta["sec_idx"] = self.allsecnames.index(syn_meta["sec_name"])
            syn_meta["seg_idx"] = int(E[1, k] / (1 / self.nseg[int(syn_meta["sec_idx"])]))
            syn_meta["seg"] = self.tree[int(E[0, k])](E[1, k])
            self.AMPA_meta.append(syn_meta)
        return AMPA

    def attach_nmda(self, E):
        """Attach double exponential NMDA synapses with sigmoid voltage
        dependence.

        Parameters
        ----------
        E : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        NMDA : list
            synapse objects
        """
        NMDA = []
        self.NMDA_meta = []
        for k in range(E.shape[1]):
            syn = h.Exp2Syn_NMDA(self.tree[int(E[0, k])](E[1, k]))
            syn.tau1 = self.P['tauN'][0]
            syn.tau2 = self.P['tauN'][1]
            syn.e = self.P['E_e']
            NMDA.append(syn)
            syn_meta = {}
            syn_meta["sec"] = self.tree[int(E[0, k])]
            syn_meta["sec_name"] = self.tree[int(E[0, k])].name()
            syn_meta["sec_idx"] = self.allsecnames.index(syn_meta["sec_name"])
            syn_meta["seg_idx"] = int(E[1, k] / (1 / self.nseg[int(syn_meta["sec_idx"])]))
            syn_meta["seg"] = self.tree[int(E[0, k])](E[1, k])
            self.NMDA_meta.append(syn_meta)
        return NMDA

    def attach_gaba(self, I):
        """Attach double exponential GABA synapses.

        Parameters
        ----------
        I : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        GABA : list
            synapse objects
        """
        GABA = []
        self.GABA_meta = []
        for k in range(I.shape[1]):
            syn = h.Exp2Syn(self.tree[int(I[0, k])](I[1, k]))
            syn.tau1 = self.P['tauG'][0]
            syn.tau2 = self.P['tauG'][1]
            syn.e = self.P['E_i']
            GABA.append(syn)
            syn_meta = {}
            syn_meta["sec"] = self.tree[int(I[0, k])]
            syn_meta["sec_name"] = self.tree[int(I[0, k])].name()
            syn_meta["sec_idx"] = self.allsecnames.index(syn_meta["sec_name"])
            syn_meta["seg_idx"] = int(I[1, k] / (1 / self.nseg[int(syn_meta["sec_idx"])]))
            syn_meta["seg"] = self.tree[int(I[0, k])](I[1, k])
            self.GABA_meta.append(syn_meta)
        return GABA

    def attach_pas_nmda(self, E):
        """Attach double exponential NMDA synapses without voltage dependence.

        Parameters
        ----------
        E : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        NMDA : list
            synapse objects
        """
        NMDA = []
        for k in range(E.shape[1]):
            syn = h.Exp2Syn(self.tree[int(E[0, k])](E[1, k]))
            syn.tau1 = self.P['tauN'][0]
            syn.tau2 = self.P['tauN'][1]
            syn.e = self.P['E_e']
            NMDA.append(syn)
        return NMDA

    def insert_active(self):
        """
        insert channels based on data from the jsonfile
        :return:
        """
        """
        adapted from allensdk.model.biophysical.utils.load_cell_parameters

        Returns
        -------
        None.

        """
        if not hasattr(self, "data"):
            self.data = self.P['data']
        passive = self.data['passive'][0]
        genome = self.data['genome']
        conditions = self.data['conditions'][0]
        if 'if_stochastic' not in self.P:
            if_stochastic = False
            stochastic_channel = []
        else:
            if_stochastic = self.P['if_stochastic']
            stochastic_channel = self.P['stochastic_channel']

        # Set fixed passive properties
        for sec in h.allsec():
            sec.Ra = passive['ra']
            sec.insert('pas')
            try:
                for seg in sec:
                    seg.pas.e = passive["e_pas"]
            except KeyError:
                pass
                # print("the model is all-active", end = '\n')
        if self.verbool:
            print("Setting Ra to be %.6g in all sections" % passive['ra'])
            print("Setting e_pas to be %.6g in all sections" % passive['e_pas'])
        # get the cm for each section
        try:
            for c in passive["cm"]:
                if 'soma' in c.keys():
                    cm_soma = c["cm"]
                elif 'axon' in c.keys():
                    cm_axon = c["cm"]
                elif 'apic' in c.keys():
                    cm_apic = c["cm"]
                elif 'dend' in c.keys():
                    cm_dend = c["cm"]
            for sec in h.allsec():
                if sec.name()[0:3] == "soma":
                    sec.cm = cm_soma
                elif sec.name()[0:3] == "apic":
                    sec.cm = cm_apic
                elif sec.name()[0:3] == "dend":
                    sec.cm = cm_dend
                elif sec.name()[0:3] == "axon":
                    sec.cm = cm_axon
                else:
                    print("Error in assigning cm in %s, set as 1" % sec.name())
                    sec.cm = 1
        except KeyError:
            pass
            # print("the model is all-active", end='\n')

        # Insert channels and set parameters
        for p in genome:
            section_array = p["section"]
            mechanism = p["mechanism"]
            param_name = p["name"]
            param_value = float(p["value"])
            if section_array == "glob":  # global parameter
                for sec in h.allsec():
                    param = getattr(sec, p["name"])
                    param = p["value"]
            else:
                if mechanism != "":
                    for sec in h.allsec():
                        if sec.name()[0:4] in section_array:
                            if not if_stochastic:
                                if h.ismembrane(str(mechanism),
                                                sec=sec) != 1:
                                    sec.insert(str(mechanism))
                                    if self.verbool:
                                        print('Adding mechanism %s to %s'
                                              % (mechanism, sec.name()))
                                setattr(sec, param_name + "_" + mechanism, param_value)
                            else:
                                if str(mechanism) in stochastic_channel:
                                    mechanism = mechanism + '_2F'
                                    if h.ismembrane(str(mechanism),
                                                    sec=sec) != 1:
                                        sec.insert(str(mechanism))
                                        if self.verbool:
                                            print('Adding mechanism %s to %s'
                                                  % (mechanism, sec.name()))
                                    setattr(sec, param_name + "_" + mechanism, param_value)
                                    N_name = 'N' + mechanism[0:-3]
                                    if mechanism in ['NaTs2_t_2F', 'NaTa_t_2F']:
                                        N = np.round(
                                            param_value * np.sum(
                                                self.area[self.get_idx(sec.name())]) * 100 / 5)
                                    else:
                                        N = np.round(
                                            param_value * np.sum(
                                                self.area[self.get_idx(sec.name())]) * 100 / 40)
                                    # assume that each channel has conductance of 10pS
                                    setattr(sec, N_name + '_' + mechanism, N)
                                    if self.verbool:
                                        print('Changing num of channel in mechanism %s in %s to %d'
                                              % (mechanism, sec.name(), N))
                                else:
                                    if h.ismembrane(str(mechanism),
                                                    sec=sec) != 1:
                                        sec.insert(str(mechanism))
                                        if self.verbool:
                                            print('Adding mechanism %s to %s'
                                                  % (mechanism, sec.name()))
                                    setattr(sec, param_name + "_" + mechanism, param_value)

                else:
                    for sec in h.allsec():
                        if sec.name()[0:4] in section_array:
                            setattr(sec, param_name, param_value)
                if self.verbool:
                    print('Setting %s to %.6g in %s'
                          % (param_name, param_value, section_array))

        for erev in conditions['erev']:
            for sec in h.allsec():
                if sec.name()[0:3] in erev["section"]:
                    try:
                        sec.ek = float(erev["ek"])
                        sec.ena = float(erev["ena"])
                        sec.eca = float(erev["eca"])
                    except:
                        pass  # print('No Na or K channel in ' + erev["section"])

    def insert_passive(self):
        """
        adapted from allensdk.model.biophysical.utils.load_cell_parameters

        Returns
        -------
        None.

        """
        if not hasattr(self, "data"):
            self.data = self.P['data']
        passive = self.data['passive'][0]
        genome = self.data['genome']
        conditions = self.data['conditions'][0]

        # Set fixed passive properties
        for sec in h.allsec():
            sec.Ra = passive['ra']
            sec.insert('pas')
            try:
                for seg in sec:
                    seg.pas.e = passive["e_pas"]
            except KeyError:
                pass
                # print("the model is all-active", end = '\n')
        if self.verbool:
            print("Setting Ra to be %.6g in all sections" % passive['ra'])
            print("Setting e_pas to be %.6g in all sections" % passive['e_pas'])
        # get the cm for each section
        try:
            for c in passive["cm"]:
                if 'soma' in c.keys():
                    cm_soma = c["cm"]
                elif 'axon' in c.keys():
                    cm_axon = c["cm"]
                elif 'apic' in c.keys():
                    cm_apic = c["cm"]
                elif 'dend' in c.keys():
                    cm_dend = c["cm"]
                elif 'section' in c.keys():
                    if 'soma' in c["section"]:
                        cm_soma = c["cm"]
                    elif 'dend' in c["section"]:
                        cm_dend = c["cm"]
                    elif 'apic' in c["section"]:
                        cm_apic = c["cm"]
                    elif 'axon' in c["section"]:
                        cm_axon = c["cm"]
            for sec in h.allsec():
                if "soma" in sec.name():
                    sec.cm = cm_soma
                elif "apic" in sec.name():
                    sec.cm = cm_apic
                elif "dend" in sec.name():
                    sec.cm = cm_dend
                elif "axon" in sec.name():
                    sec.cm = cm_axon
                else:
                    print("Error in assigning cm in %s, set as 1" % sec.name())
                    sec.cm = 1
        except KeyError:
            pass
            # print("the model is all-active", end='\n')

        # Insert channels and set parameters
        for p in genome:
            section_array = p["section"]
            mechanism = p["mechanism"]
            param_name = p["name"]
            param_value = float(p["value"])
            if section_array == "glob":  # global parameter
                for sec in h.allsec():
                    param = getattr(sec, p["name"])
                    param = p["value"]
            else:
                if mechanism != "":
                    if "Ih" in str(mechanism):
                        for sec in h.allsec():
                            if sec.name()[0:4] in section_array:
                                if h.ismembrane(str(mechanism),
                                                sec=sec) != 1:
                                    sec.insert(str(mechanism))
                                    if self.verbool:
                                        print('Adding mechanism %s to %s'
                                              % (mechanism, sec.name()))
                                setattr(sec, param_name + "_" + mechanism, param_value)
                    else:
                        continue
                else:
                    for sec in h.allsec():
                        if sec.name()[0:4] in section_array:
                            setattr(sec, param_name, param_value)
                if self.verbool:
                    print('Setting %s to %.6g in %s'
                          % (param_name, param_value, section_array))

        for erev in conditions['erev']:
            for sec in h.allsec():
                if sec.name()[0:3] in erev["section"]:
                    try:
                        sec.ek = float(erev["ek"])
                        sec.ena = float(erev["ena"])
                        sec.eca = float(erev["eca"])
                    except:
                        pass  # print('No Na or K channel in ' + erev["section"])

    def insert_active_gradient(self):
        if not hasattr(self, "data"):
            self.data = self.P['data']
        gradient = self.data['gradient']
        for g in gradient:
            section_array = g["section"]
            mechanism = g["mechanism"]
            value_name = g["name"]
            value = g["value"]
            function = g["function"]
            if mechanism != "":
                if self.verbool:
                    print('Adding mechanism %s to %s'
                          % (mechanism, section_array))
                if function == 'linear':
                    distribute_channels_linear(cell=self, sec_name=section_array, mec=mechanism,
                                               value_name=value_name,
                                               value=value, verbool=self.verbool)
                elif function == 'sigmoid':
                    distribute_channels_sigmoid(cell=self, sec_name=section_array, mec=mechanism,
                                                value_name=value_name,
                                                value=value, verbool=self.verbool)
                elif function == 'step':
                    distribute_channels_step(cell=self, sec_name=section_array, mec=mechanism,
                                             value_name=value_name, value=value,
                                             verbool=self.verbool)
                elif function == 'exponential':
                    distribute_channels_exponential(cell=self, sec_name=section_array, mec=mechanism,
                                                    value_name=value_name,
                                                    value=value, verbool=self.verbool)
                else:
                    raise ValueError('the function doesn\'t exist')
            else:
                raise ValueError('the mechanism is empty')

    def insert_active_basal(self):
        """
        add active channels to basal dendrites referenced to
        Gao PP, Graham JW, Zhou WL, Jang J, Angulo S, Dura-Bernal S, Hines M, Lytton WW, Antic SD. Local glutamate-mediated dendritic plateau potentials change the state of the cortical pyramidal neuron. J Neurophysiol. 2021 Jan 1;125(1):23-42. doi: 10.1152/jn.00734.2019. Epub 2020 Oct 21. PMID: 33085562; PMCID: PMC8087381.

        Returns
        -------
        None.

        """
        # Insert channels and set parameters
        if not hasattr(self, "data"):
            self.data = self.P['data']
        verbool = self.verbool
        soma_seg_idx = self.get_idx('soma[0]')
        for sec in h.allsec():
            if sec.name()[0:4] == 'dend':  # basal dendrites
                sec.insert('na')
                sec.insert('ca')  # HVA-Ca
                sec.insert('it')  # LVA-Ca
                sec.insert('kv')
                sec.insert('kBK')
                sec.gpeak_kBK = 2.68e-4 * 100  # changing from S/cm^2 to mho /Cm^2
                sec.gbar_kv = 40
                if verbool:
                    print('Adding mechanism na to %s' % (sec.name()))
                    print('Adding mechanism ca to %s' % (sec.name()))
                    print('Adding mechanism CaT to %s' % (sec.name()))
                    print('Adding mechanism kv to %s' % (sec.name()))
                    print('Adding mechanism kBK to %s' % (sec.name()))
                    print('Setting gpeak_kBK to 0.0268 in %s' % (sec.name()))
                    print('Setting gbar_kv to 40 in %s' % (sec.name()))
                for seg, seg_idx in zip(sec, self.get_idx(sec.name())):
                    d = self.get_intersegment_distance(soma_seg_idx, seg_idx)
                    gbar_na = 150 - d * 0.5
                    if gbar_na <= 0:
                        gbar_na = 0
                    elif gbar_na >= 2000:
                        gbar_na = 2000
                    if verbool:
                        print('Setting gbar_na to %.6g in %s'
                              % (gbar_na, sec.name()))
                    seg_mec = getattr(seg, 'na')
                    setattr(seg_mec, 'gbar', gbar_na)
                    sec.insert('kad')
                    gbar_kad = (150 + 0.7 * d) * (1 / 300 * d) / 1e4  # change from pS/um2 to mho/cm^2
                    if gbar_kad <= 0:
                        gbar_kad = 0
                    elif gbar_kad >= 2000:
                        gbar_kad = 2000
                    seg_mec = getattr(seg, 'kad')
                    setattr(seg_mec, 'gkabar', gbar_kad)
                    sec.insert('kap')
                    gbar_kap = (150 + 0.7 * d) * (1 - 1 / 300 * d) / 1e4  # change from pS/um2 to mho/cm^2
                    if gbar_kap <= 0:
                        gbar_kap = 0
                    elif gbar_kap >= 2000:
                        gbar_kap = 2000
                    seg_mec = getattr(seg, 'kap')
                    setattr(seg_mec, 'gkabar', gbar_kap)
                    if verbool:
                        print('Adding mechanism kad to %s' % (sec.name()))
                        print('Setting gbar_kad to %.6g in %s' % (gbar_kad, sec.name()))
                        print('Adding mechanism kap to %s' % (sec.name()))
                        print('Setting gbar_kap to %.6g in %s' % (gbar_kap, sec.name()))
                    if d >= 30:  # distal
                        seg_mec = getattr(seg, 'ca')
                        setattr(seg_mec, 'gbar', 0.4)
                        seg_mec = getattr(seg, 'it')
                        setattr(seg_mec, 'gbar', 1.6 / 1e4)  # change from pS/um2 to mho/cm^2
                        if verbool:
                            print('Setting gbar_ca to 0.4 in %s' % (sec.name()))
                            print('Setting gbar_it to 1.6e-4 in %s' % (sec.name()))
                    else:
                        seg_mec = getattr(seg, 'ca')
                        setattr(seg_mec, 'gbar', 2)
                        seg_mec = getattr(seg, 'it')
                        setattr(seg_mec, 'gbar', 2 / 1e4)  # change from pS/um2 to mho/cm^2
                        if verbool:
                            print('Setting gbar_ca to 0.4 in %s' % (sec.name()))
                            print('Setting gbar_it to 2e-4 in %s' % (sec.name()))
                    if d >= 50:  # "spine factor", *2 for gpas and cm
                        seg.g_pas = 3e-5 * 2
                        seg.cm = 2
                    else:
                        seg.g_pas = 3e-5
                        seg.cm = 1

    def insert_active_basal_stochastic(self):
        """
        add active channels to basal dendrites referenced to
        Gao PP, Graham JW, Zhou WL, Jang J, Angulo S, Dura-Bernal S, Hines M, Lytton WW, Antic SD. Local glutamate-mediated dendritic plateau potentials change the state of the cortical pyramidal neuron. J Neurophysiol. 2021 Jan 1;125(1):23-42. doi: 10.1152/jn.00734.2019. Epub 2020 Oct 21. PMID: 33085562; PMCID: PMC8087381.
        na is set as stochastic channels
        Returns
        -------
        None.

        """
        # Insert channels and set parameters
        if not hasattr(self, "data"):
            self.data = self.P['data']
        verbool = self.verbool
        soma_seg_idx = self.get_idx('soma[0]')
        for sec in h.allsec():
            if sec.name()[0:4] == 'dend':  # basal dendrites
                sec.insert('na_2F')
                sec.insert('ca')  # HVA-Ca
                sec.insert('it')  # LVA-Ca
                sec.insert('kv')
                sec.insert('kBK')
                sec.gpeak_kBK = 2.68e-4 * 100  # changing from S/cm^2 to mho /Cm^2
                sec.gbar_kv = 40
                if verbool:
                    print('Adding mechanism na_2F to %s' % (sec.name()))
                    print('Adding mechanism ca to %s' % (sec.name()))
                    print('Adding mechanism CaT to %s' % (sec.name()))
                    print('Adding mechanism kv to %s' % (sec.name()))
                    print('Adding mechanism kBK to %s' % (sec.name()))
                    print('Setting gpeak_kBK to 0.0268 in %s' % (sec.name()))
                    print('Setting gbar_kv to 40 in %s' % (sec.name()))
                for seg, seg_idx in zip(sec, self.get_idx(sec.name())):
                    d = self.get_intersegment_distance(soma_seg_idx, seg_idx)
                    gbar_na = 150 - d * 0.5
                    if gbar_na <= 0:
                        gbar_na = 0
                    elif gbar_na >= 2000:
                        gbar_na = 2000
                    Nna = np.round(gbar_na * np.sum(self.area[self.get_idx(sec.name())]) / 5)
                    setattr(sec, 'Nna_na_2F', Nna)
                    if self.verbool:
                        print('Changing num of channel in mechanism %s in %s to %d'
                              % ('na_2F', sec.name(), Nna))
                    if self.verbool:
                        print('Setting gbar_na to %.6g in %s'
                              % (gbar_na, sec.name()))
                    seg_mec = getattr(seg, 'na_2F')
                    setattr(seg_mec, 'gbar', gbar_na)
                    sec.insert('kad')
                    gbar_kad = (150 + 0.7 * d) * (1 / 300 * d) / 1e4  # change from pS/um2 to mho/cm^2
                    if gbar_kad <= 0:
                        gbar_kad = 0
                    elif gbar_kad >= 2000:
                        gbar_kad = 2000
                    seg_mec = getattr(seg, 'kad')
                    setattr(seg_mec, 'gkabar', gbar_kad)
                    sec.insert('kap')
                    gbar_kap = (150 + 0.7 * d) * (1 - 1 / 300 * d) / 1e4  # change from pS/um2 to mho/cm^2
                    if gbar_kap <= 0:
                        gbar_kap = 0
                    elif gbar_kap >= 2000:
                        gbar_kap = 2000
                    seg_mec = getattr(seg, 'kap')
                    setattr(seg_mec, 'gkabar', gbar_kap)
                    if verbool:
                        print('Adding mechanism kad to %s' % (sec.name()))
                        print('Setting gbar_kad to %.6g in %s' % (gbar_kad, sec.name()))
                        print('Adding mechanism kap to %s' % (sec.name()))
                        print('Setting gbar_kap to %.6g in %s' % (gbar_kap, sec.name()))
                    if d >= 30:  # distal
                        seg_mec = getattr(seg, 'ca')
                        setattr(seg_mec, 'gbar', 0.4)
                        seg_mec = getattr(seg, 'it')
                        setattr(seg_mec, 'gbar', 1.6 / 1e4)  # change from pS/um2 to mho/cm^2
                        if verbool:
                            print('Setting gbar_ca to 0.4 in %s' % (sec.name()))
                            print('Setting gbar_it to 1.6e-4 in %s' % (sec.name()))
                    else:
                        seg_mec = getattr(seg, 'ca')
                        setattr(seg_mec, 'gbar', 2)
                        seg_mec = getattr(seg, 'it')
                        setattr(seg_mec, 'gbar', 2 / 1e4)  # change from pS/um2 to mho/cm^2
                        if verbool:
                            print('Setting gbar_ca to 0.4 in %s' % (sec.name()))
                            print('Setting gbar_it to 2e-4 in %s' % (sec.name()))
                    if d >= 50:  # "spine factor", *2 for gpas and cm
                        seg.g_pas = 3e-5 * 2
                        seg.cm = 2
                    else:
                        seg.g_pas = 3e-5
                        seg.cm = 1

    def set_deficit_NMDA(self, sec_name = 'all', percentage = 0.0):
        if sec_name == 'all':
            self.w_1 = self.sec_e.shape[1] * [P['g_max_A']]
            self.w_2 = self.sec_e.shape[1] * [percentage*P['g_max_N']]
            self.w_3 = self.sec_i.shape[1] * [P['g_max_G']]
        else:
            r_na = self.sec_e.shape[1] * [self.r_na]
            for i, s in enumerate(self.NMDA_meta):
                if sec_name in s['sec_name']:
                    self.w_2[i] = percentage*P['g_max_N']
                    r_na[i] = percentage*r_na
            self.r_na = r_na

    def set_deficite_channels(self, mec_name, sec_name = 'all',  percentage = 0.3):
        """

        set conductance of certain channel to a certain percentage of original channel
        :param mec_name:
        :param prop_name:
        :param sec_name:
        :param percentage:
        :return:
        """
        prop_name = 'g' + mec_name + 'bar'
        if sec_name == 'all':
            for sec in self.allseclist:
                for seg in sec:
                    try:
                        seg_mec = getattr(seg, str(mec_name))
                        current_value = getattr(seg_mec, prop_name)
                        setattr(seg_mec, prop_name, current_value*percentage)
                        if self.verbool:
                            print('change %s in %s from %.6g to %.6g' %(prop_name, sec.name(), current_value, current_value*percentage))
                    except AttributeError:
                        pass
        else:
            for sec in self.allseclist:
                if sec.name()[0:4] in sec_name:
                    for seg in sec:
                        try:
                            seg_mec = getattr(seg, str(mec_name))
                            current_value = getattr(seg_mec, prop_name)
                            setattr(seg_mec, prop_name, current_value * percentage)
                            if self.verbool:
                                print('change %s in %s from %.6g to %.6g' % (
                                prop_name, sec.name(), current_value, current_value * percentage))
                        except AttributeError:
                            pass

    def set_weights(self, w_e, w_i):
        """Assign AMPA and NMDA weights with ratio r_n, and GABA weights.

        Parameters
        ----------
        w_e, w_i : E and I synaptic weights
        """
        self.w_e = w_e
        self.w_i = w_i
        if not isinstance(self.r_na, list):
            r = self.r_na
            self.w_1 = w_e / (1 + r)
            self.w_2 = w_e * r / (1 + r)
        else:
            if len(self.r_na) < len(self.w_1):
                r = self.r_na[0]
                self.w_1 = w_e / (1 + r)
                self.w_2 = w_e * r / (1 + r)
            else:
                for i in len(self.w_1):
                    r = self.r_na[i]
                    self.w_1[i] = w_e[i]/(1 + r)
                    self.w_2[i] = w_e[i] * r / (1 + r)

        self.w_3 = w_i

    def create_stim(self, S, syns, weights):
        """Create vecstim and netcon objects for simulation.

        Parameters
        ----------
        S : ndarray
            presynaptic spike times
        syns : list
            neuron synapse objects
        weights : ndarray
            synaptic weights

        Returns
        -------
        t_vec_list : list
            spike time vectors
        stim_list : list
            vec_stim objects
        con_list : list
            netcon objects
        """
        t_vec_list = []
        stim_list = []
        con_list = []
        for k, (t_list, syn) in enumerate(zip(S, syns)):
            stim = h.VecStim()
            t_vec = h.Vector(t_list[~np.isinf(t_list)])
            stim.play(t_vec)
            t_vec_list.append(t_vec)
            stim_list.append(stim)
            con_list.append(h.NetCon(stim, syn, 0, 0, weights[k]))
        return t_vec_list, stim_list, con_list

    def simulate(self, T, dt, v_init, S_e, S_i, I_inj=0, break_flag=False, inj_site='soma[0]'):
        """Run simulation and record voltage from every compartment.

        Parameters
        ----------
        T : int
            total simulation time (ms)
        dt : float
            time step (ms)
        v_init : float
            initial voltage (mV)
        S_e, S_i : array_like
            presynaptic spike times for each E and I synapse
        I_inj : int
            injected current at soma (default 0)
        break_flag : bool
            interupt simulation at time of first spike (default False)

        Returns
        -------
        t : ndarray
            time vector
        v : ndarray
            voltage vector
        """

        h.dt = dt
        self.dt = dt
        h.steps_per_ms = 1.0 / h.dt
        t_a, stim_a, self.ampa = self.create_stim(S_e, self.AMPA, self.w_1)
        t_n, stim_n, self.nmda = self.create_stim(S_e, self.NMDA, self.w_2)
        t_g, stim_g, self.gaba = self.create_stim(S_i, self.GABA, self.w_3)
        t = h.Vector()
        t.record(h._ref_t, dt)
        T_len = np.round(T/h.dt) + 1
        v = []
        i = []
        # ina = []
        # ik = []
        # ica = []
        # ihcn = []
        # iGABA = []
        # iAMPA = []
        # iNMDA = []
        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError as ae:
            raise Exception('neuron.h.CVode().use_fast_imem() method not found. Please update NEURON to v.7.4 or newer')
        for sec in self.tree:
            for k in range(sec.nseg):
                v_temp = h.Vector()
                v_temp.record(sec((k + 1 / 2) / sec.nseg)._ref_v, dt)
                v.append(v_temp)
                i_temp = h.Vector()
                i_temp.record(sec((k + 1 / 2) / sec.nseg)._ref_i_membrane_, dt)
                i.append(i_temp)
        if np.abs(I_inj) > 0:
            stim = h.IClamp(0.5, sec=self.tree[self.allsecnames.index(inj_site)])
            stim.dur = T - 150
            stim.amp = I_inj
            stim.delay = 100
        if break_flag:
            nc = h.NetCon(self.tree[0](0.5)._ref_v, None, sec=self.tree[0])
            nc.threshold = 0
            nc.record(self.halt)

        h.v_init = v_init
        h.tstop = T
        h.run()
        t = np.array(t)
        v = np.array(v)
        self.imem = np.asarray(i)
        self.vmem = v
        self.tvec = t
        return t, v

    def _parent_and_segment_current(self, seg_idx, parent_idx, bottom_seg,
                                    branch=False, parentsec=None,
                                    children_dict=None, connection_dict=None,
                                    conn_point=1, timepoints=None):
        """
        Return axial current from segment (seg_idx) mid to segment start,
        and current from parent segment (parent_idx) end to parent segment mid.
        Parameters
        ----------
        seg_idx : int
            Segment index
        parent_idx : int
            Parent index
        parent_ri : float
            Axial resistance from parent end to mid in units of (MΩ)
        bottom_seg : boolean
        branch : boolean
        parentsec : neuron.Section object
        timepoints : ndarray, dtype=int
            array of timepoints in simulation at which you want to compute
            the axial currents. Defaults to None. If not given,
            all simulation timesteps will be included.
        Returns
        -------
        iseg : dtype=float
            Axial current in units of (nA)
            from segment mid point to segment start point.
        ipar : dtype=float
            Axial current in units of (nA)
            from parent segment end point to parent segment mid point.
        """
        # list of axial resistance between segments
        ri_list = self.get_axial_resistance()
        # axial resistance between segment mid and parent node
        seg_ri = ri_list[seg_idx]
        vmem = self.vmem
        if timepoints is not None:
            vmem = self.vmem[:,timepoints]
        vpar = vmem[parent_idx]
        vseg = vmem[seg_idx]
        # if segment is the first in its section and it is connected to
        # top or bottom of parent section, we need to find parent_ri explicitly
        if bottom_seg and (conn_point == 0 or conn_point == 1):
            if conn_point == 0:
                parent_ri = ri_list[parent_idx]
            else:
                parent_ri = neuron.h.ri(0)
            if not branch:
                ri = parent_ri + seg_ri
                iseg = (vpar - vseg) / ri
                ipar = iseg
            else:
                # if branch, need to compute iseg and ipar separately
                [sib_idcs] = np.take(children_dict[parentsec.name()],
                                  np.where(children_dict[parentsec.name()]
                                           != seg_idx))
                sibs = [self.get_idx_name(sib_idcs)[i][1] for i in range(len(sib_idcs))]
                # compute potential in branch point between parent and siblings
                v_branch_num = vpar/parent_ri + vseg/seg_ri
                v_branch_denom = 1./parent_ri + 1./seg_ri
                for sib_idx, sib in zip(sib_idcs, sibs):
                    sib_conn_point = connection_dict[sib]
                    if sib_conn_point == conn_point:
                        v_branch_num += vmem[sib_idx]/ri_list[sib_idx]
                        v_branch_denom += 1./ ri_list[sib_idx]
                v_branch = v_branch_num/v_branch_denom
                iseg = (v_branch - vseg)/seg_ri
                # set ipar=iseg
                # only fraction of total current into parent is added per sibling
                ipar = iseg
        else:
            iseg = (vpar - vseg) / seg_ri
            ipar = iseg
        return iseg, ipar

    def get_axial_currents_from_vmem(self, timepoints=None):
        """Compute axial currents from cell sim: get current magnitude,
        distance vectors and position vectors.
        Parameters
        ----------
        timepoints : ndarray, dtype=int
            array of timepoints in simulation at which you want to compute
            the axial currents. Defaults to False. If not given,
            all simulation timesteps will be included.
        Returns
        -------
        i_axial : ndarray, dtype=float
            Shape ((cell.totnsegs-1)*2, len(timepoints)) array of axial current
            magnitudes I in units of (nA) in cell at all timesteps in timepoints,
            or at all timesteps of the simulation if timepoints=None.
            Contains two current magnitudes per segment,
            (except for the root segment): 1) the current from the mid point of
            the segment to the segment start point, and 2) the current from
            the segment start point to the mid point of the parent segment.
        d_vectors : ndarray, dtype=float
            Shape ((cell.totnsegs-1)*2, 3) array of distance vectors traveled by
            each axial current in i_axial in units of (µm). The indices of the
            first axis, correspond to the first axis of i_axial and pos_vectors.
        pos_vectors : ndarray, dtype=float
            Shape ((cell.totnsegs-1)*2, 3) array of position vectors pointing to
            the mid point of each axial current in i_axial in units of (µm). The
            indices of the first axis, correspond to the first axis
            of i_axial and d_vectors.

        Raises
        ------
        AttributeError
            Raises an exeption if the cell.vmem attribute cannot be found
        """
        if not hasattr(self, 'vmem'):
            raise AttributeError('no vmem, run cell.simulate(rec_vmem=True)')
        i_axial = []
        d_vectors = []
        pos_vectors = []
        dseg = np.c_[self.xmid - self.xstart,
                     self.ymid - self.ystart,
                     self.zmid - self.zstart]
        dpar = np.c_[self.xend - self.xmid,
                     self.yend - self.ymid,
                     self.zend - self.zmid]

        children_dict = self.get_dict_of_children_idx()
        for sec in self.allseclist:
            if not neuron.h.SectionRef(sec.name()).has_parent():
                if sec.nseg == 1:
                    # skip soma, since soma is an orphan
                    continue
                else:
                    # the first segment has more than one segment,
                    # need to compute axial currents within this section.
                    seg_idx = 1
                    parent_idx = 0
                    bottom_seg = False
                    first_sec = True
                    branch = False
                    parentsec = None
                    children_dict = None
                    connection_dict = None
                    conn_point = 1
            else:
                # section has parent section
                first_sec = False
                bottom_seg = True
                secref = neuron.h.SectionRef(sec.name())
                parentseg = secref.parent()
                parentsec = parentseg.sec
                children_dict = self.get_dict_of_children_idx()
                branch = len(children_dict[parentsec.name()]) > 1
                connection_dict = self.get_dict_parent_connections()
                conn_point = connection_dict[sec.name()]
                # find parent index
                if conn_point == 1 or parentsec.nseg == 1:
                    internal_parent_idx = -1 # last seg in sec
                elif conn_point == 0:
                    internal_parent_idx = 0 # first seg in sec
                else:
                    # if parentseg is not first or last seg in parentsec
                    segment_xlist = np.array([segment.x for segment in parentsec])
                    internal_parent_idx = np.abs(segment_xlist - conn_point).argmin()
                parent_idx = self.get_idx(section=parentsec.name())[internal_parent_idx]
                # find segment index
                seg_idx = self.get_idx(section=sec.name())[0]
            for _ in sec:
                if first_sec:
                    first_sec = False
                    continue
                iseg, ipar = self._parent_and_segment_current(seg_idx,
                                                              parent_idx,
                                                              bottom_seg,
                                                              branch,
                                                              parentsec,
                                                              children_dict,
                                                              connection_dict,
                                                              conn_point,
                                                              timepoints
                                                              )

                if bottom_seg:
                    # if a seg is connected to soma, it is
                    # connected to the middle of soma,
                    # and dpar needs to be altered.
                    par_dist = np.array([(self.xstart[seg_idx] -
                                        self.xmid[parent_idx]),
                                        (self.ystart[seg_idx] -
                                        self.ymid[parent_idx]),
                                        (self.zstart[seg_idx] -
                                        self.zmid[parent_idx])])

                else:
                    par_dist = dpar[parent_idx]
                d_vectors.append(par_dist)
                d_vectors.append(dseg[seg_idx])
                i_axial.append(ipar)
                i_axial.append(iseg)

                pos_par = np.array([self.xstart[seg_idx],
                                    self.ystart[seg_idx],
                                    self.zstart[seg_idx]]) - 0.5*par_dist

                pos_seg = np.array([self.xmid[seg_idx],
                                    self.ymid[seg_idx],
                                    self.zmid[seg_idx]]) - 0.5*dseg[seg_idx]
                pos_vectors.append(pos_par)
                pos_vectors.append(pos_seg)

                parent_idx = seg_idx
                seg_idx += 1
                branch = False
                bottom_seg = False
                parent_ri = 0
        return np.array(i_axial), np.array(d_vectors), np.array(pos_vectors)

    def get_dict_parent_connections(self):
        """
        Return dictionary with parent connection point for all sections.
        Returns
        -------
        connection_dict : dictionary
            Dictionary containing a float in range [0, 1] for each section
            in cell. The float gives the location on the parent segment
            to which the section is connected.
            The dictionary is needed for computing axial currents.
        """
        connection_dict = {}
        for sec in self.allseclist:
            connection_dict[sec.name()] = neuron.h.parent_connection()
        return connection_dict

    def get_dict_of_children_idx(self):
        """
        Return dictionary with children segment indices for all sections.
        Returns
        -------
        children_dict : dictionary
            Dictionary containing a list for each section,
            with the segment index of all the section's children.
            The dictionary is needed to find the
            sibling of a segment.
        """
        children_dict = {}
        for sec in self.allseclist:
            children_dict[sec.name()] = []
            for child in neuron.h.SectionRef(sec.name()).child:
                # add index of first segment of each child
                children_dict[sec.name()].append(int(self.get_idx(
                    section=child.name())[0]))

        return children_dict

    def get_axial_resistance(self):
        """
        Return NEURON axial resistance for all cell compartments.
        Returns
        -------
        ri_list : ndarray, dtype=float
            Shape (cell.totnsegs, ) array containing neuron.h.ri(seg.x) in units
            of (MOhm) for all segments in cell calculated using the
            neuron.h.ri(seg.x) method. neuron.h.ri(seg.x) returns the
            axial resistance from the middle of the segment to the middle of
            the parent segment. Note: If seg is the first segment in a section,
            i.e. the parent segment belongs to a different section or there is
            no parent section, then neuron.h.ri(seg.x) returns the axial
            resistance from the middle of the segment to the node connecting the
            segment to the parent section (or a ghost node if there is no
            parent)
        """

        ri_list = np.zeros(self.totnsegs)
        comp = 0
        for sec in self.allseclist:
            for seg in sec:
                ri_list[comp] = neuron.h.ri(seg.x)
                comp += 1

        return ri_list
    def get_multi_current_dipole_moments(self, timepoints=None):
        '''
        Return 3D current dipole moment vector and middle position vector
        from each axial current in space.
        Parameters
        ----------
        timepoints : ndarray, dtype=int
            array of timepoints at which you want to compute
            the current dipole moments. Defaults to None. If not given,
            all simulation timesteps will be included.
        Returns
        -------
        multi_dipoles : ndarray, dtype = float
            Shape (n_axial_currents, n_timepoints, 3) array
            containing the x-,y-,z-components of the current dipole moment
            from each axial current in cell, at all timepoints.
            The number of axial currents, n_axial_currents = (cell.totnsegs-1)*2
            and the number of timepoints, n_timepoints = cell.tvec.size.
            The current dipole moments are given in units of (nA µm).
        pos_axial : ndarray, dtype = float
            Shape (n_axial_currents, 3) array containing the x-, y-, and
            z-components giving the mid position in space of each multi_dipole
            in units of (µm).
        Examples
        --------
        Get all current dipole moments and positions from all axial currents in a
        single neuron simulation.
        >>> import LFPy
        >>> import numpy as np
        >>> cell = LFPy.Cell('PATH/TO/MORPHOLOGY', extracellular=False)
        >>> syn = LFPy.Synapse(cell, idx=cell.get_closest_idx(0,0,1000),
        >>>                   syntype='ExpSyn', e=0., tau=1., weight=0.001)
        >>> syn.set_spike_times(np.mgrid[20:100:20])
        >>> cell.simulate(rec_vmem=True, rec_imem=False)
        >>> timepoints = np.array([1,2,3,4])
        >>> multi_dipoles, dipole_locs = cell.get_multi_current_dipole_moments(timepoints=timepoints)
        '''
        i_axial, d_axial, pos_axial = self.get_axial_currents_from_vmem(timepoints=timepoints)
        Ni, Nt = i_axial.shape
        multi_dipoles = np.array([i_axial[i][:, np.newaxis]*d_axial[i] for i in range(Ni)])

        return multi_dipoles, pos_axial

    def halt(self):
        """Interupt simulation. Use with Netcon.record """
        if h.t > 100:
            h.stoprun = 1





