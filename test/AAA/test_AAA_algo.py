# -*- coding: utf-8 -*-
"""
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import pytest
import numpy as np
from os import path

from IIRrational.pytest import (  # noqa: F401
    tpath_join, plot, pprint, tpath, tpath_preclear
)
from IIRrational.v2.__main__ import IIRrationalV2fit
from IIRrational.representations import ZPKTF, asZPKTF

from IIRrational.utilities.mpl import mplfigB


def test_AAA_sym(tpath_join, tpath_preclear, pprint):
    ZPK1 = asZPKTF(((-.1+5j, -0.1-5j), (-2, -2), 1))
    F_Hz = np.linspace(0, 10, 100)
    sF_Hz = 1j*F_Hz
    TF1 = ZPK1.xfer_eval(F_Hz=F_Hz)

    axB = mplfigB()

    fvals = []
    zvals = []
    wvals = []
    for Fidx in [0, 20]:
        fvals.append(TF1[Fidx])
        zvals.append(sF_Hz[Fidx])
        #fvals.append(TF1[Fidx].conjugate())
        #zvals.append(-sF_Hz[Fidx])

    Vn = []
    Vd = []

    Widx = 0
    bad_Fidx = []
    while Widx < len(fvals):
        z = zvals[Widx]
        f = fvals[Widx]
        if z == 0:
            assert(f.imag == 0)
            bary_D = (sF_Hz - z)
            select_bad = abs(bary_D) < 1e-13
            bad_Fidx.extend(np.argwhere(select_bad)[:, 0])
            bary_D[select_bad] = float('NaN')
            bary_D = 1/bary_D
            with np.errstate(divide='ignore', invalid='ignore'):
                Vn.append(f * bary_D)
                Vd.append(bary_D)
        else:
            bary_D = (sF_Hz - z)
            select_bad = abs(bary_D) < 1e-13
            bad_Fidx.extend(np.argwhere(select_bad)[:, 0])
            bary_D[select_bad] = float('NaN')

            bary_D2 = (sF_Hz - z.conjugate())
            select_bad = abs(bary_D2) < 1e-13
            bad_Fidx.extend(np.argwhere(select_bad)[:, 0])
            bary_D2[select_bad] = float('NaN')

            #with np.errstate(divide='ignore', invalid='ignore'):
            #    Vn.append(f / bary_D)
            #    Vd.append(1/bary_D)
            #    Vn.append(f.conjugate() / bary_D2)
            #    Vd.append(1/bary_D2)

            with np.errstate(divide='ignore', invalid='ignore'):
                Vn.append(f / bary_D + f.conjugate() / bary_D2)
                Vd.append(1/bary_D + 1/bary_D2)
                Vn.append(-1j*(f / bary_D - f.conjugate() / bary_D2))
                Vd.append(-1j*(1/bary_D - 1/bary_D2))
        Widx += 1
    Vn = np.asarray(Vn).T
    Vd = np.asarray(Vd).T
    pprint(bad_Fidx)
    Vd[bad_Fidx, :] = 0
    Vn[bad_Fidx, :] = 0
    Hd1 = Vd * TF1.reshape(-1, 1)
    Hn1 = Vn
    Hd2 = Vd
    Hn2 = Vn / TF1.reshape(-1, 1)
    Hs1 = Hd1-Hn1
    Hs2 = Hd2-Hn2

    Hblock = []
    Hblock.append([Hs1.real])
    Hblock.append([Hs1.imag])
    Hblock.append([Hs2.real])
    Hblock.append([Hs2.imag])
    SX1 = np.block(Hblock)
    u,s,v = np.linalg.svd(SX1)
    pprint(s)
    pprint(v)
    Ws = v[-1, :].conjugate()
    pprint(Ws)
    #Ws = np.array(wvals)
    pprint("Ws", Ws)
    #Ws = [1 + 1j, 1j - 1j]

    N = Vn @ Ws
    D = Vd @ Ws

    TF2 = N / D
    pprint(TF2.shape)

    axB.ax0.semilogy(F_Hz, abs(TF1))
    axB.ax0.semilogy(F_Hz, abs(TF2))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 0]))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 1]))
    #axB.ax0.semilogy(F_Hz, abs())
    #axB.ax0.semilogy(F_Hz, abs(N))
    #axB.ax0.semilogy(F_Hz, abs(D))
    axB.save(tpath_join('test'))
    return


def test_AAA2(tpath_join, tpath_preclear, pprint):
    from aaa import aaa
    ZPK1 = asZPKTF(((-.1+5j, -0.1-5j), (-2, -2), 1))
    F_Hz = np.linspace(-10, 10, 100)
    sF_Hz = 1j*F_Hz
    TF1 = ZPK1.xfer_eval(F_Hz=F_Hz)

    r = aaa(TF1, sF_Hz)
    axB = mplfigB()
    pprint(r.nodes)
    pprint(r.values)
    pprint(r.weights)

    axB.ax0.semilogy(F_Hz, abs(TF1))
    axB.ax0.semilogy(F_Hz, abs(r(sF_Hz)))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 0]))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 1]))
    #axB.ax0.semilogy(F_Hz, abs())
    #axB.ax0.semilogy(F_Hz, abs(N))
    #axB.ax0.semilogy(F_Hz, abs(D))
    axB.save(tpath_join('test'))
    return

def test_AAA3(tpath_join, tpath_preclear, pprint):
    import matlab
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath('~/local/projects_sync/matlab/chebfun/')


    from aaa import aaa
    ZPK1 = asZPKTF(((-.1+5j, -0.1-5j), (-2, -2), 1))
    F_Hz = np.linspace(0, 10, 100)
    sF_Hz = 1j*F_Hz
    TF1 = ZPK1.xfer_eval(F_Hz=F_Hz)

    def mc(a):
        r = matlab.double([float(v.real) for v in a])
        i = matlab.double([float(v.imag) for v in a])
        c = eng.complex(r, i)
        return c
    
    r, pol, res, zer, zj, fj, wj = eng.aaa(mc(TF1), mc(sF_Hz), nargout=7)
    eng.workspace['z'] = mc(sF_Hz)
    eng.workspace['r'] = r
    tf2 = np.array(eng.eval('r(z);')).reshape(-1)
    pprint('z', zj)
    pprint('f', fj)
    pprint('w', wj)

    axB = mplfigB()
    axB.ax0.semilogy(F_Hz, abs(TF1))
    axB.ax0.semilogy(F_Hz, abs(tf2))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 0]))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 1]))
    #axB.ax0.semilogy(F_Hz, abs())
    #axB.ax0.semilogy(F_Hz, abs(N))
    #axB.ax0.semilogy(F_Hz, abs(D))
    axB.save(tpath_join('test'))
    return


def test_AAA_success(tpath_join, tpath_preclear, pprint):
    ZPK1 = asZPKTF(((-.1+5j, -0.1-5j), (-2, -2), 1))
    F_Hz = np.linspace(0, 10, 100)
    sF_Hz = 1j*F_Hz
    TF1 = ZPK1.xfer_eval(F_Hz=F_Hz)

    axB = mplfigB()

    fvals = (6.252499999999998+0j), (-0.36392531927705213-1.7590404622436544j), (0.6729881656804731+0.25957840236686386j)
    zvals = [0j, 2.5252525252525252j, 10j]
    wvals = [(-0.10529216178197665+0j), (-0.08370482530961398+0.35571676277694747j), (0.8537202306646741-0.355716762776946j)]
    #for Fidx in [0, 20, 40]:
    #    fvals.append(TF1[Fidx])
    #    zvals.append(sF_Hz[Fidx])
    #    #fvals.append(TF1[Fidx].conjugate())
    #    #zvals.append(-sF_Hz[Fidx])

    Vn = np.empty((len(F_Hz), len(fvals), ), dtype = complex)
    Vd = np.empty((len(F_Hz), len(fvals), ), dtype = complex)

    Widx = 0
    bad_Fidx = []
    while Widx < len(fvals):
        bary_D = (sF_Hz - zvals[Widx])
        bary_D[abs(bary_D) < 1e-13] = float('NaN')
        with np.errstate(divide='ignore', invalid='ignore'):
            Vn[:, Widx] = fvals[Widx] / bary_D
            Vd[:, Widx] = 1 / bary_D
        bad_Fidx.extend(np.argwhere(~np.isfinite(bary_D))[:, 0])
        Widx += 1
    pprint(bad_Fidx)
    Vd[bad_Fidx, :] = 0
    Vn[bad_Fidx, :] = 0
    Hd1 = Vd * TF1.reshape(-1, 1)
    Hn1 = Vn
    Hd2 = Vd
    Hn2 = Vn / TF1.reshape(-1, 1)

    SX1 = np.block([[Hd1-Hn1]])
    u,s,v = np.linalg.svd(SX1)
    pprint(s)
    pprint(v)
    Ws = v[-1, :].conjugate()
    pprint(Ws)
    #Ws = np.array(wvals)
    pprint("Ws", Ws)
    #Ws = [1 + 1j, 1j - 1j]

    N = Vn @ Ws
    D = Vd @ Ws

    TF2 = N / D
    pprint(TF2.shape)

    axB.ax0.semilogy(F_Hz, abs(TF1))
    axB.ax0.semilogy(F_Hz, abs(TF2))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 0]))
    #axB.ax0.semilogy(F_Hz, abs(Vn[:, 1]))
    #axB.ax0.semilogy(F_Hz, abs())
    #axB.ax0.semilogy(F_Hz, abs(N))
    #axB.ax0.semilogy(F_Hz, abs(D))
    axB.save(tpath_join('test'))
    return

