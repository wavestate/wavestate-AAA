# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import pytest
import numpy as np
from os import path
import scipy.signal

from IIRrational.pytest import (  # noqa: F401
    tpath_join, plot, pprint, tpath, tpath_preclear, Timer
)
from IIRrational.representations import asZPKTF
from IIRrational.testing import IIRrational_data

from IIRrational.utilities.mpl import mplfigB
from IIRrational.AAA import tfAAA


def test_AAA_mod(tpath_join, tpath_preclear, pprint):
    ZPK1 = asZPKTF(
        ((
            -.1+.5j, -0.1-.5j,
            -.1+5j, -0.1-5j,
            -.1+50j, -0.1-50j,
            -4,
        ), (
            -1, -2,
            -10,
            -2+10j, -2-10j,
        ), .001
        ))
    F_Hz = np.linspace(0, 60, 200)
    TF1 = ZPK1.xfer_eval(F_Hz=F_Hz)
    TF1 = 1/(1 - TF1)

    results = tfAAA(
        F_Hz = F_Hz,
        xfer = TF1,
        #lf_eager = True,
        #degree_max = 20,
        #nconv = 1,
        #nrel = 10,
        #rtype = 'log',
        #supports = (1e-2, 1e-1, 4.2e-1, 5.5e-1, 1.5, 2.8, 1, 5e-1, 2),
    )
    pprint("weights", results.wvals)
    pprint('poles', results.poles)
    pprint('zeros', results.zeros)
    pprint('gain', results.gain)

    TF2 = results(F_Hz)
    axB = mplfigB(Nrows = 2)
    axB.ax0.loglog(F_Hz, abs(TF1))
    axB.ax0.semilogy(F_Hz, abs(TF2))
    axB.ax1.semilogx(F_Hz, np.angle(TF1, deg = True))
    axB.ax1.semilogx(F_Hz, np.angle(TF2, deg = True))
    for z in results.supports:
        axB.ax0.axvline(z)
    axB.save(tpath_join('test'))
    return


@pytest.mark.parametrize('set_num', [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    9,  # pytest.param(9, marks=pytest.mark.xfail(reason="dynamic range")),
    10,
])
def test_AAA_rand6_lin(test_trigger, tpath_join, tpath_preclear, pprint, plot, set_num):
    data = IIRrational_data("rand6_lin100E", set_num = set_num)
    F_Hz = data.F_Hz
    TF1 = data.rep_s.data

    results = tfAAA(
        F_Hz = F_Hz,
        xfer = TF1,
        lf_eager = True,
        degree_max = 20,
        nconv = 1,
        nrel = 10,
        rtype = 'log',
        #supports = (1e-2, 1e-1, 4.2e-1, 5.5e-1, 1.5, 2.8, 1, 5e-1, 2),
    )
    pprint("weights", results.wvals)
    pprint("supports", results.supports)
    pprint('poles', results.poles)
    pprint('zeros', results.zeros)
    pprint('gain', results.gain)

    _, TF3 = scipy.signal.freqs_zpk(results.zeros, results.poles, results.gain, worN = F_Hz)

    TF2 = results(F_Hz)

    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.semilogy(F_Hz, abs(TF1))
        axB.ax0.semilogy(F_Hz, abs(TF2))
        axB.ax0.semilogy(F_Hz, abs(TF3))
        axB.ax1.semilogx(F_Hz, np.angle(TF1, deg = True))
        axB.ax1.semilogx(F_Hz, np.angle(TF2, deg = True))
        axB.ax1.semilogx(F_Hz, np.angle(TF3, deg = True))
        for z in results.supports:
            axB.ax0.axvline(z)
        axB.save(tpath_join('test'))

        axB = mplfigB(Nrows = 2)
        axB.ax0.semilogy(F_Hz, abs(TF2 / TF1))
        axB.ax0.semilogy(F_Hz, abs(TF3 / TF1))
        axB.ax1.semilogx(F_Hz, np.angle(TF2 / TF1, deg = True))
        axB.ax1.semilogx(F_Hz, np.angle(TF3 / TF1, deg = True))
        if fail:
            axB.save(tpath_join('test_fail'))
    with test_trigger(trigger, plot = plot):
        pprint(TF2 / TF1)
        pprint(TF3 / TF1)
        np.testing.assert_allclose(TF2 / TF1, 1, rtol = 1e-3)
        np.testing.assert_allclose(np.quantile(abs(TF3 / TF1 - 1), .95), 0, rtol = 0, atol = 1e-4)
    return



@pytest.mark.parametrize('set_num', [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    pytest.param(9, marks=pytest.mark.xfail(reason="dynamic range")),
    10,
])
def test_AAA_rand6_log(test_trigger, tpath_join, tpath_preclear, pprint, plot, set_num):
    data = IIRrational_data("rand6_log100E", set_num = set_num)
    F_Hz = data.F_Hz
    TF1 = data.rep_s.data

    results = tfAAA(
        F_Hz = F_Hz,
        xfer = TF1,
        lf_eager = True,
        degree_max = 20,
        nconv = 2,
        nrel = 10,
        s_tol = 0,
        rtype = 'log',
        #supports = (1e-2, 1e-1, 4.2e-1, 5.5e-1, 1.5, 2.8, 1, 5e-1, 2),
    )
    pprint("weights", results.wvals)
    pprint("supports", results.supports)

    pprint('poles true', data.rep_s.poles.fullplane)
    pprint('zeros true', data.rep_s.zeros.fullplane)
    pprint('gain true', data.rep_s.gain)

    pprint('poles fit', results.poles)
    pprint('zeros fit', results.zeros)
    pprint('gain fit', results.gain)

    #assert(len(data.rep_s.poles.fullplane) == len(results.poles))
    #assert(len(data.rep_s.zeros.fullplane) == len(results.zeros))

    _, TF3 = scipy.signal.freqs_zpk(results.zeros, results.poles, results.gain, worN = F_Hz)

    TF2 = results(F_Hz)

    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(TF1))
        axB.ax0.semilogy(F_Hz, abs(TF2))
        axB.ax0.semilogy(F_Hz, abs(TF3))
        axB.ax1.semilogx(F_Hz, np.angle(TF1, deg = True))
        axB.ax1.semilogx(F_Hz, np.angle(TF2, deg = True))
        axB.ax1.semilogx(F_Hz, np.angle(TF3, deg = True))
        for z in results.supports:
            axB.ax0.axvline(z)
        axB.save(tpath_join('test'))

        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(TF2 / TF1))
        axB.ax0.semilogy(F_Hz, abs(TF3 / TF1))
        axB.ax1.semilogx(F_Hz, np.angle(TF2 / TF1, deg = True))
        axB.ax1.semilogx(F_Hz, np.angle(TF3 / TF1, deg = True))
        if fail:
            axB.save(tpath_join('test_fail'))
    with test_trigger(trigger, plot = plot):
        pprint('TF2/TF1 - 1', TF2 / TF1 - 1)
        pprint('TF3/TF1 - 1', TF3 / TF1 - 1)
        np.testing.assert_allclose(np.quantile(abs(TF2 / TF1 - 1), .95), 0, rtol = 0, atol = 1e-3)
        np.testing.assert_allclose(np.quantile(abs(TF3 / TF1 - 1), .95), 0, rtol = 0, atol = 1e-3)
    return
