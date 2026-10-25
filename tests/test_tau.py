import logging

import numpy as np
import pytest

import cami


@pytest.fixture
def sine():
    t = np.arange(2000)
    return np.sin(2 * np.pi * t / 42)  # autocorrelation: +0.075 at lag 10, -0.075 at lag 11


def test_auto_tau_is_first_zero_of_autocorrelation(sine):
    assert cami.cami(sine, sine, tau=None) == cami.cami(sine, sine, tau=11)
    assert cami.entropy(sine, tau=None, symbolic_length=2) == cami.entropy(sine, tau=11, symbolic_length=2)


def test_auto_tau_is_logged_not_printed(sine, capsys, caplog):
    with caplog.at_level(logging.INFO, logger='cami'):
        cami.mutual_info(sine, sine, tau=None)
    assert capsys.readouterr().out == ''
    assert 'tau=11' in caplog.text


def test_pointwise_does_not_print(x, y, capsys, caplog):
    with caplog.at_level(logging.INFO, logger='cami'):
        cami.pointwise(x, y, make_plot=False)
    assert capsys.readouterr().out == ''
    assert 'Causal Mutual Information' in caplog.text
