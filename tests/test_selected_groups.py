"""Spot-check a handful of well-known MSGs against published tables.

These tests require spglib and exercise spglib_source directly.
"""

import pytest
import numpy as np

from msgjson.spglib_source import get_metadata, get_operators


# P1  (type I, trivially the ordinary SG)
def test_P1_metadata():
    meta = get_metadata(1)
    assert meta["bns_number"] == "1.1"
    assert meta["og_number"] == "1.1.1"
    assert meta["type"] == 1
    assert meta["parent_sg"] == 1


def test_P1_has_identity():
    ops = get_operators(1)
    identities = [
        o
        for o in ops
        if np.array_equal(o["W"], np.eye(3, dtype=int).tolist())
        and np.allclose(o["t"], [0, 0, 0])
        and o["theta"] == 1
    ]
    assert len(identities) == 1


# P-1' (type II grey group – all ops repeated with theta=-1)
def test_grey_group_has_antiunitary_identity():
    meta = get_metadata(2)
    assert meta["type"] == 2
    ops = get_operators(2)
    thetas = {o["theta"] for o in ops}
    assert -1 in thetas and 1 in thetas


# Pn'ma' (BNS 62.448) – classic MSG for MnF2, FeF2, etc.
@pytest.mark.parametrize(
    "bns_number,expected_type",
    [
        ("62.448", 3),  # Pn'ma' – type III (primed glides, same lattice)
    ],
)
def test_known_msg_type(bns_number, expected_type):
    # Find UNI number by scanning (small cost, only in tests)
    from msgjson.spglib_source import N_MSG

    for uni in range(1, N_MSG + 1):
        meta = get_metadata(uni)
        if meta["bns_number"] == bns_number:
            assert meta["type"] == expected_type
            return
    pytest.fail(f"{bns_number} not found in database")


def test_operator_closure_P1():
    """The single-element group {E} is closed."""
    ops = get_operators(1)
    assert len(ops) == 1
    W = np.array(ops[0]["W"])
    assert np.linalg.det(W) == pytest.approx(1.0)
