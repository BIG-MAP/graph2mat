import pytest
import torch
from e3nn import o3

from graph2mat.bindings.e3nn import ReducedTensorProducts


@pytest.fixture(
    params=[
        "1x0e",
        "1x1o",
        "2x0e + 1x2o",
        "2x0e + 2x2o",
        "1x0e + 1x1o + 1x2e",
        "1x0e + 2x3o",
        "2x2e",
        "1x0e + 1x1o + 2x2e",
        "2x0e + 2x1o + 2x2e + 2x3o",
    ]
)
def i_irreps(request):
    return request.param


@pytest.fixture(
    params=[
        "1x0e",
        "1x1o",
        "2x0e + 1x2o",
        "2x0e + 2x2o",
        "1x0e + 1x1o + 1x2e",
        "1x0e + 2x3o",
        "2x2e",
        "1x0e + 1x1o + 2x2e",
        "2x0e + 2x1o + 2x2e + 2x3o",
    ]
)
def j_irreps(request):
    return request.param


@pytest.fixture(params=["ij", "ij=ji"])
def formula(request):
    return request.param


def test_same_irreps(i_irreps, formula):
    fast_rtp = ReducedTensorProducts(formula, i=i_irreps, j=i_irreps)
    slow_rtp = o3.ReducedTensorProducts(formula, i=i_irreps, j=i_irreps)
    assert fast_rtp.irreps_out == slow_rtp.irreps_out
    assert torch.allclose(fast_rtp.change_of_basis, slow_rtp.change_of_basis)


def test_different_irreps(i_irreps, j_irreps):
    fast_rtp = ReducedTensorProducts("ij", i=i_irreps, j=j_irreps)
    slow_rtp = o3.ReducedTensorProducts("ij", i=i_irreps, j=j_irreps)
    assert fast_rtp.irreps_out == slow_rtp.irreps_out
    assert torch.allclose(fast_rtp.change_of_basis, slow_rtp.change_of_basis)
