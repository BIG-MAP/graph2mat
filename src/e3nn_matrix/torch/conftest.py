import pytest

from e3nn import o3
import numpy as np
from scipy.sparse import csr_matrix

from e3nn_matrix.data.basis import PointBasis
from e3nn_matrix.data.configuration import BasisConfiguration

from e3nn_matrix.torch.modules import BasisMatrixReadout

from e3nn_matrix.data.basis import PointBasis
from e3nn_matrix.data.table import BasisTableWithEdges
from e3nn_matrix.data.configuration import BasisConfiguration
from e3nn_matrix.data.processing import MatrixDataProcessor

from e3nn_matrix.torch.data import BasisMatrixTorchData
from e3nn_matrix.torch.modules import BasisMatrixReadout


@pytest.fixture(scope="module", params=[True, False])
def long_A_basis(request):
    return request.param


@pytest.fixture(scope="module")
def ABA_basis_configuration(long_A_basis):
    """Dummy basis configuration with"""

    point_1 = PointBasis("A", "spherical", o3.Irreps("0e"), R=5 if long_A_basis else 2)
    point_2 = PointBasis("B", "spherical", o3.Irreps("1o"), R=5)

    positions = np.array([[0, 0, 0], [3.0, 0, 0], [5.0, 0, 0]])

    basis = [point_1, point_2]

    config = BasisConfiguration(
        point_types=["A", "B", "A"],
        positions=positions,
        basis=basis,
        cell=None,
        pbc=(False, False, False),
    )

    return config
