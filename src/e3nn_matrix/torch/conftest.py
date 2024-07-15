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


@pytest.fixture(scope="module", params=["normal", "long_A", "nobasis_A"])
def basis_type(request):
    return request.param


@pytest.fixture(scope="module")
def ABA_basis_configuration(basis_type):
    """Dummy basis configuration with"""

    if basis_type == "nobasis_A":
        point_1 = PointBasis("A", R=5)
    else:
        point_1 = PointBasis(
            "A",
            R=5 if basis_type == "long_A" else 2,
            irreps=o3.Irreps("0e"),
            basis_convention="spherical",
        )

    point_2 = PointBasis("B", R=5, irreps=o3.Irreps("1o"), basis_convention="spherical")

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
