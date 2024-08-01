# import pytest

# from e3nn import o3
# import numpy as np
# from scipy.sparse import csr_matrix

# from graph2mat.data.basis import PointBasis
# from graph2mat.data.configuration import BasisConfiguration

# from graph2mat.torch.modules import BasisMatrixReadout

# from graph2mat import (
#     PointBasis,
#     BasisTableWithEdges,
#     BasisConfiguration,
#     MatrixDataProcessor,
# )

# from graph2mat.torch.data import TorchBasisMatrixData
# from graph2mat.torch.modules import BasisMatrixReadout


# @pytest.fixture(scope="module", params=["normal", "long_A", "nobasis_A"])
# def basis_type(request):
#     return request.param


# @pytest.fixture(scope="module")
# def ABA_basis_configuration(basis_type):
#     """Dummy basis configuration with"""

#     if basis_type == "nobasis_A":
#         point_1 = PointBasis("A", R=5)
#     else:
#         point_1 = PointBasis(
#             "A",
#             R=5 if basis_type == "long_A" else 2,
#             irreps=o3.Irreps("0e"),
#             basis_convention="spherical",
#         )

#     point_2 = PointBasis("B", R=5, irreps=o3.Irreps("1o"), basis_convention="spherical")

#     positions = np.array([[0, 0, 0], [3.0, 0, 0], [5.0, 0, 0]])

#     basis = [point_1, point_2]

#     config = BasisConfiguration(
#         point_types=["A", "B", "A"],
#         positions=positions,
#         basis=basis,
#         cell=None,
#         pbc=(False, False, False),
#     )

#     return config
